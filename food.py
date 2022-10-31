
english_lst = ['Pastizz','Imqaret','Gbejniet','Ghaq Tal-Ghasel','Zalzett Malti','Qassatat','Coin']
foods_list = ['Pastizz','Imqaret','Gbejniet','Ghaq Tal-Ghasel','Zalzett Malti','Qassatat','Coin']
food_diction = {'BG': 0,'Pastizz': 1, 'Imqaret': 2, 'Gbejniet': 3, 'Ghaq Tal-Ghasel': 4, 'Zalzett Malti': 5, 'Qassatat': 6, 'Coin': 7}

calorie_per_cm_squared = {'Pastizz': 2.227, 'Imqaret': 2.690, 'Gbejniet': 8.261, 'Ghaq Tal-Ghasel': 3.567, 'Zalzett Malti': 4.914, 'Qassatat': 7.702}
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def get_calorie(class_name, real_food_area):
        return calorie_per_cm_squared[class_name] * real_food_area


##Configurations
class FoodConfig(Config):
    """Configuration for training on the toy  dataset.
    """
    # Training 2 images per GPU as the image size is quite large
    NAME = 'food'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 1 foods

    # Using  smaller anchors because our foods are quite small objects 
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images  have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the dataset is simple and small
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


## Load Dataset
class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, subset):
        """Load a subset of the food dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only n+1 class to add.
        for n, i in enumerate(english_lst):
            self.add_class("food", n + 1, i)

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "annotation.json")))
        # Add images
        for a in annotations:
            polygons = annotations[a]
            image_path = os.path.join(dataset_dir, a + ".jpg")
            image = skimage.io.imread(image_path, plugin='pil')
            height, width = image.shape[:2]

            self.add_image(
                "food",
                image_id=a + ".jpg",  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
               Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "food":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # print(info["height"], info["width"], len(info["polygons"]))
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            p = list(p.values())
            rr, cc = skimage.draw.polygon(p[0]['BR'][1::2], p[0]['BR'][::2])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        items_names = [''.join(key.keys()) for key in info['polygons']]
        item_ids = list(map(lambda x: food_diction[x], items_names))
        return mask.astype(np.bool), np.array(item_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "food":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
