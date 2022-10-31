import sys

from numpy import expand_dims, os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img
from matplotlib import pyplot

for x in range(1, 42):

    ROOT_DIR = os.path.abspath("../../")
    sys.path.append(ROOT_DIR)
    PASTIZZI_DIR = os.path.join(ROOT_DIR, "datasets\\Pastizzi_Dataset\\ReSized_JPEG_Images\\")
    AUGMENTED_DIR = os.path.join(ROOT_DIR, "datasets\\Pastizzi_Dataset\\Augmented_Images\\Pastizzi_Images_")

    # Giving the correct path depending on the current iteration
    if x < 10:
        PASTIZZI_DIR = PASTIZZI_DIR + "Pastizzi_Images_0" + str(x) + ".jpg"
    else:
        PASTIZZI_DIR = PASTIZZI_DIR + "Pastizzi_Images_" + str(x) + ".jpg"

    # load the image
    img = load_img(PASTIZZI_DIR)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 brightness_range=[0.3, 2.0])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(10):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        save_img(AUGMENTED_DIR + str(x) + str(i) + '.jpg', image, data_format=None, file_format='JPEG', scale=True)
