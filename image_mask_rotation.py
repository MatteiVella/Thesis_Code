import copy
import json
import os
import cv2
import math as mt
import numpy as np
import scipy as sc
import ast
import glob

horizontal_images = ['IMG_8688', 'IMG_8745', 'IMG_8747', 'IMG_8751', 'IMG_8753', 'IMG_8757', 'IMG_8760', 'IMG_8761',
                     'IMG_8768', 'IMG_8770', 'IMG_8775', 'IMG_8776']

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate2D(pts, cnt, ang=sc.pi / 4):
    return sc.dot(pts - cnt, sc.array([[sc.cos(ang), sc.sin(ang)], [-sc.sin(ang), sc.cos(ang)]])) + cnt


def rotateImages(degree, read_path, save_path):
    included_extensions = ['JPG']
    file_names = [fn for fn in os.listdir(read_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    for name in file_names:
        path = read_path + '\\' + name
        img = cv2.imread(path)
        r = rotate_image(img, -degree)
        name = name.replace('.JPG', '_'+str(degree)+'.JPG')
        cv2.imwrite(save_path + name, r)


def rotate_boundregions(annotation_path,saving_path, degree_angle):
    f = open(annotation_path)
    data_set = json.load(f)
    items = {}
    new_br = {}
    main_json = {}
    images = []
    alltogether = []
    counter = 0

    # These are just loops in order to build the JSON File
    for img in data_set:
        count = 0
        for anno in data_set[img]:

            for a in anno:

                BR = anno[a].get('BR')
                x = []
                y = []
                xy = []

                no_coords = (len(BR) / 2)

                num = 0
                num2 = 1
                counter = 0

                # HERE WE ARE BUILDING THE BR ARRAY WHICH IS MADE UP OF THIS FORMAT : x1,y1,x2,y2,x3,y3 etc...
                # THIS IS BEING TRANSFORMED INTO THIS FORMAT : (x1,y1),(x2,y2)

                for i in range(int(no_coords)):
                    x.insert(counter, BR[num])
                    y.insert(counter, BR[num2])
                    num = num + 2
                    num2 = num2 + 2
                    counter = counter + 1

                counter = 0
                for z in range(int(no_coords)):
                    temp_x = x[counter]
                    temp_y = y[counter]
                    xy.insert(counter, [temp_x, temp_y])
                    counter = counter + 1

                angle_degrees = degree_angle
                radian = mt.radians(angle_degrees)
                if img in horizontal_images:
                    ots = np.around(rotate2D(xy, sc.array([1632, 1224]), radian))
                else:
                    ots = np.around(rotate2D(xy, sc.array([1224, 1632]), radian))
                ots_tuple = tuple(map(tuple, ots))
                new_br['BR'] = ots.tolist()

                # GETTING THE LIST OF BR'S IN THE CORRECT FORMAT
                str_new_br = str(new_br['BR']).replace('[', '').replace(']', '')
                str_new_br = '['+ str_new_br + ']'
                list_new_br = ast.literal_eval(str_new_br)
                new_br['BR'] = list_new_br
                items[str(a)] = copy.deepcopy(new_br)
                alltogether.insert(count, items)

                count = count+1
                items = {}

        main_json[img+'_'+str(degree_angle)] = alltogether
        alltogether = []

    with open(saving_path+str(degree_angle)+'.json', 'w', encoding='utf-8') as f:
        json.dump(main_json, f, ensure_ascii=False, indent=4)


def mergeAnnotationFiles(annotationspath):
    # Give path of folder where all annotation files are found
    # EXAMPLE : C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset/augmentations/
    result = []
    for f in glob.glob(annotationspath+"*.json"):
        with open(f, "rb") as infile:
            result.append(json.load(infile))

    with open(annotationspath+"annotation_final.json", "w", encoding='utf-8') as outfile:
        json.dump(result, outfile)


# Example of how to use the methods

degree = 15
#for x in range(1, 23):
#    rotateImages(
#        x*degree,
#        r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\resized\combined\Config3\test',
#        r'C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/augmented/Config3/test/'
#    )

#for x in range(1, 23):
#    rotate_boundregions(
#        annotation_path=r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\resized\combined\Config3\test\annotation.json',
#        saving_path='C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/augmented/Config3/test/annotation_rotated_masks_',
#        degree_angle=x*degree
#    )

mergeAnnotationFiles('C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/augmented/Config3/test/')


