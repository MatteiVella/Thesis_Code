# Importing Image class from PIL module
import sys
from PIL import Image

import cv2
import os

relevant_path = r"C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\original\test\TopView"
included_extensions = ['JPG']
print(os.listdir(relevant_path))
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

for name in file_names:

    img = cv2.imread('C://Users//matte//OneDrive//Desktop//Thesis//Thesis_Code//datasets//MalteseFood_Dataset_Final//original//test//TopView//'
                     + name)
    scale_percent = 0.81
    if img.shape[0] > img.shape[1]:
        dim_size = (round(img.shape[1]*scale_percent-1), round(img.shape[0]*scale_percent-2) )
        new_img = cv2.resize(img, dim_size, interpolation=cv2.INTER_AREA)
    else:
        dim_size = (round(img.shape[1] * scale_percent - 2), round(img.shape[0] * scale_percent - 1))
        new_img = cv2.resize(img, dim_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite('C://Users//matte//OneDrive//Desktop//Thesis//Thesis_Code//datasets//MalteseFood_Dataset_Final//resized//test//TopView//'+name, new_img)