import json
import random
import cv2
import numpy as np
import scipy as sc
import math as mt
from PIL import Image, ImageDraw


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate2D(pts, cnt, ang=sc.pi / 4):
    return sc.dot(pts - cnt, sc.array([[sc.cos(ang), sc.sin(ang)], [-sc.sin(ang), sc.cos(ang)]])) + cnt



image = cv2.imread(
    r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\augmented\Config1\train\IMG_8755_15.jpg')
r = rotate_image(image, 0)
cv2.imwrite('Mask_RCNN/Test.jpg', r)

# 1.) HERE WE ARE GETTING A BR VALUE FROM THE EXPORTED COCO VALUES AND TRANSFORMING THEM INTO A DRAWABLE POLYGON

BR = [
    2003.0, 1854.0, 2001.0, 1861.0, 1999.0, 1867.0, 1995.0, 1873.0,
    1992.0, 1879.0, 1988.0, 1884.0, 1983.0, 1889.0, 1978.0, 1893.0,
    1973.0, 1897.0, 1967.0, 1901.0, 1961.0, 1904.0, 1955.0, 1907.0,
    1949.0, 1909.0, 1942.0, 1910.0, 1935.0, 1911.0, 1929.0, 1911.0,
    1922.0, 1911.0, 1915.0, 1910.0, 1909.0, 1909.0, 1902.0, 1907.0,
    1896.0, 1904.0, 1890.0, 1901.0, 1884.0, 1897.0, 1879.0, 1893.0,
    1874.0, 1889.0, 1870.0, 1884.0, 1865.0, 1879.0, 1862.0, 1873.0,
    1859.0, 1867.0, 1856.0, 1861.0, 1854.0, 1854.0, 1853.0, 1848.0,
    1852.0, 1841.0, 1852.0, 1834.0, 1852.0, 1828.0, 1853.0, 1821.0,
    1854.0, 1814.0, 1856.0, 1808.0, 1859.0, 1802.0, 1862.0, 1796.0,
    1865.0, 1790.0, 1870.0, 1785.0, 1874.0, 1780.0, 1879.0, 1775.0,
    1884.0, 1771.0, 1890.0, 1767.0, 1896.0, 1764.0, 1902.0, 1762.0,
    1909.0, 1760.0, 1915.0, 1758.0, 1922.0, 1757.0, 1929.0, 1757.0,
    1935.0, 1757.0, 1942.0, 1758.0, 1949.0, 1760.0, 1955.0, 1762.0,
    1961.0, 1764.0, 1967.0, 1767.0, 1973.0, 1771.0, 1978.0, 1775.0,
    1983.0, 1780.0, 1988.0, 1785.0, 1992.0, 1790.0, 1995.0, 1796.0,
    1999.0, 1802.0, 2001.0, 1808.0, 2003.0, 1814.0, 2005.0, 1821.0,
    2005.0, 1828.0, 2006.0, 1834.0, 2005.0, 1841.0, 2005.0, 1848.0
        ]

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

image_path = r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\food_mask\Mask_RCNN\Test.jpg'
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
colors = ["red", "green", "blue", "yellow",
          "purple", "orange"]

# 1.) HERE WE CHECKING IF THE IMAGE IS LANDSCAPE OR PORTRAIT
# 2.) THEN WE USE THE ROTATE2D METHOD TO GET THE ROTATED VALUES
# 3.) FINALLY THE RETURN VALUE IS CONVERTED TO A TUPLE.

angle_degrees = 0
radian = mt.radians(angle_degrees)

w, h = image.size
if (w < h):
    ots = rotate2D(xy, sc.array([1224, 1632]), radian)
else:
    ots = rotate2D(xy, sc.array([1632, 1224]), radian)

ots_tuple = tuple(map(tuple, np.around(ots)))
draw.polygon(ots_tuple, fill=random.choice(colors))
image.show()
