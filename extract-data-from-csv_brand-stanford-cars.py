# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:19:31 2021

@author: Adrian
"""

import pandas as pd
import os
import cv2
from console_progressbar import ProgressBar

def get_brand_class(img_class):
    if img_class == 1:
        return 1
    elif img_class >= 2 and img_class <= 7:
        return 2
    elif img_class >= 8 and img_class <= 11:
        return 3
    elif img_class >= 12 and img_class <= 25:
        return 4
    elif img_class >= 26 and img_class <= 38:
        return 5
    elif img_class >= 39 and img_class <= 44:
        return 6
    elif img_class >= 45 and img_class <= 46:
        return 7
    elif img_class >= 47 and img_class <= 50:
        return 8
    elif img_class >= 51 and img_class <= 53:
        return 9
    elif img_class >= 54 and img_class <= 75:
        return 10
    elif img_class >= 76 and img_class <= 81:
        return 11
    elif img_class == 82:
        return 12
    elif img_class >= 83 and img_class <= 97:
        return 13
    elif img_class == 98:
        return 14
    elif img_class >= 99 and img_class <= 100:
        return 15
    elif img_class >= 101 and img_class <= 104:
        return 16
    elif img_class == 105:
        return 17
    elif img_class >= 106 and img_class <= 117:
        return 18
    elif img_class >= 118 and img_class <= 122:
        return 19
    elif img_class == 123:
        return 20
    elif img_class >= 124 and img_class <= 125:
        return 21
    elif img_class >= 126 and img_class <= 129:
        return 22
    elif img_class >= 130 and img_class <= 140:
        return 23
    elif img_class >= 141 and img_class <= 142:
        return 24
    elif img_class == 143:
        return 25
    elif img_class == 144:
        return 26
    elif img_class >= 145 and img_class <= 149:
        return 27
    elif img_class >= 150 and img_class <= 153:
        return 28
    elif img_class >= 154 and img_class <= 155:
        return 29
    elif img_class == 156:
        return 30
    elif img_class == 157:
        return 31
    elif img_class == 158:
        return 32
    elif img_class == 159:
        return 33
    elif img_class == 160:
        return 34
    elif img_class >= 161 and img_class <= 166:
        return 35
    elif img_class == 167:
        return 36
    elif img_class >= 168 and img_class <= 171:
        return 37
    elif img_class == 172:
        return 38
    elif img_class == 173:
        return 39
    elif img_class == 174:
        return 40
    elif img_class >= 175 and img_class <= 177:
        return 41
    elif img_class == 178:
        return 42
    elif img_class >= 179 and img_class <= 180:
        return 43
    elif img_class >= 181 and img_class <= 184:
        return 44
    elif img_class == 185:
        return 45
    elif img_class >= 186 and img_class <= 189:
        return 46
    elif img_class >= 190 and img_class <=192:
        return 47
    elif img_class >= 193 and img_class <=195:
        return 48
    elif img_class == 196:
        return 49
    
csv_path=r"C:\Facultate\Disertatie\Datasets\Stanford_cars\github-version\cars_data.csv"
images_path = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\car_ims"
dst_folder = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\only_brands\data"
img_size=96


pb = ProgressBar(total=100, prefix='Save data', suffix='', decimals=3, length=50, fill='=')
rows = pd.read_csv(csv_path)
for i, row in rows.iterrows():
    img_class = row["ClassID"]
    test_class = row["TestSet"]
    class_name = row["ClassName"]
    filename = row["filename"]
    bbox_x1 = row["BBOX_X1"]
    bbox_x2 = row["BBOX_X2"]
    bbox_y1 = row["BBOX_Y1"]
    bbox_y2 = row["BBOX_Y2"]
    
    image_path = os.path.join(images_path, filename)
    print(image_path)
    
    src_image = cv2.imread(image_path)
    height_from_image, width_from_image = src_image.shape[:2]
    
    margin = 8
    
    x1 = max(0, bbox_x1 - margin)
    y1 = max(0, bbox_y1 - margin)
    x2 = min(bbox_x2 + margin, width_from_image)
    y2 = min(bbox_y2 + margin, height_from_image)
    
    #dst_path = os.path.join(dst_folder, str(img_class)) 
    brand_class = get_brand_class(img_class)
    
    dst_path = os.path.join(dst_folder, str(brand_class))
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    dst_path = os.path.join(dst_path, filename)
    
    crop_image = src_image[x1:x2, y2:y1]
    dst_img = cv2.resize(src=crop_image, dsize=(img_size, img_size))
    cv2.imwrite(dst_path, dst_img)
    
    pb.print_progress_bar((i + 1) * 100 / 16186)