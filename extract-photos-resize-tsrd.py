# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:36:04 2021

@author: Adrian
"""

import os
import cv2 as cv

train_anno_file = r"C:\Facultate\Disertatie\Datasets\TSRD\TsignRecgTrain4170Annotation.txt"
train_images_path = r"C:\Facultate\Disertatie\Datasets\TSRD\TSRD-Train"
train_data = r"C:\Facultate\Disertatie\Datasets\TSRD\data\32\train"

test_anno_file = r"C:\Facultate\Disertatie\Datasets\TSRD\TsignRecgTest1994Annotation.txt"
test_data = r"C:\Facultate\Disertatie\Datasets\TSRD\data\32\test"
test_images_path = r"C:\Facultate\Disertatie\Datasets\TSRD\TSRD-Test"

def extract_data_from_file(file_path, train_dir, dst_folder):
    lines = []
    margin = 2 # 5 pentru 96
    img_size = 32
    with open(file_path) as f:
        lines = f.readlines()
    
    for line in lines:
        data=line.split(";")
        print(data)
        img_filename = data[0]
        width = int(data[1])
        height = int(data[2])
        x1 = int(data[3])
        x2 = int(data[5])
        y1 = int(data[4])
        y2 = int(data[6])
        img_class = data[7]
        
        
        src_image_path = os.path.join(train_dir, img_filename)
        print(src_image_path)
        
        src_image = cv.imread(src_image_path)
        height_from_image, width_from_image = src_image.shape[:2]
            
        if height_from_image != height and width_from_image != width:
            height = height_from_image
            width = width_from_image
            
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        
        dst_path = os.path.join(dst_folder, img_class)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, img_filename)
        
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_size, img_size))
        cv.imwrite(dst_path, dst_img)
        
        
extract_data_from_file(train_anno_file, train_images_path, train_data)
extract_data_from_file(test_anno_file, test_images_path, test_data)