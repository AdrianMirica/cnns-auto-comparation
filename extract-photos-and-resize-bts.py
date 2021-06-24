# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:09:24 2021

@author: Adrian
"""

import os
import cv2 as cv
import re
import pandas as pd

csv_train_folder = r"C:\Facultate\Disertatie\Datasets\BTSC\CSV_training"
train_folder = r"C:\Facultate\Disertatie\Datasets\BTSC\Training"
dst_train_folder = r"C:\Facultate\Disertatie\datasets\BTSC\data\32\train"

dst_test_folder = r"C:\Facultate\Disertatie\datasets\BTSC\data\32\test"
test_folder = r"C:\Facultate\Disertatie\datasets\BTSC\Testing"
csv_test_folder = r"C:\Facultate\Disertatie\datasets\BTSC\CSV_testing"

img_size=32

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def extract_data_from_csv(csv_path, dst_folder, train_dir):
    margin = 2 #5 pt 96
    csv_list = sorted_alphanumeric(os.listdir(csv_path))
    print(csv_list)
    print(csv_path)
    for csv in csv_list:
        class_name_from_csv_name = csv[3:8]
        #print(class_name_from_csv_name)
        csv_file_path = os.path.join(csv_path, csv)
        print(csv_file_path)
        rows = pd.read_csv(csv_file_path)
        #print(rows)
        for index, row in rows.iterrows():
            row_data = row['Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId']
            data = row_data.split(";")
            #print(data)
            img_class = data[7]
            img_filename = data[0]
            x1 = int(data[3])
            x2 = int(data[5])
            y1 = int(data[4])
            y2 = int(data[6])
            width = int(data[1])
            height = int(data[2])
            
            train_image_folder_path = os.path.join(train_dir, class_name_from_csv_name)
            src_image_path = os.path.join(train_image_folder_path, img_filename)
            
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


extract_data_from_csv(csv_train_folder, dst_train_folder, train_folder)
extract_data_from_csv(csv_test_folder, dst_test_folder, test_folder)