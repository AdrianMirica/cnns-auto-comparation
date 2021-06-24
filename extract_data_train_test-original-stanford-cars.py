# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:59:07 2021

@author: Adrian
"""

import numpy as np
import os
import cv2
from numpy import save
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def convert_images_to_data(imageset_path):
    images = []
    #index = 0;
    classes = []
    classes_list_dir = sorted_alphanumeric(os.listdir(imageset_path))
    sorted(classes_list_dir)
    for class_name in classes_list_dir:
        #print("class_name: " + str(class_name))
        img_class_folder = os.path.join(imageset_path, class_name)
        #print(img_class_folder)
        class_folder_images = sorted_alphanumeric(os.listdir(img_class_folder))
        for image_name in class_folder_images:
            image_path = os.path.join(img_class_folder, image_name)
            #print(image_path)
            image=cv2.imread(image_path)
            images.append(image)
            classes.append(int(class_name) - 1)
            
        #index = index + 1;
        #if index == 30:
            #break
            
    X=np.array(images)
    y=np.array(classes)
    return (X,y)

train_images_path = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\data\train"
test_images_path = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\data\test"

x_train, y_train = convert_images_to_data(train_images_path)
x_test, y_test = convert_images_to_data(test_images_path)

save('x_train.npy', x_train)
save('x_test.npy', x_test)

save('y_train.npy', y_train)
save('y_test.npy', y_test)

print("Done!")