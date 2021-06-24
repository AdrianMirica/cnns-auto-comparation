# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:29:05 2021

@author: Adrian
"""

import numpy as np
import os
import cv2
from numpy import save
import re

dataset_path_train = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\only_brands\data"


#labels.append('%01d' % (class_id,))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def load_data(dataset_path):
    images = []
    classes = []
    classes_list_dir = sorted_alphanumeric(os.listdir(dataset_path))
    for class_name in classes_list_dir:
        train_folder_path = os.path.join(dataset_path,class_name)
        print(train_folder_path)
        class_int = int(class_name)
        
        class_folder_images = sorted_alphanumeric(os.listdir(train_folder_path)) 
        for image_name  in class_folder_images:
            image_path = os.path.join(train_folder_path, image_name)
            print(image_path)
            image=cv2.imread(image_path)
            images.append(image)
            classes.append(class_int)
            
    class_count = np.max(classes)+1
    print('The number of different classes is %d' % class_count) 
    
    X=np.array(images)
    y=np.array(classes)
    return (X,y)


x, y = load_data(dataset_path_train)
print("Done extracting train data")
#x_test, y_test = load_data(dataset_path_test)
#rint("Done extracting test data")

save('x.npy', x)

save('y.npy', y)

print("Done!")
  
