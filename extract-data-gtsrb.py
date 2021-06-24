# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:04:48 2021

@author: Adrian
"""

import pandas as pd
import numpy as np
import os
import cv2
from numpy import save

dataset_path=r"C:\Facultate\Disertatie\Datasets\GTSRB"

def load_data(dataset):
    images = []
    classes = []
    img_size = 96
    rows = pd.read_csv(dataset)
    #rows = rows.sample(frac=1).reset_index(drop=True)
    
    for i, row in rows.iterrows():
        img_class = row["ClassId"]
        img_path = row["Path"]
        image = os.path.join(dataset_path, img_path)
        
        image = cv2.imread(image)
        image_rs = cv2.resize(image, (img_size, img_size), 3)
        
        R, G, B = cv2.split(image_rs)
        
        img_r = cv2.equalizeHist(R)
        img_g = cv2.equalizeHist(G)
        img_b = cv2.equalizeHist(B)
        
        new_image = cv2.merge((img_r, img_g, img_b))
        
        if i % 500 == 0:
            print(f"loaded: {i}")
        images.append(new_image)
        classes.append(img_class)
        
    X = np.array(images)
    y = np.array(classes)
    return (X, y)
train_data = r"C:\Facultate\Disertatie\datasets\GTSRB\Train.csv"
test_data = r"C:\Facultate\Disertatie\datasets\GTSRB\Test.csv"

(x_train, y_train) = load_data(train_data)
(x_test, y_test) = load_data(test_data)

save('x_train_96.npy', x_train)
save('x_test_96.npy', x_test)

save('y_train_96.npy', y_train)
save('y_test_96.npy', y_test)

print("UPDATE: Normalizing data")

# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0
# print("UPDATE: One-Hot Encoding data")
# num_labels = len(np.unique(y_train))
# y_train = keras.utils.to_categorical(y_train, num_labels)
# y_test = keras.utils.to_categorical(y_test, num_labels)
# class_totals = y_train.sum(axis=0)
# class_weight = class_totals.max() / class_totals