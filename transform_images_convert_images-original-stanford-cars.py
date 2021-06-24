# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:47:23 2021

@author: Adrian
"""

import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
from console_progressbar import ProgressBar


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_train_data(fnames, labels, bboxes):
    src_folder =r"C:\Facultate\Disertatie\Datasets\Stanford_cars\cars_train"
    num_samples = len(fnames)

    #train_split = 0.8
    #num_train = int(round(num_samples * train_split))
    #train_indexes = random.sample(range(num_samples), num_train)

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 8 pixels
        margin = 8
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        #if i in train_indexes: pt set cu validare
        #if i in num_samples:
        dst_folder = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\data\train"
        #else:
            #dst_folder = "C:\Facultate\Disertatie\datasets\standford_cars\data\valid"

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


def save_test_data(fnames, labels, bboxes):
    src_folder = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\cars_test"
    dst_folder = r"C:\Facultate\Disertatie\Datasets\Stanford_cars\data\test"
    num_samples = len(fnames)

    pb = ProgressBar(total=100, prefix='Save test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]
        
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 8 pixels
        margin = 8
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)
        
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


def process_train_data():
    print("Processing train data...")
    cars_annos = scipy.io.loadmat("C:\Facultate\Disertatie\Datasets\Stanford_cars\car_devkit\devkit\cars_train_annos.mat")
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    class_ids = []
    bboxes = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        labels.append('%01d' % (class_id,))
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        class_ids.append(class_id)
        fnames.append(fname)

    labels_count = np.unique(class_ids).shape[0]
    print(np.unique(class_ids))
    print('The number of different cars is %d' % labels_count)

    save_train_data(fnames, labels, bboxes)


def process_test_data():
    print("Processing test data...")
    cars_annos_with_labels = scipy.io.loadmat("C:\Facultate\Disertatie\Datasets\Stanford_cars\cars_test_annos_withlabels.mat")
    annotations = cars_annos_with_labels['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []
    class_ids = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        labels.append('%01d' % (class_id,))
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        class_ids.append(class_id)
        fnames.append(fname)
        
    labels_count = np.unique(class_ids).shape[0]
    print(np.unique(class_ids))
    print('The number of different cars is %d' % labels_count)    

    save_test_data(fnames, labels, bboxes)


if __name__ == '__main__':
    # parameters
    img_width, img_height = 96, 96

    print('Extracting cars_train.tgz...')
    if not os.path.exists('cars_train'):
        with tarfile.open("C:\Facultate\Disertatie\Datasets\Stanford_cars\cars_train.tgz", "r:gz") as tar:
            tar.extractall()
    print('Extracting cars_test.tgz...')
    if not os.path.exists('cars_test'):
        with tarfile.open("C:\Facultate\Disertatie\Datasets\Stanford_cars\cars_test.tgz", "r:gz") as tar:
            tar.extractall()
    print('Extracting car_devkit.tgz...')
    if not os.path.exists('devkit'):
        with tarfile.open("C:\Facultate\Disertatie\Datasets\Stanford_cars\car_devkit.tgz", "r:gz") as tar:
            tar.extractall()

    cars_meta = scipy.io.loadmat("C:\Facultate\Disertatie\Datasets\Stanford_cars\devkit\cars_meta.mat")
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    print('class_names.shape: ' + str(class_names.shape))
    print('Sample class_name: [{}]'.format(class_names[8][0][0]))
    
    # from numpy import save
    # save("cars_class_names.npy", class_names)

    ensure_folder(r"C:\Facultate\Disertatie\datasets\standford_cars\data\train")
    #ensure_folder(r"C:\Facultate\Disertatie\datasets\standford_cars\data\valid")
    ensure_folder(r"C:\Facultate\Disertatie\datasets\standford_cars\data\test")

    process_train_data()
    process_test_data()

print("Done!")