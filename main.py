import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import pandas as pd
import cv2
import os, os.path
import json


def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


def preprocess(img):
    img = gray_scale(img)
    img = equalise(img)
    img = img/255
    return img

##TRAIN##
path='C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\test'
# images=os.listdir(path)
# type(images)
# len(images)
#
# img_data_train=[]
# for img in images:
#     img_arr=cv2.imread(os.path.join(path, img))
#     img_data_train.append(img_arr)
#
# for i in range(len(img_data_train)):
#     plt.imshow(img_data_train[i])
#     #plt.imshow(img, cmap=plt.get_cmap('gray'))
#     #plt.show()
#
# ##TEST##
# #path
# path='C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\train'
# images=os.listdir(path)
# type(images)
# len(images)
#
# img_data_test=[]
# for img in images:
#     img_arr=cv2.imread(os.path.join(path,img))
#     img_data_test.append(img_arr)
#
# for i in range(len(img_data_test)):
#     plt.imshow(img_data_test[i])
#     #plt.imshow(img, cmap=plt.get_cmap('gray'))
#     #plt.show()
#
# ##VALIDATION##
# path='C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\val'
# images=os.listdir(path)
# type(images)
# len(images)
#
# img_data_val=[]
# for img in images:
#     img_arr=cv2.imread(os.path.join(path,img))
#     img_data_val.append(img_arr)
#
# for i in range(len(img_data_val)):
#     plt.imshow(img_data_val[i])
#     img = preprocess(img_data_val[i])
#    # plt.imshow(img, cmap=plt.get_cmap('gray'))
#     plt.axis("off")
#     #plt.show()
data =[]
data = pd.read_json('C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\bdd100k\\labels\\bdd100k_labels_images_val.json')
#print(data)

with open ('C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\bdd100k\\labels\\bdd100k_labels_images_val.json') as f:
    data = json.load(f)
   # print (data)

li = [item.get('labels') for item in data]
print(li)

for item in data:
    for data_item in item['labels']:
        print (data_item['category'])

for item in data:
    for data_item in item['labels']:
        print (data_item['id'])






