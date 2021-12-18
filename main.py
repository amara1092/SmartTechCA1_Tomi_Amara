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
path='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\train10'
images=os.listdir(path)
type(images)
len(images)

img_data_train=[]
for img in images:
    img_arr=cv2.imread(os.path.join(path, img))
    img_data_train.append(img_arr)

for i in range(len(img_data_train)):
    plt.imshow(img_data_train[i])
    #plt.show()

##TEST##
#path
path='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\test10'
images=os.listdir(path)
type(images)
len(images)

img_data_test=[]
for img in images:
    img_arr=cv2.imread(os.path.join(path,img))
    img_data_test.append(img_arr)

for i in range(len(img_data_test)):
    plt.imshow(img_data_test[i])
    #plt.show()

##VALIDATION##
path='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\val10'
images=os.listdir(path)
type(images)
len(images)

img_data_val=[]
for img in images:
    img_arr=cv2.imread(os.path.join(path,img))
    img_data_val.append(img_arr)

for i in range(len(img_data_val)):
    plt.imshow(img_data_val[i])
    img = preprocess(img_data_val[i])
    plt.imshow(img)
    plt.axis("off")
   # plt.show()

data = pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_val.json')
print(data)
