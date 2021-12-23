import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras
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


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def newlabels(labels):
  updlabels = pd.DataFrame.from_records(labels)
  updlabels = updlabels['category'].unique().tolist()
  return ','.join(updlabels)


##TRAIN##
trainpath ='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\train10'
images=os.listdir(trainpath)
type(images)
len(images)
img_size = 224

img_data_val=[]
for img in images:
    for i in range(len(img_data_val)):
        plt.imshow(img_data_val[i])
        img = preprocess(img_data_val[i])
        plt.imshow(img)
        img = cv2.resize(img, (img_size, img_size))
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.axis("off")
        #plt.show()

##TEST##
testpath='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\test10'
images=os.listdir(testpath)
type(images)
len(images)
img_size = 74

img_data_val=[]
for img in images:
    for i in range(len(img_data_val)):
        plt.imshow(img_data_val[i])
        img = preprocess(img_data_val[i])
        plt.imshow(img)
        img = cv2.resize(img, (img_size, img_size))
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.axis("off")
        #plt.show()

##VALIDATION##
valpath='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\val10'
images=os.listdir(valpath)
type(images)
len(images)
img_size = 224

img_data_val=[]
for img in images:
    for i in range(len(img_data_val)):
        plt.imshow(img_data_val[i])
        img = preprocess(img_data_val[i])
        plt.imshow(img)
        img = cv2.resize(img, (img_size, img_size))
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.axis("off")
        #plt.show()




##LABELS##
##TRAIN##
train_labels= pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_train.json')
#print(train_labels.head())
#print(train_labels.labels[0])
train_labels['newlabels'] = train_labels['labels'].map(newlabels)
#print(train_labels)
updlabels = ",".join(train_labels.newlabels).split(",")
updlabels = list(set(updlabels))
# for upd in updlabels:
#     if 'drivable area' in upd:
#         del upd ['drivable area']
#         print ('success')
#     elif 'lane' in upd:
#         del upd['lane']
#         print ('success')
#print(updlabels)
datatest = train_labels.copy()
for updated in updlabels:
    datatest[updated] = datatest['newlabels'].str.contains(updated)
    datatest[updated] = datatest[updated].astype(int)

print("1")
labels = list(datatest.columns.values)
labels = labels[3:]
print(labels)
counts = []
for label in labels:
    counts.append((label, datatest[label].sum()))
info = pd.DataFrame(counts, columns=['Labels', 'Occurrence'])
info = info.sort_values(['Occurrence']).reset_index(drop=True)

sns.set_style("darkgrid")
plt.figure(figsize=(12,10))
order= ['train','motor','rider','bike','bus', 'truck', 'person','traffic light',
        'traffic sign', 'lane','drivable area', 'car']
ax= sns.barplot(labels, datatest.iloc[:,3:].sum().values, order= order)

plt.title("Image Classification Labels", fontsize=22)
plt.ylabel('Total Occurrences', fontsize=20)
plt.xlabel('Labels', fontsize=20)

#adding the text labels
rects = ax.patches
labels = info['Occurrence']
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=16)
plt.xticks(rotation=45, fontsize=16, ha='right')
plt.yticks(fontsize=16)
plt.show()

val_labels = pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_val.json')
#print(val_labels.head())

###Data Processing#####



# filename='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\car'
# data =[]
# data = pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_val.json')
# names = []
#
# for item in data["name"]:
#     names.append(item)
#     print(names)
# #print(data)
#
# # with open ('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_val.json') as f:
# #     data = json.load(f)
# #
# #    # print (data)
# #
# #
# # li = [item.get('labels') for item in data]
# # #print(li)
# # items = json.dumps(li)
# # #print(type(items))
# #
# # for item in data:
# #     for data_item in item['labels']:
# #         data = (data_item['category'])
# #         box = (data_item['box2d'])
# #         if 'car' in data :
# #             a = item.get('name')
# #             print(a)
# #
# # # createFolder('./car/')
# # # cv2.imwrite('./car/', a)
# #
# # for box2d in box:
# #     x1 = int(box2d['x1'])
# #     y1 = int(box2d['y1'])
# #     x2 = int(box2d['x2'])
# #     y2 = int(box2d['y2'])
# #     cropped_image = img[y1:y1 + y2, x1:x1 + x2]
# #     x = np.append(x, preprocess(cropped_image))
# #
# #
#
# #
# # for item in data:
# #     for data_item in item['labels']:
# #         print (data_item['id'])
#
#
#
