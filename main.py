import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras
import random
import requests
from PIL import Image
from tqdm import tqdm

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


def le_net_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

##TRAIN##
trainpath ='C:\\Users\\amara\\Downloads\\bdd100k_images_100k\\bdd100k\\images\\100k\\train10'
images=os.listdir(trainpath)
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


# ##LABELS##
# ##TRAIN##
train_labels= pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_train.json')
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

###Data Processing#####

labelslist = list(datatest.columns.values)
labelslist = labelslist[3:]
print(labelslist)

num = []
for ll in labelslist:
    num.append((ll, datatest[ll].sum()))
stats = pd.DataFrame(num, columns=['Labels', 'Occurrence'])
print("1")
#stats = stats.sort_values(['Occurrence']).reset_index(drop=True)
print(stats)
rowSums = datatest.iloc[:, 3:].sum(axis=1)
multiLl_counts = rowSums.value_counts()
multiLl_counts = multiLl_counts.iloc[1:]
label_count = pd.DataFrame(multiLl_counts, columns=['Total # of images']).rename_axis('# of Labels', axis=1)
print(label_count)

IMG_SIZE = (86,86)

train_img_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
trainSet = train_img_datagen.flow_from_dataframe(datatest, directory=trainpath, x_col ='name', y_col = updlabels, class_mode = "multi_output", seed= 24, subset = 'training', batch_size=55888, target_size = IMG_SIZE)

train_images, train_labels = next(trainSet)
train_labels_ = np.array(train_labels).reshape(len(train_labels), -1).T
train_img = train_images.reshape(train_images.shape[0], -1)
m_train = train_img.shape[0]

print ("Number of training samples: " + str(m_train))
print ("train_images shape: " + str(train_images.shape))
print ("train_img shape: " + str(train_img.shape))
print ("train_labels_ shape: " + str(train_labels_.shape))
model = modified_model()
print(model.summary())

# model_history = model.fit(train_img,train_labels_,epochs=15, batch_size=64,
#                           validation_data=(val_img, val_labels_))
# #model_report function to retrieve metrics
# model_report(model, model_history, train_img, train_labels_, test_img, test_labels_)
# val_labels = pd.read_json('C:\\Users\\amara\\Downloads\\labels\\bdd100k\\labels\\bdd100k_labels_images_val.json')
# #print(val_labels.head())


##CODE FOR SPLITTING THE TRAIN JSON INTO TEST##
# with open(
#         'C:\\Users\\smell\\Documents\\smarttech\\bdd100k_images_100k\\images\\100k\\bdd100k\\labels\\bdd100k_labels_images_train.json') as f:
#     data = json.load(f)
#     for i, x in enumerate(data):
#         with open(str(i) + '.json', 'w') as f_out:
#             json.dump(x, f_out)
#             #                print(data)
#             train_lables = data
#
# test_labels = train_test_split(train_labels, test_size=0.3)

##Validation##

data_val = pd.read_json('validation.json')
#print(data_val)
#data_val.info()

data_val['updated_labels'] = data_val['labels'].map(newlabels)
#print(data_val)

category_list = ",".join(data_val.updated_labels).split(",")
category_list = list(set(category_list))
#print(category_list)
#
data_validation = data_val.copy()
for target in category_list:
    data_validation[target] = data_validation['updated_labels'].str.contains(target)
    data_validation[target] = data_validation[target].astype(int)
    #print(data_validation)

    labels = list(data_validation.columns.values)
    labels = labels[3:]
    #print(labels)

    counts = []
    for label in labels:
        counts.append((label, data_validation[label].sum()))
    dv_results= pd.DataFrame(counts, columns=['Labels', 'Appearance'])
    #print(dv_results)

    rowSums = data_validation.iloc[:, 3:].sum(axis=1)
    multiLabels = rowSums.value_counts()
    multiLabels = multiLabels.iloc[1:]
    label_count = pd.DataFrame(multiLabels, columns=['Total # of images']).rename_axis('# of Labels', axis=1)
    #print(label_count)


## BOX2D attempt crop##
# for item in data:
#     for data_item in item['labels']:
#         data = (data_item['category'])
#         box = (data_item['box2d'])
#         if 'car' in data :
#             a = item.get('name')
#             print(a)
#
#
# for box2d in box:
#     x1 = int(box2d['x1'])
#     y1 = int(box2d['y1'])
#     x2 = int(box2d['x2'])
#     y2 = int(box2d['y2'])
#     cropped_image = img[y1:y1 + y2, x1:x1 + x2]
#     x = np.append(x, preprocess(cropped_image))
#
#



