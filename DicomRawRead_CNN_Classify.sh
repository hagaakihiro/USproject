#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
#from keras.optimizers import Adam, SGD, RMSprop
#import keras.backend as K
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from unet2thin import UNet
#from unet import UNet
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img

import sklearn.model_selection as crv
import pandas as pd
import math
import random

import os.path
import matplotlib.pyplot as plt
#from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

import dicom
import pylab
import glob
from DicomRawRead import load_DICOM_image, load_Struct, load_One_Simple_Raw_image
import itertools


# ニューラルネットワークのモデルを定義
def CNN_model(num_class,input_s):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_s))
    model.add(Activation('relu'))
#    model.add(Conv2D(32, (3, 3)))#
#    model.add(Activation('relu'))#
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
#    model.add(Conv2D(64, (3, 3)))#
#    model.add(Activation('relu'))#
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
#    model.add(Conv2D(64, (3, 3)))#
#    model.add(Activation('relu'))#
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    # コンパイル

    return model
#################################



## Loading training/test data
colsize = 120
rowsize = 120
bb = "float64"
num_s = 50
num_test = 1
list_4 = ["s","2","3","4"]
num_class = len(list_4)
re_t = np.zeros(num_class*num_class,dtype ='int')
for ip in range(1):
#for ip in range(num_s):
    foldername = "20180913_mean/"
    for aa in range(1,2):
        if aa == 0:
            hh = "all_"
        elif aa == 1:
            hh = "10s_"
        X = []
        Y = []
        its_num = 0
        for its in list_4:
            filename_mean = foldername + hh + 'ave_trainimage_%03d_%s.raw' % (ip,its)
            filename_sd = foldername + hh + 'sd_trainimage_%03d_%s.raw' % (ip,its)
            mean_img = load_One_Simple_Raw_image(filename_mean, colsize, rowsize, bb)
            sd_img = load_One_Simple_Raw_image(filename_sd, colsize, rowsize, bb)
            im = np.reshape(mean_img,(colsize, rowsize))
#            plt.imshow(im)
#            plt.show()
            for itt in range(num_s-1):
                filename_mean = foldername + hh + 'ave_trainimage_%03d_%03d_%s.raw' % (ip,itt,its)
                filename_sd = foldername + hh + 'sd_trainimage_%03d_%03d_%s.raw' % (ip,itt,its)
                mean_img = load_One_Simple_Raw_image(filename_mean, colsize, rowsize, bb)
                sd_img = load_One_Simple_Raw_image(filename_sd, colsize, rowsize, bb)
                im = np.reshape(mean_img,(colsize, rowsize))
                img = img_to_array(im)
                X.append(img)
                Y.append(its_num)
            its_num += 1
#                plt.imshow(img)
#                plt.show()
        X_train = np.asarray(X)
        Y_train = np.asarray(Y)
        print(Y)
        Y_train = np_utils.to_categorical(Y_train, num_class)

        X_test = []
        Y_test = []
        its_num = 0
        for its in list_4:
            filename_mean = foldername + hh + 'ave_testimage_%03d_%s.raw' % (ip,its)
            filename_sd = foldername + hh + 'sd_testimage_%03d_%s.raw' % (ip,its)
            mean_img = load_One_Simple_Raw_image(filename_mean, colsize, rowsize, bb)
            sd_img = load_One_Simple_Raw_image(filename_sd, colsize, rowsize, bb)
            im = np.reshape(mean_img,(colsize, rowsize))
            img = img_to_array(im)
            X_test.append(img)
            Y_test.append(its_num)
            its_num += 1
        X_test = np.asarray(X_test)
        Y_test = np.asarray(Y_test)
        Y_test = np_utils.to_categorical(Y_test, num_class)
        print(Y_test)
        shape = (colsize, rowsize, 1)
        model = CNN_model(num_class, shape)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=5, epochs=1, validation_data = (X_test, Y_test), verbose = 1)


# To predict test data
        p_table = model.predict(X_test, batch_size=1, verbose=0)
        print(p_table)
        seq = [x for x in range(num_class)]
        permu = list(itertools.permutations(seq))

        re_table = np.reshape(re_t, (num_class, num_class))*0
        for i in range(num_test):
            max_sum = 0.0
            for vect in permu:
                sum_val = p_table[4*i,vect[0]]*p_table[4*i+1,vect[1]]*p_table[4*i+2,vect[2]]*p_table[4*i+3,vect[3]]
                if sum_val > max_sum:
                    max_sum = sum_val
                    max_vec = vect
            re_table[0,max_vec[0]] += 1
            re_table[1,max_vec[1]] += 1
            re_table[2,max_vec[2]] += 1
            re_table[3,max_vec[3]] += 1
            print(max_vec, max_sum)
        print(re_table)

        predict_classes = model.predict_classes(X_test)
        print(predict_classes)
        table = np.reshape(re_t, (num_class, num_class))*0
        table[0,predict_classes[0]] = 1
        table[1,predict_classes[1]] = 1
        table[2,predict_classes[2]] = 1
        table[3,predict_classes[3]] = 1
        print(table)

        if ip == 0:
            if aa == 0:
                a_all_table = table
                a_all_re_table = re_table
            elif aa == 1:
                all_table = table
                all_re_table = re_table
        else:
            if aa == 0:
                a_all_table += table
                a_all_re_table += re_table
            elif aa == 1:
                all_table += table
                all_re_table += re_table

#print("all")
#print(a_all_table)
#print(a_all_re_table)

print("10s")
print(all_table)
print(all_re_table)


sys.exit()




# X_train has not been normalized as [0,1]

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)



#estimator = KerasRegressor(build_fn=CNN_model, epochs=3, batch_size=5, verbose=1)
#estimator.fit(X_train, y_train)
"""
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()
"""

