#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Input, BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, Adamax, Nadam, SGD, RMSprop
from keras.initializers import he_normal

from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

import sklearn.model_selection as crv
import pandas as pd
import math
import random

import os.path
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.layers.merge import Multiply

import sys
import csv

import dicom
import pylab
import glob
from DicomRawRead import load_DICOM_image, load_Struct, load_One_Simple_Raw_image, load_One_Simple_Raw_image_3D
import itertools
import h5py

# ニューラルネットワークのモデルを定義
def CNN_model2(num_class,input_s):
    model = Sequential()

    model.add(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_s,kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv3D(64, (3, 3, 3), padding='same',kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv3D(64, (1, 3, 3), padding='same',kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv3D(64, (1, 3, 3), padding='same',kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv3D(32, (1, 3, 3), padding='same',kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    return model
#################################
def loading_usimage(itest,mode,list_Asy,ShiftData,list_foldername,ShiftData2,list_foldername2,ShiftData3,list_foldername3,list_N,list_LAD,list_LCX,list_RCA,list_AI,list_AI_LAD,list_AI_LCX,list_AI_RCA,list_4,list_3,colsize, rowsize, height, bb):
    
    # Removed data for Data2 database
    rem_list_Data2_LAD_s = [36]
    rem_list_Data2_RCA_s = [18,20,23,27]

    rem_list_Data2_N_2 = [20,49]
    rem_list_Data2_LCX_2 = [50,51]
    rem_list_Data2_RCA_2 = [19]

    rem_list_Data2_N_3 = [5,23,28]
    rem_list_Data2_LCX_3 = [18,41]
    rem_list_Data2_RCA_3 = [17]

    rem_list_Data2_N_4 = [24,28,43]
    rem_list_Data2_LAD_4 = [21]
    rem_list_Data2_LCX_4 = [6]
    ###
    
    X = []
    Y = []
    XT = []
    YT = []
    XV = []
    YV = []

    for iAsy in range(len(list_Asy)):
        Shift = pd.read_csv(ShiftData[iAsy], encoding="SHIFT-JIS")
        Shift.columns = ["file","machine","mode","4","5","6","7","8","9","10","11"]
        modeshift = Shift[Shift["mode"] == mode]
        num_s = len(modeshift.iloc[:,0])
        # For 2 classes 
        if iAsy == 0:
            cl_num = 0
        else:
            cl_num = 1
        
        for ip in range(num_s):
            for ic in range(modeshift.iloc[ip,10]):
                if iAsy == 0:
                    listt = list_N
                elif iAsy == 1:
                    listt = list_LAD
                elif iAsy == 2:
                    listt = list_LCX
                elif iAsy == 3:
                    listt = list_RCA
                for iphase in listt:

                    filename_mean = list_foldername[iAsy] + 'image_%03d_%s_%02d_%s.raw' % (ip+1,list_4[0],ic,iphase)
                    mean_img = load_One_Simple_Raw_image_3D(filename_mean, colsize, rowsize, height, bb)
                    mean_img = np.float32(mean_img/np.max(mean_img))
                    im = np.reshape(mean_img, (height, colsize, rowsize))
                    #for ii in range(10):
                    #    plt.imshow(im[ii,:,:])
                    #    plt.show()
                    #sys.exit()

                    img = img_to_array(im)
                    
                    if ip+itest == (ip+itest)//istep*istep:
                        if iphase == "00":
                            XT.append(img)
                            YT.append(cl_num)
                            #print(cl_num, filename_mean)
                    elif ip+1+itest == (ip+1+itest)//istep*istep:
                        if iphase == "00":
                            XV.append(img)
                            YV.append(cl_num)
                            #print(cl_num, filename_mean)
                    else:
                        if iAsy == 0:
                            X.append(img)
                            Y.append(cl_num)
                            #print(cl_num, filename_mean)
                        else:
                            if iphase == "00":
                                X.append(img)
                                Y.append(cl_num)
                                #print(cl_num, filename_mean)
                    
    #sys.exit()

    # Data2

    for iAsy in range(len(list_Asy)):
        Shift2 = pd.read_csv(ShiftData2[iAsy], encoding="SHIFT-JIS")
        Shift2.columns = ["file","machine","mode","4","5","6","7","8","9","10","11"]
        modeshift2 = Shift2[Shift2["mode"] == mode]
        num_s = len(modeshift2.iloc[:,0])
        # For 2 classes 
        if iAsy == 0:
            cl_num = 0
        else:
            cl_num = 1

        for ip in range(num_s):
            for ic in range(modeshift2.iloc[ip,10]):
                if iAsy == 0:
                    listt = list_N
                elif iAsy == 1:
                    listt = list_LAD
                elif iAsy == 2:
                    listt = list_LCX
                elif iAsy == 3:
                    listt = list_RCA
                for iphase in listt:
                    filename_mean = list_foldername2[iAsy] + 'image_%03d_%s_%02d_%s.raw' % (ip+1,list_4[0],ic,iphase)
                    mean_img = load_One_Simple_Raw_image_3D(filename_mean, colsize, rowsize, height, bb)
                    mean_img = np.float32(mean_img/np.max(mean_img))
                    im = np.reshape(mean_img, (height, colsize, rowsize))
                    #for ii in range(10):
                    #    plt.imshow(im[ii,:,:])
                    #    plt.show()
                    img = img_to_array(im)

                    if not (mode == "short" and ((iAsy == 1 and ip+1 in rem_list_Data2_LAD_s) or (iAsy == 3 and ip+1 in rem_list_Data2_RCA_s))) or (mode == "2ch" and ((iAsy == 0 and ip+1 in rem_list_Data2_N_2) or (iAsy == 2 and ip+1 in rem_list_Data2_LCX_2) or (iAsy == 3 and ip+1 in rem_list_Data2_RCA_2))) or (mode == "3ch" and ((iAsy == 0 and ip+1 in rem_list_Data2_N_3) or (iAsy == 2 and ip+1 in rem_list_Data2_LCX_3) or (iAsy == 3 and ip+1 in rem_list_Data2_RCA_3))) or (mode == "4ch" and ((iAsy == 0 and ip+1 in rem_list_Data2_N_4) or (iAsy == 1 and ip+1 in rem_list_Data2_LAD_4) or (iAsy == 2 and ip+1 in rem_list_Data2_LCX_4))):
                        X.append(img)
                        Y.append(cl_num)
                        #print(cl_num, filename_mean)
                    """
                    if ip+itest == (ip+itest)//istep*istep:
                        XT.append(img)
                        YT.append(cl_num)
                        #Train_Val.append("Val")
                        #print(ip,its,filename_mean,its_num)
                    elif ip+1+itest == (ip+1+itest)//istep*istep:
                        XV.append(img)
                        YV.append(cl_num)
                        #Train_Val.append("Val")
                        #print(ip,filename_mean)
                    else:
                        if iAsy == 0:
                            X.append(img)
                            Y.append(cl_num)
                            #Train_Val.append("Train")
                        else:
                            if iphase == "00" or iphase == "03" or iphase == "07":
                            #if iphase == "00":
                                X.append(img)
                                Y.append(cl_num)
                    """
    #sys.exit()
    
    # AI-AMED data
    its_num = 0
    ShiftA = pd.read_csv(ShiftData3[0], encoding="SHIFT-JIS")
    ShiftA2 = ShiftA[ShiftA["Asy"] != "999"]
    ShiftA3 = ShiftA2[ShiftA2["Asy"] != "none"]
    ShiftA4 = ShiftA3[ShiftA3["mode"] == mode]    
    datalen = len(ShiftA4.iloc[:,0])
    
    for ip in range(datalen):
        filename = ShiftA4.iloc[ip,0]
        num_cycles = ShiftA4.iloc[ip,10]
        iAsy = ShiftA4.iloc[ip,12]
        # For 2 classes 
        if iAsy == "0":
            cl_num = 0
        else:
            cl_num = 1

        for ic in range(ShiftA4.iloc[ip,10]):
            if iAsy == "0":
                listt = list_AI
            elif iAsy == "1":
                listt = list_AI_LAD
            elif iAsy == "2":
                listt = list_AI_LCX
            elif iAsy == "3":
                listt = list_AI_RCA
            for iphase in listt:
                filename_mean = list_foldername3[0] + '%s%02d_%s.raw' % (ShiftA4.iloc[ip,11],ic,iphase)
                mean_img = load_One_Simple_Raw_image_3D(filename_mean, colsize, rowsize, height, bb)
                mean_img = np.float32(mean_img/np.max(mean_img))
                im = np.reshape(mean_img, (height, colsize, rowsize))
                #for ii in range(10):
                #    plt.imshow(im[ii,:,:])
                #    plt.show()
                img = img_to_array(im)
                    
                if ip+itest == (ip+itest)//istep*istep:
                    if iphase == "00":
                        XT.append(img)
                        YT.append(cl_num)
                        #print(cl_num,filename_mean)
                #elif ip+1+itest == (ip+1+itest)//istep*istep:
                #    if iphase == "00":
                #        XV.append(img)
                #        YV.append(cl_num)
                #        #print(cl_num,filename_mean)
                else:
                    X.append(img)
                    Y.append(cl_num)
                    #print(cl_num,filename_mean)
                    
                    
        its_num += 1
    return X,Y,XT,YT,XV,YV
#    sys.exit()
#########################

### Start ####

list_Asy = ["N","LAD","LCX","RCA"]
ShiftData = ['information_heart_beat_2018_N_mod.csv','information_heart_beat_2018_LAD_mod.csv','information_heart_beat_2018_LCX_mod.csv','information_heart_beat_2018_RCA_mod.csv']

list_foldername = ["pickup10_2018_N/","pickup10_2018_LAD/","pickup10_2018_LCX/","pickup10_2018_RCA/"]

ShiftData2 = ['information_heart_beat_Data2_N_mod.csv','information_heart_beat_Data2_LAD_mod.csv','information_heart_beat_Data2_LCX_mod.csv','information_heart_beat_Data2_RCA_mod.csv']

list_foldername2 = ["pickup10_2018Data2_N/","pickup10_2018Data2_LAD/","pickup10_2018Data2_LCX/","pickup10_2018Data2_RCA/"]

ShiftData3 = ['information_heart_beat_AI-AMED_mod_new.csv']
list_foldername3 = ["pickup10_AI-AMED/"]

# Epochs
nepo = 2
## Loading training/test data
colsize = 120
rowsize = 120
height = 10
bb = "float32"
#bb = "float64"
rgb = 1
num_s = 50


istep = 5

list_4 = ["s"]

if list_4[0] == "l":
    mode = "long"
elif list_4[0] == "s":
    mode = "short"
elif list_4[0] == "2":
    mode = "2ch"
elif list_4[0] == "3":
    mode = "3ch"
elif list_4[0] == "4":
    mode = "4ch"

list_3 = ["00"]
list_N = ["00","01","09"]
list_LAD = ["00"]
list_LCX = ["00"]
list_RCA = ["00"]

list_AI = ["00"] # 277 => 477
list_AI_LAD = ["00"] # 122 => 222
list_AI_LCX = ["00","01","09"] # 22 => 122
list_AI_RCA = ["00","01"] # 49 => 149



num_class = 2
re_t = np.zeros(num_class*num_class,dtype ='int')
shape = (height, colsize, rowsize, 1)

## Model select
model = CNN_model2(num_class, shape)
model.summary()

outfoldername = "./tensorlog"
if not os.path.exists(outfoldername):
    os.makedirs(outfoldername)
model_name = '%s/test.json' % outfoldername
json_string = model.to_json()
open(model_name, 'w').write(json_string)


for itest in range(istep):

    print("loading data ...")
    X,Y,XT,YT,XV,YV = loading_usimage(itest,mode,list_Asy,ShiftData,list_foldername,ShiftData2,list_foldername2,ShiftData3,list_foldername3,list_N,list_LAD,list_LCX,list_RCA,list_AI,list_AI_LAD,list_AI_LCX,list_AI_RCA,list_4,list_3,colsize, rowsize, height, bb)

        
    X_train = np.asarray(X)
    Y_train = np.asarray(Y)
    X_val = np.asarray(XV)
    Y_val = np.asarray(YV)
    ndim = X_train.shape[0]
    X_train = np.reshape(X_train,(ndim, height, colsize, rowsize, 1))
    ndim_val = X_val.shape[0]
    X_val = np.reshape(X_val,(ndim_val, height, colsize, rowsize, 1))
    print("train data: ", X_train.shape)
    print("val data:   ", X_val.shape)

    Y_train = np_utils.to_categorical(Y_train, num_class)
    Y_val = np_utils.to_categorical(Y_val, num_class)
    
    X_test = np.asarray(XT)
    ndim_test = X_test.shape[0]
    X_test = np.reshape(X_test,(ndim_test, height, colsize, rowsize, 1))
    Y_test = np.asarray(YT)
    Y_test_raw = Y_test
    Y_test = np_utils.to_categorical(Y_test, num_class)

    print("test data:  ", X_test.shape)
    #sys.exit()
    num_test = len(X_test)
    
    acc_check = 0
    max_acc = 0
    jj_sum = 0
    # model optimization #
    for jj in range(1):
        model = model_from_json(open(model_name).read())
        
        ## Model save information
        fpath_n = '%s/weights_%02d_%02d.hdf5' % (outfoldername,itest,jj)
        cp_cb = ModelCheckpoint(filepath = fpath_n, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        def step_decay(epoch, jj):
            x = 0.003 + 0.0001*jj# - epoch*0.0000004*(jj*3) # 0.003 0.001 0.0005
            if x < 0: x = 0.0001
            return x
        lr_cb = LearningRateScheduler(step_decay)
        ##
        model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=32, epochs=nepo, validation_data = (X_val, Y_val), verbose = 1, callbacks=[cp_cb, lr_cb])
        
        acc_val = history.history['val_accuracy'][np.argmin(history.history['val_loss'])]
        acc_train =  history.history['accuracy'][np.argmin(history.history['val_loss'])]
        #acc_val = history.history['val_acc'][np.argmin(history.history['val_loss'])]
        #acc_train =  history.history['acc'][np.argmin(history.history['val_loss'])]
        threshold_val =  history.history['val_loss'][np.argmin(history.history['val_loss'])]
        threshold_train =  history.history['loss'][np.argmin(history.history['val_loss'])]
        print("loss values:" jj, threshold_train, threshold_val)
        
        #Learing plot
        imgname = '%s/plot_loss_%02d_%02d.png' % (outfoldername,itest,jj)
        plt.plot(np.log(history.history['loss']))
        plt.plot(np.log(history.history['val_loss']))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(imgname)
        plt.close("all")
        output_name = '%s/loss_%02d_%02d.csv' % (outfoldername,itest,jj)
    
        model.load_weights(fpath_n) 
        p_table = model.predict(X_test, batch_size=1, verbose=0)

        predict_classes = np.argmax(p_table,axis=1)
        Y_test_raw_id = Y_test_raw

        table = np.reshape(re_t, (num_class, num_class))*0
        for i in range(num_test):
            if Y_test_raw_id[i] == 0:
                if predict_classes[i] == 0:
                    table[0,0] += 1
                elif predict_classes[i] == 1:
                    table[0,1] += 1
            elif Y_test_raw_id[i] == 1:
                if predict_classes[i] == 0:
                    table[1,0] += 1
                elif predict_classes[i] == 1:
                    table[1,1] += 1
        print(table)
        test_acc = (table[0,0]+table[1,1])/num_test
        
        output_data_0 = pd.DataFrame(table)
        outfilename_0 = '%s/table_%02d.csv'% (outfoldername, itest)
        output_data_0.to_csv(outfilename_0, index=None,header=["NOR","ABN"])
        
        output_d = pd.DataFrame([[itest,jj,sum_acc,acc_train,acc_val,threshold_train,threshold_val,test_acc]])
        if itest == 0 and jj == 0:
            output_data_acc = output_d
        else:
            output_data_acc = output_data_acc.append(output_d)

        outfilename_0 = '%s/result_acc.csv'% (outfoldername)
        #output_data_acc.columns=["itest","jj","sum_acc","acc_train","acc_val","threshold_train","threshold_val","test_acc"]
        output_data_acc.to_csv(outfilename_0,index=False,encoding="SHIFT-JIS")

        # model optimization -- end#
        
sys.exit()



