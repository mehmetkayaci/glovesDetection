#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "C:/Users/Mehmet/glovesDetection/dataset" #dataset dizini

CATEGORIES = ["hands","gloves"] #dataset dizini alt klasorleri ayni zamanda kategori olarak kullaniliyor

for category in CATEGORIES:
    path = os.path.join(DATADIR,category) # klasor dizinlerini bulma
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE) # 2d array olusturduk grayscale ile
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break





IMG_SIZE = 100

new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)





training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category) # klasor dizinlerini bulma
        class_num= CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE) # 2d array olusturduk grayscale ile
                training_data.append([new_array,class_num])
            except Exception as e:
                print("training data olusturulamadi")
                
create_training_data()   





print(len(training_data)) #toplam data sayisi





import random #dataseti random olarak karistirma

random.shuffle(training_data)





x_train = []
y_train = []
for features ,label in training_data:
    x_train.append(features)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten

x_train = x_train/255.0 # scaling 255  maximum pixel degeri

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=30,activation='relu'))
model.add(Flatten()) #### flatten'a bak
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss="binary_crossentropy",optimizer = "adam")



y_train = np.reshape(y_train,(1132,1)) # shape




model.fit(x=x_train,y=y_train,batch_size=32,epochs=20,verbose=1,validation_split = 0.1)




import pandas as pd
dataloss = pd.DataFrame(model.history.history)
dataloss.plot()



# model.save("gloves_detection_model_v2.h5")














# In[ ]:




