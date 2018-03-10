import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.metrics import categorical_crossentropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randint
from numpy import array
from keras import backend as K
import itertools
import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

posList = []
positive_text = []
train_path = './data/train/'
for file_ in os.listdir(train_path+'pos'):
    posList.append(file_)
for i in posList:
    file_ = open(train_path+'pos/'+i)
    positive_text.append(file_.readlines())

print(positive_text)



train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes = ['cat','dog'],batch_size = 10)

#Scaler is used to get all of our input between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

model = Sequential([
        Conv2D(64,(3,3),activation ='relu',input_shape =(224,224,3)),
        Conv2D(32,(3,3),activation = 'relu'),
        Flatten(),
        Dense(2,activation = 'softmax')])

#model = load_model('firstKeras.h5')
model.compile(Adam(lr=.001),loss='categorical_crossentropy',metrics=['accuracy'])


model.fit_generator(train_batches,steps_per_epoch=15,epochs = 20, shuffle = True,verbose = 2)
#predictions = model.predict_classes(scaled_tests,batch_size=10,verbose=0)
model.save('firstKeras.h5')
K.clear_session()
