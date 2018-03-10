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
from ProcessData import *
import itertools
import os
import pickle


import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = './data/train/'
# file1 = open('positive_texts.pkl','rb')
# positive_texts = pickle.load(file1)
# file2 = open('negative_texts.pkl','rb')
# negative_texts = pickle.load(file2)
# file1.close(),file2.close()
positive_texts = textfile_to_array(path+'pos/')
negative_texts = textfile_to_array(path+'neg/')
positive_encoded = process_for_model(positive_texts)
negative_encoded = process_for_model(negative_texts)
# for i in positive_encoded:
#     while len(i)<350:
#         i.append(-1)
#
# for i in negative_encoded:
#     while len(i)<350:
#         i.append(-1)


training_labels = []
train_batches = []
for i in range(len(positive_encoded)):
    train_batches+=(positive_encoded[i])
    training_labels.append([0])
for i in range(len(negative_encoded)):
    train_batches+=(negative_encoded[i])
    training_labels.append([1])

train_batches = array(train_batches)/1000
training_labels = array(training_labels)
#Scaler is used to get all of our input between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = []
print(training_labels.shape)

model = Sequential([
        Dense(16,input_shape=(168,),activation='relu'),
        Dense(2,activation='softmax')
])

#model = load_model('firstKeras.h5')
model.compile(Adam(lr=.001),loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_batches, training_labels, batch_size = 2,epochs = 20, shuffle = True,verbose = 2)#predictions = model.predict_classes(scaled_tests,batch_size=10,verbose=0)
model.save('firstKeras.h5')
K.clear_session()
