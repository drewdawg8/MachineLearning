import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import os
import re
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
np.random.seed(7)
class TextClassifier:

    def __init__(self):
        self.tokenizer = Tokenizer(num_words = 5000)
        self.top_words = 5000
        self.max_words = 500
        self.model = model = Sequential()
        model.add(Embedding(self.top_words,32,input_length = self.max_words))
        model.add(Conv1D(filters = 32,kernel_size=3,padding='same',activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(250,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

    def train(self, X_train, y_train, X_test = None, y_test = None):
        self.model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 6,batch_size=128,verbose = 2)

    def predict(self,X):
        return self.model.predict()

    def init_tokenizer(self,X):
        self.tokenizer.fit_on_texts(X)

    def eval(self,X,y):
        return self.model.evaluate(X, y, verbose=0)

    def proccess_text(self,X):
        vocab_text = self.tokenizer.texts_to_sequences(X)
        vocab_text = sequence.pad_sequences(np.array(vocab_text),maxlen=500)
        return vocab_text

    def organize_text(self,pos_path,neg_path):
        data = {'label':[],'text':[]}
        pos_texts = self.text_to_array(pos_path)
        neg_texts = self.text_to_array(neg_path)
        for i in pos_texts:
            data['label'].append(0)
            data['text'].append(i)
        for i in neg_texts:
            data['label'].append(1)
            data['text'].append(i)
        return data

    def text_to_array(self,path):
            ''' Takes a path argument and retrieves all the text lines from
                within a folder'''
            name_list = []
            texts = []
            for file_ in os.listdir(path):
                name_list.append(file_)
            for i in name_list:
                file_ = open(path+i,encoding="utf8")
                for line in file_.read().splitlines():
                    texts.append(re.sub('[^a-zA-z0-9\s]','',line.lower()))
            return texts
