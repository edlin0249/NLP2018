import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
import json

class DataManager:
    def __init__(self):
        self.data = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)

        X, Y = [[], [], []], []
        with open(data_path,'r') as f:
            data = json.load(f)
            for item in data:
                #if with_label:
                    #lines = line.strip().split(' +++$+++ ')
                tmp = ''
                for idx, value in enumerate(item['snippet']):
                    tmp += (value + ' ')
                item['snippet'] = tmp.strip()
                #X.append([item['tweet'], item['snippet'], item['target']])
                X[0].append(item['tweet'])
                X[1].append(item['snippet'])
                X[2].append(item['target'])
                Y.append(float(item['sentiment']))
                #else:
                    #X.append(data['tweet'])

        #if with_label:
        self.data[name] = [X,Y]
        #else:
            #self.data[name] = [X]

    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.tokenizer = Tokenizer(num_words=vocab_size)
        for key in self.data:
            print ('tokenizing %s'%key)
            #texts = self.data[key][0]
            for idx, value in enumerate(self.data[key][0]):
                self.tokenizer.fit_on_texts(value)
        
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))

    # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            for idx, value in enumerate(self.data[key][0]):
                tmp = self.tokenizer.texts_to_sequences(value)
                #if idx != 2:
                self.data[key][0][idx] = list(pad_sequences(tmp, maxlen=maxlen))
                for index, value in enumerate(self.data[key][0][idx]):
                    self.data[key][0][idx][index] = list(value)
                #else:
                #    self.data[key][0][idx] = list(pad_sequences(tmp, maxlen=1))
                #    for index, value in enumerate(self.data[key][0][idx]):
                #        self.data[key][0][idx][index] = list(value)
        #self.data[key][0] = np.array(self.data[key][0])
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            for idx, value in enumerate(self.data[key][0]):
                self.data[key][0][idx] = self.tokenizer.texts_to_matrix(self.data[key][0][idx],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))

    def get_semi_data(self,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        print(data[0])
        X = np.array(data[0])
        Y = np.array(data[1])
        data_size = len(X[0])
        val_size = int(data_size * ratio)
        print("val_size = %d"%val_size)
        return (X[:, val_size:],Y[val_size:]),(X[:, :val_size],Y[:val_size])
    