import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
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

        X, Y = [], []
        with open(data_path,'r') as f:
            data = json.load(f)
            for item in data:
                #if with_label:
                    #lines = line.strip().split(' +++$+++ ')
                X.append(item['tweet'])
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
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
        self.tokenizer.fit_on_texts("<PAD>")
        
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
    def to_token_corpus(self, maxlen):
        self.maxlen = maxlen
        token_corpus = []
        maxlen = 0
        for key in self.data:
            print ('Converting %s to sequences'%key)
            #print(self.data[key][0])
            for sent in self.data[key][0]:
                #print(sent)
                wordseq = text_to_word_sequence(sent)
                if len(wordseq) > maxlen:
                    maxlen = len(wordseq)
                #print(wordseq)
                #if 'downgrades' in wordseq:
                #    print("word 'downgrades' not in vocabulary")
            
                token_corpus.append(wordseq+["<PAD>"])
            #print("maxlen = %d"%maxlen)
        #print("len(token_corpus) = %d"%len(token_corpus))
        return token_corpus
    
    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
    
    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))

    #padding sentence
    def padding_sent(self, maxlen):
        for key in self.data:
            print("key = %s"%key)
            for idx, sent in enumerate(self.data[key][0]):
                #print(sent)
                wordseq = text_to_word_sequence(sent)
                wordseq = wordseq + ["<PAD>"] * (maxlen - len(wordseq))
                #print(wordseq)
                self.data[key][0][idx] = wordseq
        #print(self.data["train_data"][0])
        #print(self.data["test_data"][0])
        #print(np.array(self.data["train_data"][0]).shape)
        #print(np.array(self.data["test_data"][0]).shape)
    
    #transform word2vec
    def sent_to_word2vec(self, word2vec):
        for key in self.data:
            print("key = %s"%key)
            for sent_idx, sent in enumerate(self.data[key][0]):
                for word_idx, word in enumerate(self.data[key][0][sent_idx]):
                    self.data[key][0][sent_idx][word_idx] = word2vec[word]
        #print(np.array(self.data["train_data"][0]).shape)
        #print(np.array(self.data["test_data"][0]).shape)



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
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
    