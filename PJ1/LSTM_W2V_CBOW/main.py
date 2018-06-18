import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from utils.util import DataManager
import json

from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=50,type=int)

# model parameter
parser.add_argument('--loss_function', default='mse')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
args = parser.parse_args()

train_path = 'data/training_set.json'
test_path = 'data/test_set.json'
semi_path = 'data/training_nolabel.txt'

def to_word2vec(token_corpus):
    #print(token_corpus)
    print("pretraining word2vec")
    vector_size = 512
    window_size = 5
    word2vec = Word2Vec(sentences=token_corpus, min_count=1, size=vector_size,  window=window_size, negative=20, iter=10, sg=0)
    print("pretraining word2vec is over!!")
    return word2vec

# build model
def simpleRNN(args):
    inputs = Input(shape=(args.max_length,512))

    # Embedding layer
    #embedding_inputs = Embedding(args.vocab_size, 
    #                             args.embedding_dim, 
    #                             trainable=True)(inputs)
    # RNN 
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=return_sequence)
                       #dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size, 
                        return_sequences=return_sequence)
                        #dropout=dropout_rate)

    RNN_output = RNN_cell(inputs)

    # DNN layer
    outputs = Dense(args.hidden_size//2, 
                    activation='tanh')(RNN_output)
                    #kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    #outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='tanh')(outputs)
        
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss='mse', optimizer=adam, metrics=['mse'])
    
    return model



def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
    
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
        dm.add_data('test_data', test_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    elif args.action == 'test':
        dm.add_data('train_data', train_path, True)
        dm.add_data('test_data', test_path, True)
    else:
        raise Exception ('Action except for train, semi, and test')
    
    """      
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)
    """                     
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    """
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 
    """
    # convert to sequences
    token_corpus = dm.to_token_corpus(args.max_length)
    #word2vec = to_word2vec(token_corpus)
    if args.action == "train":
        word2vec = to_word2vec(token_corpus)
        save_path_word2vec_model = os.path.join(save_path,'word2vec.model')
        word2vec.save(save_path_word2vec_model)
    elif args.action == "test":
        path = os.path.join(load_path,'word2vec.model')
        if os.path.exists(path):
            print ('load model from %s' % path)
            word2vec = Word2Vec.load(path)
        else:
            raise ValueError("Can't find the file %s" %path)

    word2vec = word2vec.wv
    #print(word2vec['downgrades'])

    #padding sentence
    dm.padding_sent(args.max_length)
    dm.sent_to_word2vec(word2vec)
    #(X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)


    # initial model
    print ('initial model...')
    model = simpleRNN(args)    
    model.summary()

    print("args.load_model =", args.load_model)
    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        #print ('Warning : testing without loading any model')
        print('args.action is %s'%(args.action))
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
        
     # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        #print(type(X))
        #print(type(X[0]))
        #print(X[0][0])
        #print(X)
        #earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='max')
        #X, Y, X_val, Y_val = np.array(X), np.array(Y), np.array(X_val), np.array(Y_val)
        #print(X)
        #print(X[0])
        #X_val = np.reshape(X_val, (X_val.shape[0], args.max_length, X_val.shape[2]))
        save_path_model_h5 = os.path.join(save_path,'model.h5')
        """
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='max' )
        """
        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size)#,
                            #callbacks=[checkpoint, earlystopping] )

        model.save(save_path_model_h5) 

    # testing
    elif args.action == 'test' :
        args.val_ratio = 0
        (X,Y), (X_val, Y_val) = dm.split_data('test_data', args.val_ratio)
        predictions = model.predict(X)
        predictions = predictions.reshape(-1)
        scores = model.evaluate(X, Y)
        print("test data mse by keras = %f" % scores[1])
        print("test data mse by sklearn = %f" % mean_squared_error(Y, predictions))
        for idx, value in enumerate(predictions):
            if value > 0:
                predictions[idx] = 1
            elif value == 0:
                predictions[idx] = 0
            elif value < 0:
                predictions[idx] = -1

        for idx, value in enumerate(Y):
            if value > 0:
                Y[idx] = 1
            elif value == 0:
                Y[idx] = 0
            elif value < 0:
                Y[idx] = -1

        print("test data micro f1 score by sklearn = %f" % f1_score(Y, predictions, average='micro'))
        print("test data macro f1 score by sklearn = %f" % f1_score(Y, predictions, average='macro'))

        (X,Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)
        predictions = model.predict(X)
        predictions = predictions.reshape(-1)
        scores = model.evaluate(X, Y)
        print("train data mse by keras = %f" % scores[1])
        print("train data mse by sklearn = %f" % mean_squared_error(Y, predictions))
        for idx, value in enumerate(predictions):
            if value > 0:
                predictions[idx] = 1
            elif value == 0:
                predictions[idx] = 0
            elif value < 0:
                predictions[idx] = -1

        for idx, value in enumerate(Y):
            if value > 0:
                Y[idx] = 1
            elif value == 0:
                Y[idx] = 0
            elif value < 0:
                Y[idx] = -1

        print("train data micro f1 score by sklearn = %f" % f1_score(Y, predictions, average='micro'))
        print("train data macro f1 score by sklearn = %f" % f1_score(Y, predictions, average='macro')) 

        #raise Exception ('Implement your testing function')


    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='max' )
        # repeat 10 times
        for i in range(10):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)

if __name__ == '__main__':
        main()