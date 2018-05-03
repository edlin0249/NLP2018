import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, multiply, Lambda, dot, Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
from keras.backend import int_shape
import tensorflow as tf

from utils.util import DataManager
import json

parser = argparse.ArgumentParser(description='Sentiment regression')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=6, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=40,type=int)

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

def attention_3d_block(TimeDistributed_tweets, TimeDistributed_snippets, Dense_targets):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(TimeDistributed_tweets.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(TimeDistributed_tweets)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t_tmp = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(
        TimeDistributed_snippets)
    #merged = concatenate([h_t_tmp, Dense_targets])
    h_t = multiply([h_t_tmp, Dense_targets])
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([TimeDistributed_tweets, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh',
                             name='attention_vector')(pre_activation)
    return attention_vector

# build model
def simpleRNN(args):
    inputs_tweets = Input(shape=(args.max_length,), name='tweets')
    inputs_snippets = Input(shape=(args.max_length,), name='snippets')
    inputs_targets = Input(shape=(args.max_length,), name='targets')

    # Embedding layer
    embedding_tweets = Embedding(args.vocab_size, 
                                 args.embedding_dim, 
                                 trainable=True)(inputs_tweets)

    embedding_snippets = Embedding(args.vocab_size, 
                                 args.embedding_dim, 
                                 trainable=True)(inputs_snippets)

    embedding_targets = Embedding(args.vocab_size, 
                                 args.embedding_dim, 
                                 trainable=True)(inputs_targets)
    # RNN 
    return_sequence = True
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=return_sequence)
                       #dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size, 
                        return_sequences=return_sequence)
                        #dropout=dropout_rate)

    RNN_tweets = RNN_cell(embedding_tweets)
    RNN_snippets = RNN_cell(embedding_snippets)
    TimeDistributed_tweets = TimeDistributed(Dense(20))(RNN_tweets)
    TimeDistributed_snippets = TimeDistributed(Dense(20))(RNN_snippets)
    embedding_targets_lastone = Lambda(lambda x: x[:, -1, :], output_shape=(args.embedding_dim,), name='embedding_targets_lastone')(embedding_targets)
    Dense_targets = Dense(20)(embedding_targets_lastone)

    attention_mul = attention_3d_block(TimeDistributed_tweets, TimeDistributed_snippets, Dense_targets)
    # attention_mul = Flatten()(attention_mul)
    output_1 = Dense(50, activation='tanh')(attention_mul)
    output_final = Dense(1, activation='tanh')(output_1)

    #TimeDistributed_tweets = TimeDistributed(Dense(20))(RNN_tweets)
    #TimeDistributed_snippets = TimeDistributed(Debse(20))(RNN_snippets)
    #Concat_snippets = Concatenate(axis=0)(TimeDistributed_snippets)
    #Dense_snippets = Dense(20)(Concat_snippets)
    #Dense_targets = Dense(20)(inputs_targets)
    #Snippets_Targets = Multiply([Dense_snippets, Dense_targets])
    

    # DNN layer
    #outputs = Dense(args.hidden_size//2, 
    #                activation='relu')(RNN_output)
                    #kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    #outputs = Dropout(dropout_rate)(outputs)
    #outputs = Dense(1, activation='linear')(outputs)
    print("output_final.shape =", output_final.shape)
    model =  Model(inputs=[inputs_tweets, inputs_snippets, inputs_targets], outputs=output_final)

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
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    elif args.action == 'test':
        dm.add_data('train_data', train_path, True)
        dm.add_data('test_data', test_path, True)
    else:
        raise Exception ('Action except for train, semi, and test')
            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)
                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 

    # convert to sequences
    dm.to_sequence(args.max_length)

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
        #earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        """
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='max' )
        """
        tweets = X[0, :]
        snippets = X[1, :]
        targets = X[2, :]
        print("tweets's shape = ", tweets.shape)
        print("snippets's shape = ", snippets.shape)
        print("targets's shape = ", targets.shape)
        print("Y's shape = ", Y.shape)
        #model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
        history = model.fit([tweets, snippets, targets], Y, 
                            validation_data=([X_val[0, :], X_val[1, :], X_val[2, :]], Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size)#,
                            #callbacks=[checkpoint, earlystopping] )
        predictions = model.predict([tweets, snippets, targets])
        #print(predictions.shape)
        #print(predictions)

        model.save(save_path) 

    # testing
    elif args.action == 'test' :
        args.val_ratio = 0
        (X,Y), (X_val, Y_val) = dm.split_data('test_data', args.val_ratio)
        tweets = X[0, :]
        snippets = X[1, :]
        targets = X[2, :]
        #print("tweets.shape =", tweets.shape)
        #print("snippets.shape =", snippets.shape)
        #print("targets.shape =", targets.shape)
        predictions = model.predict([tweets, snippets, targets])
        #print(predictions)
        #print(Y.shape)
        #scores = np.sum((predictions - Y)**2)/len(Y)
        scores = model.evaluate([tweets, snippets, targets], Y)
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
        #print("test data scores[1](loss = mse) = %f" % scores[1])
        #raise Exception ('Implement your testing function')
        (X,Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)
        tweets = X[0, :]
        snippets = X[1, :]
        targets = X[2, :]
        predictions = model.predict([tweets, snippets, targets])
        #scores = np.sum((predictions - Y)**2)/len(Y)
        scores = model.evaluate([tweets, snippets, targets], Y)
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