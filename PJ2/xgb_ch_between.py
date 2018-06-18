import tensorflow as tf
import numpy as np
import os
import datetime
import time
from text_cnn import TextCNN
import data_helpers_ch_between as data_helpers
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_dir", "data/TRAIN_FILE.txt", "Path of train data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 20, "Max sentence length in train(98)/test(70) data (Default: 100)")
tf.flags.DEFINE_string("eval_dir", "data/TEST_FILE.txt", "Path of evaluation data")

# xxxxx parameters
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    with tf.device('/cpu:0'):
        e1_list, e2_list, pos1, pos2, between, y = data_helpers.load_data_and_labels(FLAGS.train_dir,FLAGS.max_sentence_length,mode='train')


    x = []
    for idx in range(len(e1_list)):
        between_flatten = []
        for ele in between[idx]:
            between_flatten += ele
        x.append(e1_list[idx] + e2_list[idx] + [pos1[idx]] + [pos2[idx]] + between_flatten)
    
    x = np.array(x).astype('float32')

    ########### GGGGGGGGGGGGGG
    # x = np.array([list(i) for i in text_vec])
    print(x[0])
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    print( 'x_dev = ' , x_dev.shape)
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    #### xgboost_block
    import xgboost as xgb
    print('x_dev = ' , x_dev.shape)
    X_train = x_train
    X_valid = x_dev
    Y_train = np.argmax(y_train, axis=1)
    Y_valid = np.argmax(y_dev, axis=1)

    # read in data
    d_train = xgb.DMatrix(X_train,label=Y_train)
    d_valid = xgb.DMatrix(X_valid, label=Y_valid)
    watchlist  = [(d_train,'train'),(d_valid,'eval')]
    
    # specify parameters via map
    params = {'max_depth': 6, 'eta': 0.15, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 19 ,'nthread': 4}
    num_round = 20000
    bst = xgb.train(params, d_train, num_round, watchlist, early_stopping_rounds=5)

    
    ## GGGGGGGGGGGGGGGGGGGGGGGGGGG
    e1_list, e2_list, pos1, pos2, between, y = data_helpers.load_data_and_labels(FLAGS.eval_dir,FLAGS.max_sentence_length,mode='eval')

    ### GGGGGGGGGGGGGGGGGGGGGGGGGGGG
    x_eval = []
    for idx in range(len(e1_list)):
        between_flatten = []
        for ele in between[idx]:
            between_flatten += ele
        x_eval.append(e1_list[idx] + e2_list[idx] + [pos1[idx]] + [pos2[idx]] + between_flatten)
    
    x_eval = np.array(x_eval).astype('float32')


    # x_eval = np.array([list(i) for i in text_vec]).reshape(-1,100).astype(int)


    y_eval = np.argmax(y, axis=1)

    # make prediction
    d_test = xgb.DMatrix(x_eval)
    preds = bst.predict(d_test, ntree_limit=bst.best_ntree_limit)


    labelsMapping = {0: 'Other',
                 1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                 3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                 5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                 7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                 9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                 11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                 13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                 15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                 17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

    with open('./ans/ans_xgb_between.txt','w') as f:
        for idx,ele in enumerate(preds):
            f.write('%d\t%s\n' %(idx + 8001, labelsMapping[ele]))
    


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()