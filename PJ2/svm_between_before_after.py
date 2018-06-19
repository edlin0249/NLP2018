import tensorflow as tf
import numpy as np
import os
import datetime
import time
#from text_cnn import TextCNN
import data_helpers_ch_between_before_after as data_helpers
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Parameters
# ============n======================================

# Data loading params
tf.flags.DEFINE_string("train_dir", "./data/TRAIN_FILE.txt", "Path of train data")
tf.flags.DEFINE_float("dev_sample_percentage", 0.0, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train(98)/test(70) data (Default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("text_embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("position_embedding_dim", 100, "Dimensionality of position embedding (Default: 100)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (Default: 2,3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Data loading params
tf.flags.DEFINE_string("eval_dir", "./data/TEST_FILE.txt", "Path of evaluation data")
tf.flags.DEFINE_string("output_dir", "result/prediction.txt", "Path of prediction for evaluation data")
tf.flags.DEFINE_string("target_dir", "result/answer.txt", "Path of target(answer) file for evaluation data")

# Eval Parameters
#tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
#tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    with tf.device('/cpu:0'):
        e1_list, e2_list, pos1, pos2, between, before, after, y = data_helpers.load_data_and_labels(FLAGS.train_dir, mode='train')

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    # text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # text_vec = np.array(list(text_vocab_processor.fit_transform(x_text)))
    # print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    # pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # pos_vocab_processor.fit(pos1 + pos2)
    # pos1_vec = np.array(list(pos_vocab_processor.transform(pos1)))
    # pos2_vec = np.array(list(pos_vocab_processor.transform(pos2)))
    # print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))


    #pos1_vec = np.zeros(pos1_vec.shape)
    #pos2_vec = np.zeros(pos2_vec.shape)
    # x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    x = []
    for idx in range(len(e1_list)):
        between_flatten = between[idx].reshape(-1,).tolist()
        before_flatten = before[idx].reshape(-1,).tolist()
        after_flatten = after[idx].reshape(-1,).tolist()        
        x.append(e1_list[idx]+e2_list[idx]+[pos1[idx]]+[pos2[idx]]+between_flatten+before_flatten+after_flatten)
    x = np.array(x).astype('float32')

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
    #dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))

    x_train = x_shuffled[:]
    x_train = x_train.reshape(x_train.shape[0], -1)
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # x_train = x_train.reshape(x_train.shape[0], -1)
    # x_dev = x_dev.reshape(x_dev.shape[0], -1)

    #y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    y_train = y_shuffled[:]
    y_train = np.argmax(y_train, axis=1)
    #y_dev = np.argmax(y_dev, axis=1)
    #print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    #from sklearn.svm import SVC
    clf = SVC(verbose=True, kernel='linear')
    clf.fit(x_train, y_train)
    #rfc = RandomForestClassifier(bootstrap=False, verbose=1)
    #rfc.fit(x_train, y_train)
    print('The accuracy of SVM on training set:', clf.score(x_train, y_train))
    #print('The accuracy of SVM on validation set:', clf.score(x_dev, y_dev))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
    #pos_vocab_processor.save(os.path.join(out_dir, "position_vocab"))

    #import pickle
    #s = pickle.dumps(clf)
    joblib.dump(clf, './svm_between_before_after_novalid.pkl')
    clf = joblib.load('./svm_between_before_after_novalid.pkl')

    with tf.device('/cpu:0'):
        e1_list, e2_list, pos1, pos2, between, before, after, y = data_helpers.load_data_and_labels(FLAGS.eval_dir, mode='eval')

    # Map data into vocabulary
    # text_path = os.path.join(checkpoint_dir, "..", "text_vocab")
    # text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    # text_vec = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    # position_path = os.path.join(checkpoint_dir, "..", "position_vocab")
    # position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    # pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
    # pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

    # x_eval = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    # x_eval = x_eval.reshape(x_eval.shape[0], -1)
    x_eval = []
    for idx in range(len(e1_list)):
        between_flatten = between[idx].reshape(-1,).tolist()
        before_flatten = before[idx].reshape(-1,).tolist()
        after_flatten = after[idx].reshape(-1,).tolist()        
        x_eval.append(e1_list[idx]+e2_list[idx]+[pos1[idx]]+[pos2[idx]]+between_flatten+before_flatten+after_flatten)
    x_eval = np.array(x_eval).astype('float32')
    #y_eval = np.argmax(y, axis=1)
    y_pred = clf.predict(x_eval)

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

    with open("ans_svm_between_before_after_novalid.txt", "w") as fout:
        for idx, value in enumerate(y_pred):
            fout.write("%d\t%s\n"%(idx+8001, labelsMapping[value]))

    print("outputing prediction is over...")


            


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
