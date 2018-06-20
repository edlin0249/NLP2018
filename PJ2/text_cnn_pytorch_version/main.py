import argparse
import data_helpers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from text_cnn import *
import numpy as np
import datetime
import time
import os
import tensorflow as tf

def train(args):
    print("Starting load_data_and_labels...")
    x_text, pos1, pos2, y = data_helpers.load_data_and_labels(args.train_dir, mode='train')
    #print(x_text)
    #for words
    print("Starting buildvoc...")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(args.max_sentence_length)
    text_vec = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))
    #all_tokens = tokenize(x_text)
    #word2idx, idx2word = buildvoc(all_tokens)
    #alltokens2idx = indexify(all_tokens, word2idx)
    #text_vec = padding(alltokens2idx, args.max_sentence_length)
    #print(np.array(text_vec).shape)
    #for pos1+pos2
    print("Starting buildposvoc...")
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(args.max_sentence_length)
    pos_vocab_processor.fit(pos1 + pos2)
    pos1_vec = np.array(list(pos_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(pos_vocab_processor.transform(pos2)))
    print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    #pos1_tokens, pos2_tokens = tokenizepos(pos1, pos2)
    #pos2idx, idx2pos = buildposvoc(pos1_tokens, pos2_tokens)
    #pos1_vec, pos2_vec = indexifypos(pos1_tokens, pos2_tokens, pos2idx)
    #print(np.array(pos1_vec).shape)
    #print(np.array(pos2_vec).shape)

    x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    x_dev = Variable(torch.from_numpy(np.array(x_dev).transpose((1, 0, 2))))
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    y_dev = Variable(torch.LongTensor(np.argmax(y_dev, axis=1).tolist()))
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    print("Initializing model...")
    cnn = TextCNN(sequence_length=x_train.shape[2], num_classes=y_train.shape[1], text_vocab_size=len(text_vocab_processor.vocabulary_),
                text_embedding_size=args.text_embedding_dim, pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=args.position_embedding_dim, filter_sizes=list(map(int, args.filter_sizes.split(","))),
                num_filters=args.num_filters, dropout_keep_prob=args.dropout, l2_reg_lambda=args.l2_reg_lambda)


    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
    pos_vocab_processor.save(os.path.join(out_dir, "position_vocab"))

    print("Starting load word2vec...")
    if args.word2vec:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), args.text_embedding_dim))
        # load any vectors from the word2vec
        print("Load word2vec file {0}".format(args.word2vec))
        with open(args.word2vec, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1).decode('latin-1')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                idx = text_vocab_processor.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        cnn.W_text.weight = torch.nn.Parameter(torch.from_numpy(initW).float(), requires_grad=False)
        print(cnn.W_text.weight.requires_grad)
        print("Success to load pre-trained word2vec model!\n")

        # Generate batches
    batches = data_helpers.batch_iter( list(zip(x_train, y_train)), args.batch_size, args.num_epochs)

    print("Starting train model...")
    for idx, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        x_batch = np.array(x_batch).transpose((1, 0, 2))
        x_batch = Variable(torch.from_numpy(x_batch))
        #print(y_batch)
        y_batch = Variable(torch.LongTensor(np.argmax(y_batch, axis=1).tolist()))
        optimizer.zero_grad()

        logits = cnn(x_batch[0], x_batch[1], x_batch[2])
        loss = loss_func(logits, y_batch)
        loss.backward()
        optimizer.step()

        if idx % args.display_every == 0:
            corrects = (torch.max(logits, 1)[1].view(y_batch.size()).data == y_batch.data).sum()
            accuracy = 100.0 * corrects/y_batch.size(0)
            print("%d-th batch, loss = %f, corrects = %d, accuracy = %f"%(idx, loss, corrects, accuracy))

        if idx % args.evaluate_every == 0:
            logits = cnn(x_dev[0], x_dev[1], x_dev[2])
            loss = F.cross_entropy(logits, y_dev)
            corrects = (torch.max(logits, 1)[1].view(y_dev.size()).data == y_dev.data).sum()
            accuracy = 100.0 * corrects/y_dev.size(0)
            print("validation set, loss = %f, corrects = %d, accuracy = %f"%(loss, corrects, accuracy))

        if idx % args.checkpoint_every == 0:
            if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)
            save_path = checkpoint_prefix+"/textcnn_step_{}.pt".format(idx)
            torch.save(cnn.state_dict(), save_path)


def test(args):
    print("Starting load_data_and_labels")
    x_text, pos1, pos2, y = data_helpers.load_data_and_labels(args.test_dir, mode='eval')

    text_path = os.path.join(args.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    text_vec = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(args.checkpoint_dir, "..", "position_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

    x_eval = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    #y_eval = np.argmax(y, axis=1)
    #print("Starting buildvoc...")
    #for words
    # all_tokens = tokenize(x_text)
    # word2idx, idx2word = buildvoc(all_tokens)
    # alltokens2idx = indexify(all_tokens, word2idx)
    # text_vec = padding(alltokens2idx, args.max_sentence_length)
    #for pos1+pos2
    #print("Starting buildposvoc...")
    # pos2idx, idx2pos = buildposvoc(pos1, pos2)
    # pos1_vec, pos2_vec = indexifypos(pos1, pos2, pos2idx)
    # x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])

    #x_test = x
    #y_test = y

    print("Starting initialize CNN model...")
    cnn = TextCNN(sequence_length=x_eval.shape[2], num_classes=19, text_vocab_size=len(text_vocab_processor.vocabulary_),
            text_embedding_size=args.text_embedding_dim, pos_vocab_size=len(position_vocab_processor.vocabulary_),
            pos_embedding_size=args.position_embedding_dim, filter_sizes=list(map(int, args.filter_sizes.split(","))),
            num_filters=args.num_filters, dropout_keep_prob=args.dropout, l2_reg_lambda=args.l2_reg_lambda)

    print("Starting load trained model...")
    try:
        #print(type(args.model_path))
        cnn.load_state_dict(torch.load(args.model_path))
    except:
        raise "model path no found when testing"

    cnn.eval()

    print("Starting load word2vec...")
    if args.word2vec:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), args.text_embedding_dim))
        # load any vectors from the word2vec
        print("Load word2vec file {0}".format(args.word2vec))
        with open(args.word2vec, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1).decode('latin-1')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                idx = text_vocab_processor.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        cnn.W_text.weight = torch.nn.Parameter(torch.from_numpy(initW).float(), requires_grad=False)
        print("Success to load pre-trained word2vec model!\n")

    # Generate batches
    #batches = data_helpers.batch_iter( list(zip(x_train, y_train)), args.batch_size, args.num_epochs)
    x_eval = np.array(x_eval).transpose((1, 0, 2))
    x_eval = Variable(torch.from_numpy(x_eval))
    print("Starting predict...")
    logits = cnn(x_eval[0], x_eval[1], x_eval[2])

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

    print("Starting output result...")
    with open(args.output_file, "w") as fout:
        for idx, val in enumerate(logits):
            #print(np.argmax(val.data.numpy()))
            fout.write("%d\t%s\n"%(idx+8001, labelsMapping[np.argmax(val.data.numpy())]))



def main(args):
    if args.action == 'train':
        train(args)
    else:
        test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # action
    parser.add_argument('action', choices=['train', 'test'], help='choose train or test')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-num_epochs', type=int, default=100, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-display_every',  type=int, default=10,   help='Number of iterations to display training info. [default: 10]')
    parser.add_argument('-evaluate_every', type=int, default=100, help='Evaluate model on dev set after this many steps [default: 100]')
    parser.add_argument('-checkpoint_every', type=int, default=100, help='Save model after this many steps [default: 100]')
    parser.add_argument('-early_stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save_best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-model_path', default=None, help='path to load one model')
    parser.add_argument('-checkpoint_dir', default=None, help='Checkpoint directory from training run')
    parser.add_argument('-output_file', default='./ans_textcnn_pytorch.txt', help='filename of output the result after prediction')
    # data 
    parser.add_argument('-train_dir', default="../data/TRAIN_FILE.txt", help="Path of train data")
    parser.add_argument('-test_dir', default="../data/TEST_FILE.txt", help="Path of test data")
    parser.add_argument('-dev_sample_percentage', default=0.1, type=float, help="Percentage of the training data to use for validation")
    parser.add_argument('-max_sentence_length', default=100, type=int, help="Max sentence length in train(98)/test(70) data (Default: 100)")
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-word2vec', default="../GoogleNews-vectors-negative300.bin", help='Word2vec file with pre-trained embeddings')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    #parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-text_embedding_dim', type=int, default=300, help="Dimensionality of word embedding (Default: 300)")
    parser.add_argument('-position_embedding_dim', type=int, default=100, help="Dimensionality of position embedding (Default: 100)")
    parser.add_argument('-num_filters', type=int, default=128, help="Number of filters per filter size (Default: 128)")
    parser.add_argument('-filter_sizes', type=str, default='2,3,4,5', help="Comma-separated filter sizes (Default: 2,3,4,5)")
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    parser.add_argument("-l2_reg_lambda", default=3.0, type=float, help="L2 regularization lambda (Default: 3.0)")
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    args = parser.parse_args()
    main(args)