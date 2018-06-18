import numpy as np
import pandas as pd
import nltk
import re
import gensim
from tqdm import tqdm

WORD_EMBEDDING_LENGTH = 300

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(path,maxlen=10,mode='train'):
    data = []
    lines = [line.strip() for line in open(path)]
    
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    print('load embedding finished.')
    
    if mode == 'train':
        stride = 4
    elif mode == 'eval':
        stride = 1
    else:
        exit('error mode on load_data_and_labels in data_helper')
        
    for idx in tqdm(range(0, len(lines), stride)):
        id = lines[idx].split("\t")[0]
        if mode == 'train':
            relation = lines[idx + 1]
        elif mode == 'eval':
            relation = 'Other'

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")

        tokens = nltk.word_tokenize(sentence)

        tokens.remove('_/e1_')
        tokens.remove('_/e2_')

        e1pos = tokens.index("_e1_")
        del tokens[e1pos]
        e2pos = tokens.index("_e2_")
        del tokens[e2pos]

        sentence = " ".join(tokens)
        sentence = clean_str(sentence)

        # use gensim to transfer to word embedding
        try:
            e1 = model.wv[tokens[e1pos]].tolist()
        except KeyError:
            e1 = np.zeros(shape=(WORD_EMBEDDING_LENGTH,)).tolist()
        try:
            e2 = model.wv[tokens[e2pos]].tolist()
        except KeyError:
            e2 = np.zeros(shape=(WORD_EMBEDDING_LENGTH,)).tolist()

        # add between word (maxlen)
        between = np.zeros(shape=(maxlen,WORD_EMBEDDING_LENGTH))
        min_idx = min(e1pos,e2pos)
        max_idx = max(e1pos,e2pos)
        cnt = 0
        for idx in range(min_idx + 1 , max_idx):
            if cnt > (maxlen - 1):
                break
            try:
                word_e = model.wv[tokens[idx]]
            except KeyError:
                word_e = np.zeros(shape=(WORD_EMBEDDING_LENGTH,))
            between[cnt] = word_e
            cnt += 1
        
        # add before_after word (3 words)
        before = np.zeros(shape=(3,WORD_EMBEDDING_LENGTH))
        after = np.zeros(shape=(3,WORD_EMBEDDING_LENGTH))
        min_idx = min(e1pos,e2pos)
        max_idx = max(e1pos,e2pos)
        min_left = max(min_idx - 3 ,0)
        max_right = min(max_idx + 3, len(tokens) )
        
        cnt = 0
        for idx in range(min_left,min_idx):
            try:
                word_e = model.wv[tokens[idx]]
            except KeyError:
                word_e = np.zeros(shape=(WORD_EMBEDDING_LENGTH,))
            before[cnt] = word_e
            cnt += 1

        cnt = 0
        for idx in range(max_idx,max_right):
            try:
                word_e = model.wv[tokens[idx]]
            except KeyError:
                word_e = np.zeros(shape=(WORD_EMBEDDING_LENGTH,))
            after[cnt] = word_e
            cnt += 1

        # add POS
        partofspeech_e1 = pos2onehot(nltk.pos_tag(tokens[e1pos])[0][1])
        partofspeech_e2 = pos2onehot(nltk.pos_tag(tokens[e2pos])[0][1])


        data.append([id, e1, e2, e1pos, e2pos, partofspeech_e1, partofspeech_e2 , between, before, after, relation])

    df = pd.DataFrame(data=data, columns=["id", "e1" , "e2", "e1_pos", "e2_pos", "partofspeech_e1", "partofspeech_e2", "between", "before", "after", "relation"])
    labelsMapping = {'Other': 0,
                     'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                     'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                     'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                     'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                     'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                     'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                     'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                     'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                     'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    df['label'] = [labelsMapping[r] for r in df['relation']]

    e1_list = df['e1'].tolist()
    e2_list = df['e2'].tolist()

    pos1 = df['e1_pos'].tolist()
    pos2 = df['e2_pos'].tolist()
    
    partofspeech_e1 = df['partofspeech_e1'].tolist()
    partofspeech_e2 = df['partofspeech_e2'].tolist()

    between = df['between'].tolist()
    before = df['before'].tolist()
    after = df['after'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()

    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return e1_list, e2_list, pos1, pos2, partofspeech_e1, partofspeech_e2, between, before, after, labels


def get_relative_position(df, max_sentence_length=100):
    # Position data
    pos1 = []
    pos2 = []

    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        e1 = df.iloc[df_idx]['e1_pos']
        e2 = df.iloc[df_idx]['e2_pos']

        d1 = ""
        d2 = ""
        for word_idx in range(len(tokens)):
            d1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            d2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        for _ in range(max_sentence_length - len(tokens)):
            d1 += "999 "
            d2 += "999 "
        pos1.append(d1)
        pos2.append(d2)

    return pos1, pos2

def pos2onehot(pos):
    dic ={ "$": 0, "''": 1,"(": 2,")": 3,",": 4,"--": 5,".": 6,":": 7,"CC": 8,"CD": 9,"DT": 10,"EX": 11,"FW": 12,"IN": 13,"JJ": 14,"JJR": 15,"JJS": 16,"LS": 17,"MD": 18,"NN": 19,"NNP": 20,"NNPS": 21,"NNS": 22,"PDT": 23,"POS": 24,"PRP": 25,"PRP$": 26,"RB": 27,"RBR": 28,"RBS": 29,"SYM": 30,"TO": 31,"UH": 32,"VB": 33,"VBD": 34,"VBG": 35,"VBN": 36,"VBP": 37,"VBZ": 38,"WDT": 39,"WP": 40,"WP$": 41,"WRB": 42,"``": 43 }
    onehot = np.zeros(shape=(44,)).tolist()
    onehot[dic[pos]] = 1
    return onehot