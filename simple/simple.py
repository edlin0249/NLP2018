import json
from pprint import pprint
from sklearn.metrics import mean_squared_error
import numpy as np
import re
from keras.preprocessing.text import Tokenizer

'''

Json format
“tweet”: analyzed tweet
“target”: targeted cashtag
“snippet”: key snippet for targeted cashtag
“sentiment”: sentiment score

'''


trainP = 0.9


def loadData():
	trainData = json.load(open('data/training_set.json'))
	testData = json.load(open('data/test_set.json'))
	trainNum = int(len(trainData) * trainP)
	return trainData[:trainNum], trainData[trainNum:], testData

def getTargetList(data):
	l = []
	for i in data:
		l.append(i['target'])
	return list(set(l))

def simpleReg(data):
	'''
		avg the sentiment score as prediction
	'''
	d = dict()
	cnt = dict()
	
	for i in data:
		target = i['target']
		senti = i['sentiment']
		if target in d:
			d[target] += float(senti)
			cnt[target] += 1
		else:
			d[target] = float(senti)
			cnt[target] = 1
	
	for i in d:
		d[i] = d[i] / cnt[i]
	return d

def simpleRegPred(d,dataset):
	pred = []
	for i in dataset:
		if i['target'] in d: 
			pred.append(d[i['target']])
		else:
			pred.append(0)

	print('Valid simpleReg: ',evaluation(pred,[float(i['sentiment']) for i in dataset]))
	print('Valid 0: ',evaluation([0]*len(dataset),[float(i['sentiment']) for i in dataset]))
	return 

def evaluation(pred,ground,mode='mse'):
	if mode == 'mse':
		return mean_squared_error(ground, pred)

def scoreDistribution(x):
	import matplotlib.pyplot as plt
	s = [float(i['sentiment']) for i in x]
	plt.hist(s, bins='auto')
	plt.title("Sentiment score distribution")
	plt.show()
	return


# Build dictionary
# vocab_size : maximum number of word in you dictionary
def tokenize(data,vocab_size=50000):
	print ('create new tokenizer')
	tokenizer = Tokenizer(num_words=vocab_size)
	print ('tokenizing')
	tokenizer.fit_on_texts(data)
	return tokenizer

# Convert words in data to index and pad to equal size
#  maxlen : max length after padding
def to_sequence(tokenizer, maxlen,data):

	tmp = (tokenizer.texts_to_sequences(data))
	# tmp = np.array(pad_sequences(tmp, maxlen=maxlen)) 

	return tmp

def genCorpus(data):
	corpus = [] 
	for i in data:
		sentence = ' '.join(np.array(i['snippet']).flatten().tolist()   )
		sentence = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sentence).split())
		corpus.append(sentence)
	return corpus

def parseWord(data):
	corpus = genCorpus(data)
	tokenizer = tokenize(corpus)

	for i in data:
		sentence = i['snippet']
		print(to_sequence(tokenizer,20,sentence))
		input()




if __name__ == '__main__':
	
	trainData, validData, testData = loadData()

	# d = simpleReg(trainData)

	parseWord(testData)