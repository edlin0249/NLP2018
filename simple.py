import json
from pprint import pprint
from sklearn.metrics import mean_squared_error
import numpy as np
'''
Json format
“tweet”: analyzed tweet
“target”: targeted cashtag
“snippet”: key snippet for
targeted cashtag
“sentiment”: sentiment score

'''


trainP = 0.9


def loadData():
	trainData = json.load(open('NLP_Project 1/training_set.json'))
	testData = json.load(open('NLP_Project 1/test_set.json'))
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

if __name__ == '__main__':
	
	trainData, validData, testData = loadData()
	scoreDistribution(trainData+validData+testData)

	d = simpleReg(trainData)

	pred = []
	for i in validData:
		if i['target'] in d: 
			pred.append(d[i['target']])
		else:
			pred.append(0)

	pprint(evaluation(pred,[float(i['sentiment']) for i in validData]))