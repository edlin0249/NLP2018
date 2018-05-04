import json
from pprint import pprint
import preprocessor as p ## need to install tweet-preprocessor

def loadData():
	trainData = json.load(open('data/training_set.json'))
	testData = json.load(open('data/test_set.json'))
	return trainData, testData
def getCorpus(D):
	corpus = []
	for i in D:
		corpus.append(p.clean(i['tweet']))
	return corpus


if __name__ == '__main__' :

	trainData,testData = loadData()
	corpus = getCorpus(trainData)
	for i in corpus:
		print(i)