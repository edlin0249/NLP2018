import json
from pprint import pprint
from sklearn.metrics import mean_squared_error
import numpy as np
import re
from util import DataManager

'''

Json format
“tweet”: analyzed tweet
“target”: targeted cashtag
“snippet”: key snippet for targeted cashtag
“sentiment”: sentiment score

'''

VOCSIZE=1000
def prepData(vocsize=VOCSIZE,maxlen=20):
	D = DataManager()
	D.add_data('trainData','data/training_set.json')
	D.add_data('testData','data/test_set.json')
	D.tokenize(vocsize)
	D.to_sequence(maxlen)
	return D


def evaluation(pred,ground,mode='mse'):
	if mode == 'mse':
		return mean_squared_error(ground, pred)

def buildLinearSystem(D,vocsize=VOCSIZE):
	b = np.array(D.get_data('trainData')[1]).reshape(-1,1)
	A = np.zeros(shape=(b.shape[0],vocsize))
	cnt = np.zeros(shape=b.shape)
	for idx,single in enumerate(D.get_data('trainData')[0]):
		for ele in single:
			A[idx,ele] += 1
			cnt[idx,0] += 1

	# add the constrain
	A = np.concatenate( (A, np.ones( shape=(1,A.shape[1]) ) ), axis=0)
	b = np.concatenate( (b, np.array([[1]])), axis=0)
	cnt = np.concatenate( (cnt, np.array( [[vocsize]] )), axis=0)
	
	return A,np.multiply(b,cnt)

def inference(D,scoreArr):
	ans = []
	for single in D.get_data('testData')[0]:
		scoreSum = 0
		cnt = 0
		for ele in single:
			scoreSum = scoreArr[ele]
			cnt += 1
		ans.append(scoreSum / cnt)
	return ans

if __name__ == '__main__':
	
	D = prepData()
	A,b = buildLinearSystem(D)
	x = np.clip(np.linalg.lstsq(A,b)[0],-1,1) # solve the linear system
	# the value calculated by LS might over the [-1,1]
	ans = inference(D,x)
	print(evaluation(ans,D.get_data('testData')[1]))
