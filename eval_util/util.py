from sklearn.metrics import mean_squared_error,f1_score

def evalMatrix(ground,pred,mode):
    if mode == 'mse':
        return mean_squared_error(ground, pred)
    elif mode == 'f1_score':
        return f1_score(ground,pred)