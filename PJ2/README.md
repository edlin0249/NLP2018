
# NLP2018 project2 - Semantic Relations

### file
+ xgb_*.py - train & eval script (xgboost model)
+ svm_*.py - train & eval script (svm linear kernal model)
+ rfc_*.py - train & eval script (random forest model)
+ data_helpers_*.py - the data parser for each model 
+ ensemble.py - ensemble all result in ans_good/
+ ensemble.txt - the best prediction file by ensembling all model we do


### How to exec scripts

#### before you exec scripts, you need to download the word embedding from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM

train and eval: `python3 {model you want to use}.py` (will generate *.txt (* depends on model you selected) in current directory )
ensemble: `python3 ensemble.py` (will generate ensemble.txt in current directory)


### Requirement
+ tqdm 4.19.5
+ tensorflow 1.4.0
+ numpy 1.13.3
+ sklearn 0.19.1
+ nltk 3.2.5
+ gensim 3.1.0
+ pandas 0.21.0