Because the model is too big to upload to github directly, I upload the model to dropbox. Hence, before start testing, as follows if currently in the same directory with the *READMD.md*:

wget -P ./model/train_model/ https://www.dropbox.com/s/qbej7xjhkxwtlw6/model.h5?dl=1
wget -P ./model/train_model/ https://www.dropbox.com/s/vwhrnz5hlshnjk7/word2vec.model?dl=1
mv ./model/train_model/model.h5?dl=1 ./model/train_model/model.h5
mv ./model/train_model/word2vec.model?dl=1 ./model/train_model/word2vec.model

train

>python3 main.py train_model train

test:

>python3 main.py train_model test --load_model train_model

test data scores(loss = mse) = 0.942384

references:

https://ntumlta.github.io/2017fall-ml-hw4/RNN_model.html

https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras