Because the model is too big to upload to github directly, I upload the model to dropbox. Hence, before start testing, as follows if currently in the same directory with the *READMD.md*:

wget -P ./model/train_model/ https://www.dropbox.com/s/eakc4rkvchjnryk/model.h5?dl=1

wget -P ./model/train_model/ https://www.dropbox.com/s/oti4kqx54wd9c34/token.pk?dl=1

mv ./model/train_model/model.h5?dl=1 ./model/train_model/model.h5

mv ./model/train_model/token.pk?dl=1 ./model/train_model/token.pk

train

>python3 main.py train_model train --cell LSTM

test:

>python3 main.py train_model test --cell LSTM --load_model train_model

test data scores(loss = mse) = 0.525961

references:

https://ntumlta.github.io/2017fall-ml-hw4/RNN_model.html

https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras

https://github.com/philipperemy/keras-attention-mechanism/issues/14

https://stackoverflow.com/questions/48309322/keras-multiply-layer-in-functional-api

https://stackoverflow.com/questions/43977463/valueerror-could-not-broadcast-input-array-from-shape-224-224-3-into-shape-2