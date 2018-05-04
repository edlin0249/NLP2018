Because the model is too big to upload to github directly, I upload the model to dropbox. Hence, before start testing, as follows if currently in the same directory with the *READMD.md*:

wget -P ./model/train_model/ https://www.dropbox.com/s/q60j4il021z81k7/model.h5?dl=1

wget -P ./model/train_model/ https://www.dropbox.com/s/gqe8x7u8k9f5pyr/token.pk?dl=1

mv ./model/train_model/model.h5?dl=1 ./model/train_model/model.h5

mv ./model/train_model/token.pk?dl=1 ./model/train_model/token.pk

train

>python3 main.py train_model train

test:

>python3 main.py train_model test --load_model train_model

test data scores(loss = mse) = 0.116881

references:

https://ntumlta.github.io/2017fall-ml-hw4/RNN_model.html

https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras