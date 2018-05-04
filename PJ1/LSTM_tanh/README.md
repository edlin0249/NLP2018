train

>python3 main.py train_model train --cell LSTM

test:

>python3 main.py train_model test --cell LSTM --load_model train_model

test data scores(loss = mse) = 0.095292

references:

https://ntumlta.github.io/2017fall-ml-hw4/RNN_model.html

https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras