# NLP2018 project2 - Semantic Relations by pytorch implementation

## Requirement
* python 3.6
* pytorch  0.3.1
* tensorflow 1.3.0
* numpy 1.13.0
* nltk 3.2.5
* sklearn 0.19.1

## Note
* please put **[
GoogleNews-vectors-negative300.bin.gz]** (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM) in the parent directory of Directory *text_cnn_pytorch_version*, e.g. `../GoogleNews-vectors-negative300.bin`
* please put *TRAIN_FILE.txt* and *TEST_FILE.txt* in Directory *data* and put Directory *data* in the parent directory of Directory *text_cnn_pytorch_version*, e.g. `../data/TRAIN_FILE.txt` and `../data/TEST_FILE.txt`
* generate *ans_textcnn_pytorch.txt* as final prediction result in the current working directory, *text_cnn_pytorch_version*

## Usage
### train
    ```bash
    $ python3 main.py train
    ```
### test
    ```bash
    $ bash execute.sh
    ```

## references:
* **CNN text classification Pytorch implementation** [[github]](https://github.com/Shawn1993/cnn-text-classification-pytorch)
* **Convolutional Neural Networks for Sentence Classification** [[paper]](https://arxiv.org/abs/1408.5882)
* **ResCNN Relation Extraction Tensorflow implementation** [[github]](https://github.com/darrenyaoyao/ResCNN_RelationExtraction)
* **Convolutional Neural Networks for Relation Extraction Tensorflow implementation** [[github]](https://github.com/roomylee/cnn-relation-extraction)
* **Relation Classification via Convolutional Deep Neural Network** (COLING 2014), D Zeng et al. **[[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Relation_Classification_via_Convolutional_Deep_Neural_Network.md)** [[paper]](http://www.aclweb.org/anthology/C14-1220)



