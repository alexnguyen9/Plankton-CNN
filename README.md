# Plankton-CNN
Classifying Plankton with convolutional neural networks



1. Make sure you download the kaggle "train" data and put it into the data folder, the data from kaggle can be downloaded [here](https://www.kaggle.com/c/datasciencebowl)

2. Run `get_data.py` to load the data from the train folder and load it into Pytorch's DataLoader.

* Run `test_transforms.py` to test out different data augmentation techniques (flips and other transforms)
* Run `test_batchsize.py` to test out different batch sizes during training
* Run `train_and_evalulate.py` to evaluate the 5 models
* Run `different_optimizers.py` to evaluate different optimization algorithms
