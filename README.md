# Plankton-CNN
Classifying Plankton with convolutional neural networks



1. Make sure you download the kaggle "train" data and put it into the data folder, the data from kaggle can be downloaded [here](https://www.kaggle.com/c/datasciencebowl)

2. Run `get_data.py` to load the data from the train folder and load it into Pytorch's DataLoader.

* Run `test_transforms.py` to test out different data augmentation techniques (flips and other transforms)
  * This returns the dataframes:
      * `trans_train_loss` contains the training loss for each of the transformations
      * `trans_train_acc` contains the training accuracy for each transformation
      * `trans_val_loss` contains the validation loss for each of the transformations
      * `trans_val_acc` contains the validation accuracy for each transformation
      
      
* Run `test_batchsize.py` to test out different batch sizes during training
  * This returns the dataframes:
      * `val_batch_loss` contains the validation loss for each of the batch sizes
      * `val_batch_acc` contains the accuracy loss for each of the batch sizes
      
* Run `train_models.py` to evaluate the 5 models
  * This returns the dataframes:
      * `train_loss` contains the training loss for each of the models
      * `train_acc` contains the training accuracy for each models
      * `val_loss` contains the validation loss for each of the models
      * `val_acc` contains the validation accuracy for each models


* Run `different_optimizers.py` to evaluate different optimization algorithms
  * This returns the dataframes:
      * `Opt_val_loss` contains the validation loss for the alternative optimization algorithms
      * `Opt_val_acc` contains the validation accuracy for the alternative optimization algorithms

A report and presentation can be found in the repo for a more complete explanation of what I did for the project.
