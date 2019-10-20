# Import Libraries

import torch
import pandas as pd

from models import Net1
from train_and_evalulate import *
from get_data import *



# Set up GPU integration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Only testing model 1
model1 = Net1()
model1.to(device)




# number of epochs
n_epochs = 50

# empty lists to hold results for each batch size
loss_16 = []
loss_32 = []
loss_48 = []
loss_64 = []
acc_16 = []
acc_32 = []
acc_48 = []
acc_64 = []


# emppty dataframe to hold all validation loss and accuracy results
val_batch_loss = pd.DataFrame(index=[x for x in range(50)])
val_batch_acc = pd.DataFrame(index=[x for x in range(50)])




for num_batch in ['16','32','48','64']:
    
    for layer in model1.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()

    for epoch in range(n_epochs):
        train_model(epoch,model1,optimizer1,eval('train_loader' + num_batch))
        evaluate(eval('train_loader' + num_batch),[],[]) # ignore training loss and validation
        evaluate_val(validation_loader,model1,eval('loss_'+num_batch),eval('acc_'+num_batch))
        



val_batch_loss.assign(b16 = loss_16,b32 = loss_32,b48 = loss_48,b64 = loss_64)
val_batch_acc.assign(b16 = acc_16,b32 = acc_32,b48 = acc_48,b64 = acc_64)

