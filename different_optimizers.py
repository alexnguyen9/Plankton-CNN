
# Import Libraries

import torch
import pandas as pd
import torch.optim as optim

# Get models
from models import *
from train_and_evaluate import *
from get_data import *


# Set to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# we use Model 5

model5 = Net5()
model5.to(device)

# Initialize diffferent optimizers
optimizer5_nesterov = optim.SGD(model5.parameters(), lr=0.004,momentum=0.9,nesterov=True)
optimizer5_adagrad = optim.Adagrad(model5.parameters())
optimizer5_adam = optim.Adam(model5.parameters())



# empty dataframe to hold validation metrics
n_epochs = 50
Opt_val_loss  =  pd.DataFrame(index=[x for x in range(n_epochs)])
Opt_val_acc  =  pd.DataFrame(index=[x for x in range(n_epochs)])


for i  in ['nesterov','adagrad','adam']:
    
    #reset layers
    for layer in model5.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()

    # hold metrics for current optimizer
    v_loss = []
    v_acc = []

    for epoch in range(n_epochs):
        train_model(epoch,model5,eval('optimizer5_'+i),train_loader16)
        evaluate(train_loader16,model5,[],[])
        evaluate_val(validation_loader,model5,v_loss,v_acc)


    Opt_val_loss[i] = v_loss
    Opt_val_acc[i] = v_acc

