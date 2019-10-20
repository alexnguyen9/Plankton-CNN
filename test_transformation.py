# Import Libraries

import torch
import pandas as pd



from models import Net1
from train_and_evaluate import *
from get_data import *



# Set up GPU integration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Only testing model 1
model1 = Net1()
model1.to(device)



# dataframe to hold results of the transformations
train_loss = pd.DataFrame(index=[x for x in range(30)])
train_acc = pd.DataFrame(index=[x for x in range(30)])
val_loss = pd.DataFrame(index=[x for x in range(30)])
val_acc = pd.DataFrame(index=[x for x in range(30)])


no_trans_train_loss = [] 
no_trans_train_acc = []
no_trans_val_loss = [] 
no_trans_val_acc = []


flips_only_train_loss = [] 
flips_only_train_acc = []
flips_only_val_loss = [] 
flips_only_val_acc = []


all_transform_train_loss = [] 
all_transform_train_acc = []
all_transform_val_loss = [] 
all_transform_val_acc = []




# 30 epochs

# no transformation
for epoch in range(30):
    train_model(epoch,model1,optimizer1,train_loader_notrans)
    evaluate(train_loader_notrans,model1,no_trans_train_loss,no_trans_train_acc)
    evaluate_val(validation_loader,model1,no_trans_val_loss,no_trans_val_acc)

    
# for some reason the models dont reset so i have to reset them like this 
for layer in model1.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()
    
#flips only
for epoch in range(30):
    train_model(epoch,model1,optimizer1,train_loader_flips)
    evaluate(train_loader_notrans,model1,flips_only_train_loss,flips_only_train_acc)
    evaluate_val(validation_loader,model1,flips_only_val_loss,flips_only_val_acc)

# for some reason the models dont reset so i have to reset them like this
for layer in model1.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

#
for epoch in range(30):
    train_model(epoch,model1,optimizer1,train_loader)
    evaluate(train_loader_notrans,model1,all_transform_train_loss,all_transform_train_acc)
    evaluate_val(validation_loader,model1,all_transform_val_loss,all_transform_val_acc)



# Store results
train_loss.assign(no_trans_train_loss = no_trans_train_loss, flips_only_train_loss = flips_only_train_loss, all_transform_train_loss =all_transform_train_loss)
train_acc.assign(no_trans_train_acc = no_trans_train_acc, flips_only_train_acc = flips_only_train_acc, all_transform_train_acc=all_transform_train_acc)
val_loss.assign(no_trans_val_loss = no_trans_val_loss, flips_only_val_loss = flips_only_val_loss, all_transform_val_loss=all_transform_val_loss)
val_acc.assign(no_trans_val_acc = no_trans_val_acc, flips_only_val_acc = flips_only_val_acc, all_transform_val_acc = all_transform_val_acc)

