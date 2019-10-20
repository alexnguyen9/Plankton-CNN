# Import Libraries
import torch
import pandas as pd
import torch.optim as optim

# Get models
from models import *
from train_and_evaluate import *
from get_data import *




# Get models

# Set models
model1 = Net1()
model2 = Net2()
model3 = Net3()
model4 = Net4()
model5 = Net5()

# Set to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)
model5.to(device)

# Set Optimiziers
optimizer1 = optim.SGD(model1.parameters(), lr=0.01,momentum=0.9)
optimizer2 = optim.SGD(model2.parameters(), lr=0.01,momentum=0.9)
optimizer3 = optim.SGD(model3.parameters(), lr=0.01,momentum=0.9)
optimizer4 = optim.SGD(model4.parameters(), lr=0.004,momentum=0.9)
optimizer5 = optim.SGD(model5.parameters(), lr=0.004,momentum=0.9)




# dataframes to hold results

train_loss = pd.DataFrame(index=[x for x in range(50)])
train_acc  = pd.DataFrame(index=[x for x in range(50)])
val_loss   = pd.DataFrame(index=[x for x in range(50)])
val_acc    = pd.DataFrame(index=[x for x in range(50)])

# 50 epochs
n_epochs = 50



# for each of the 5 models
for i in range(1,5+1):
    
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []

    
    for epoch in range(n_epochs):
        
        #train, evalulate models
        train_model(epoch,eval('model'+str(i)),eval('optimizier'+str(i)),train_loader16)
        evaluate(train_loader16,eval('model'+str(i)),t_loss,t_acc)
        evaluate_val(validation_loader,eval('model'+str(i)),v_loss,v_acc)
    
    #Store Results
    train_loss['model'+str(i)] = t_loss
    train_acc['model'+str(i)]  = t_acc
    val_loss['model'+str(i)]   = v_loss
    val_acc['model'+str(i)]    = v_acc

