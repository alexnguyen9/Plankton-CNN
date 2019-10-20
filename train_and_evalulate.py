
import torch
import torch.nn as nn
import torch.nn.functional as F


# Set cross entropy loss

criterion = nn.CrossEntropyLoss()



#Code is initially from : https://www.kaggle.com/juiyangchang/cnn-with-pytorch-0-995-accuracy

# function to train model, show loss after every 300 batches
def train_model(epoch,model,optimizer,train_loader):
    model.train()
    #exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))



# evaluate training set at the end of every epoch
def evaluate(data_loader,model,t_loss,t_acc):
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += F.cross_entropy(output, target, reduction="sum").item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
        
    print('\nAverage Train loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    t_loss.append(loss)
    t_acc.append(correct.item()/len(data_loader.dataset))



#evaluate validation set at the end of every epoch
def evaluate_val(data_loader,model,v_loss,v_acc):
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        #data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += F.cross_entropy(output, target, reduction="sum").item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
        
    print('\nAverage Validation loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    

    v_loss.append(loss)
    v_acc.append(correct.item()/len(data_loader.dataset))

