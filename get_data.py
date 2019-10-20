import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# Class to get data
class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)




## Hold transformations
        
# No transformations
transform_notrans = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((80,80)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.95,), std=(0.2,))])


# Only flips
transform_flips = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.Resize((80,80)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.95,), std=(0.2,))])


#Flips/rotations/translation
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=255),
     transforms.Resize((80,80)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.95,), std=(0.2,))])



    
# load the data
data = ImageFolder(root='data/train')


validation_size = 5000

# Train and validation set
train,validation = torch.utils.data.random_split(data, (len(data)-validation_size,validation_size))


# Set validations (no transforms)
validation = MyDataset(validation,transform = transform_notrans)
validation_loader = torch.utils.data.DataLoader(validation, batch_size=16, shuffle = True, num_workers=0)




##  Set training sets with various transformations

# no transformations
train_notrans = MyDataset(train,transform = transform_notrans)

# with flips
train_flips = MyDataset(train,transform = transform_flips)

# flips/rotations/translations
train = MyDataset(train,transform=transform)



## Create different train loaders with different batch sizes

train_loader16 = torch.utils.data.DataLoader(train, batch_size=16, shuffle = True, num_workers=4)
train_loader32 = torch.utils.data.DataLoader(train, batch_size=32, shuffle = True, num_workers=4)
train_loader48 = torch.utils.data.DataLoader(train, batch_size=48, shuffle = True, num_workers=4)
train_loader64 = torch.utils.data.DataLoader(train, batch_size=64, shuffle = True, num_workers=4)



## Try with differen transformations

# No transformations
train_loader_notrans = torch.utils.data.DataLoader(train_notrans, batch_size=32, shuffle = True, num_workers=0)
# Only flips
train_loader_flips = torch.utils.data.DataLoader(train_flips, batch_size=32, shuffle = True, num_workers=0)
# With flips/rotations/translation
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle = True, num_workers=0)

