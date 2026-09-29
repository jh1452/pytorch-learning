import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
import random
import numpy as np

lr = 0.001
device = ("cuda" if torch.cuda.is_available() else "cpu")
mean =(0.5,0.5,0.5)

std = (0.5,0.5,0.5)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



class CNN(nn.Module):
    def __init__(self):
        super().__init__
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=4,stride=2,padding=1)
        self.fc1  = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,10)
        self.dropout_con2d = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout_con2d(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout_con2d(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])


train_base = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,

)

val_base = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=val_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)
length = len(train_base)

indices = torch.randperm(length).tolist()

split = int (length*0.8)

train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(train_base,train_indices)
val_dataset = Subset(val_base,val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

model = CNN().to(device)

best_val_accrucy = 0.0

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)



for epoch in range(100):
    model.train()
    train_loss = 0.0
    for data,lable in train_loader:
        data = data.to(device)
        lable = lable.to(device)
        optimizer.zero_grad()
        logists = model(data)
        loss = criteria(logists,lable)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_dataset)


      