import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

os.makedirs("./cnn/checkpoints", exist_ok=True)
lr = 0.001
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
  
    torch.cuda.manual_seed_all(42)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout_conv = nn.Dropout2d(p=0.25)   # 卷积后用 Dropout2d
        self.dropout_fc = nn.Dropout(p=0.5)        # 全连接用普通 Dropout

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout_conv(x)                   # 卷积块后
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout_conv(x)                   # 卷积块后
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)                     # fc1 后
        x = self.fc2(x)
        return x

"""
RandomCrop 和 RandomHorizontalFlip 在 PIL Image 上操作，放在 ToTensor 之前更自然
（虽然 RandomHorizontalFlip 对 Tensor 也支持，但 RandomCrop 对 Tensor 支持有限/行为不同）。

Normalize 必须在 ToTensor 之后，因为它需要 Tensor 输入。
你原来的写法把 Normalize 放在了 ToTensor 后面勉强可以，但顺序反了会更麻烦。
"""

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),  #相当于平移图片
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_base = datasets.CIFAR10(root='./data', 
                              train=True, 
                              download=True, 
                              transform=train_transform)
val_base = datasets.CIFAR10(root='./data',
                            train=True,
                            download=False,
                            transform=val_transform)   #训练不需要数据增强
num_samples = len(train_base)

indices = torch.randperm(num_samples).tolist()

split = int(num_samples * 0.8)

train_indices = indices[:split]
val_indices = indices[split:]
train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)
test_dataset = datasets.CIFAR10(root='./data', 
                                train=False, 
                                download=True, 
                                transform=test_transform)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                        )

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                       )

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                        )

model = CNN(num_classes=10).to(device)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_accuracy = 0.0

for epoch in range(200):

    

    train_loss = 0.0
    #训练阶段
    model.train()
    
    for data, label in train_loader:

        # 数据放到 GPU / CPU
        data = data.to(device)
        label = label.to(device)  

        # 1. 清空上一轮梯度
        optimizer.zero_grad()

        # 2. 前向传播
        logits = model(data)

        # 3. 计算损失
        loss = criteria(logits, label)  #label里面有什么内容

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        train_loss += loss.item()*data.size(0)  #train_loss += loss.item() * data.size(0)
    train_loss /= len(train_dataset)
    #区别在于：你到底是在算“每个 batch 的平均 loss”，还是“所有样本的平均 loss”。
    # 验证阶段

    model.eval()  

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for data, label in val_loader:

            data = data.to(device)
            label = label.to(device)

            logits = model(data)

            loss = criteria(logits, label)

            val_loss += loss.item()*data.size(0)

            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == label).sum().item()

            total += label.size(0)

    val_loss /= len(val_dataset)

    val_accuracy = correct / total
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        """

保存的是优化器自己的信息。比如你用：
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


Adam 不只是需要模型参数，它还会保存每个参数的：
一阶动量
二阶动量
当前 step
学习率等参数

所以 Adam 的 state_dict() 里大概会有：
{    "state": {        0: {            "step": ...,            "exp_avg": ...,            "exp_avg_sq": ...        },        ...    },    "param_groups": [        {            "lr": 0.001,            "betas": (0.9, 0.999),            ...        }    ]}


这也是为什么断点续训时通常两个都要保存：

加载时：
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

        """

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                #"optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
            },
            "./cnn/checkpoints/best_checkpoint.pth"
        )

        print(
            f"Best checkpoint saved! "
            f"Val Acc: {val_accuracy:.4f}"
        )
    print(
        f"Epoch [{epoch+1}/200] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )



# =========================
# 加载最佳 checkpoint
# =========================

checkpoint = torch.load(
    "./cnn/checkpoints/best_checkpoint.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Loaded best checkpoint "
    f"from epoch {checkpoint['epoch'] + 1}, "
    f"Val Acc: {checkpoint['val_accuracy']:.4f}"
)

model.eval()


test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for data, label in test_loader:

        data = data.to(device)
        label = label.to(device)

        logits = model(data)

        loss = criteria(logits, label)

        test_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == label).sum().item()

        total += label.size(0)

test_loss /= len(test_loader)

test_accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

