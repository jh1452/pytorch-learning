import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt
# ============================================================

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平输入   
        x = torch.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)  # 输出层
        return x

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

dataset = datasets.CIFAR10(root='./data', 
                           train=True, 
                           download=True, 
                           transform=transform)

Subdataset = Subset(dataset, indices=range(128))  # 取前128个样本作为子数据集

dataloader = DataLoader(Subdataset, batch_size=128, shuffle=False)  # 创建数据加载器

for images, labels in dataloader:
    print("Images shape:", images.shape)  # 输出图像的形状
    print("Labels shape:", labels.shape)  # 输出标签的形状
    break  # 只取一个批次进行演示

model = mlp()  # 实例化模型

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

losses = []  # 用于存储每个epoch的损失值
accuracies = []  # 用于存储每个epoch的准确率
criterion = nn.CrossEntropyLoss() 
for epoch in range(100):
    for images, labels in dataloader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        # 定义损失函数
        loss = criterion(outputs, labels)  # 计算损失
        predictions = outputs.argmax(dim=1)

        accuracy = (
            predictions == labels
        ).float().mean()
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")  # 打印损失和准确率

epochs = range(1, len(losses)+1)  # 定义x轴的范围为1到100
plt.plot(epochs, losses, marker='o' )  # 绘制损失曲线
plt.title('Training Loss over Epochs')  # 设置标题
plt.xlabel('Epochs')  # 设置x轴标签 
plt.ylabel('Loss')  # 设置y轴标签
plt.show()  # 显示图像


plt.plot(epochs, accuracies, marker='o')  # 绘制准确率曲线
plt.title('Training Accuracy over Epochs')  # 设置标题  

plt.xlabel('Epochs')  # 设置x轴标签
plt.ylabel('Accuracy')  # 设置y轴标签
plt.show()  # 显示图像


   

