# =========================
# 保存前预测
# =========================
import torch
import torch.nn as nn
from torchvision import datasets , transforms
from torch.utils.data import DataLoader 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()   #一个是类，一个是函数  torch.relu;
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   #没写 padding，默认也是 padding=0。
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
val_base = datasets.CIFAR10(root='./data',
                            train=True,
                            download=False,
                            transform=val_transform)

val_loader = DataLoader(val_base,
                        batch_size=64)
model = CNN()
model.to(device)
model.eval()

images, labels = next(iter(val_loader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():

    logits_before = model(images)

    pred_before = torch.argmax(
        logits_before,
        dim=1
    )


# =========================
# 保存 checkpoint
# =========================

torch.save(
    model.state_dict(),
    "best_model.pth"
)


# =========================
# 创建一个新的模型
# =========================

new_model = CNN(
    num_classes=10
).to(device)


# =========================
# 加载 checkpoint
# =========================

new_model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

new_model.eval()


# =========================
# 加载后预测
# =========================

with torch.no_grad():

    logits_after = new_model(images)

    pred_after = torch.argmax(
        logits_after,
        dim=1
    )


# =========================
# 比较预测结果
# =========================

print(
    torch.equal(
        pred_before,
        pred_after
    )
)