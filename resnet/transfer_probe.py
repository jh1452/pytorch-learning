import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# 1. Pretrained weights
# =========================

weights = ResNet18_Weights.DEFAULT   #最好的这个权重


# =========================
# 2. Matching preprocessing
# =========================

preprocess = weights.transforms()

print(preprocess)


# =========================
# 3. Dataset
# =========================

train_set = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=preprocess
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)


# =========================
# 4. Load pretrained model
# =========================

model = resnet18(
    weights=weights
)


# =========================
# 5. Freeze backbone
# =========================

for param in model.parameters():
    param.requires_grad = False


# =========================
# 6. Replace classification head
# =========================



in_features = model.fc.in_features
model.fc = nn.Linear(
    in_features,
    10
)

model = model.to(device)


# =========================
# 7. Check trainable params
# =========================

for name, param in model.named_parameters():    #这个什么区别

    if param.requires_grad:
        print(
            "Trainable:",
            name,
            param.shape
        )


# =========================
# 8. Loss
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# 9. Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# =========================
# 10. Train
# =========================

for epoch in range(5):

      # Backbone 保持 eval 行为
    model.eval()

    # 分类头训练
    model.fc.train()

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss.item():.4f}"
    )