import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. 随机种子
# ============================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 3. Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Main Branch
        #
        # x
        # ↓
        # Conv3x3
        # ↓
        # BN
        # ↓
        # ReLU
        # ↓
        # Conv3x3
        # ↓
        # BN
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # ----------------------------------------------------
        # Shortcut
        #
        # 如果 shape 一样：
        #
        # x -------------------->
        #
        # 使用 Identity
        #
        #
        # 如果 channel 或 H/W 改变：
        #
        # x
        # ↓
        # Conv1x1
        # ↓
        # BN
        #
        # 使用 Projection Shortcut
        # ----------------------------------------------------

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()


    def forward(self, x):

        # ----------------------------------------------------
        # Shortcut Branch
        # ----------------------------------------------------

        identity = self.shortcut(x)

        # ----------------------------------------------------
        # Main Branch
        # ----------------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        # ----------------------------------------------------
        # Residual Addition
        #
        # y = F(x) + shortcut(x)
        # ----------------------------------------------------

        out = out + identity

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        out = self.relu(out)

        return out


# ============================================================
# 4. Small ResNet
# ============================================================

class SmallResNet(nn.Module):

    def __init__(self,num_classes=10):
        super().__init__()

        # ----------------------------------------------------
        # Stem
        #
        # [N, 3, 32, 32]
        #
        #       ↓
        #
        # [N, 32, 32, 32]
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # Stage 1
        #
        # [N,32,32,32]
        #
        #       ↓
        #
        # [N,32,32,32]
        #
        # 不改变 channel
        # 不改变 H/W
        # ----------------------------------------------------

        self.stage1 = nn.Sequential(

            ResidualBlock(
                32,
                32,
                stride=1
            ),

            ResidualBlock(
                32,
                32,
                stride=1
            )
        )


        # ----------------------------------------------------
        # Stage 2
        #
        # [N,32,32,32]
        #
        #       ↓
        #
        # [N,64,16,16]
        #
        # channel:
        # 32 → 64
        #
        # H/W:
        # 32 → 16
        # ----------------------------------------------------

        self.stage2 = nn.Sequential(

            ResidualBlock(
                32,
                64,
                stride=2
            ),

            ResidualBlock(
                64,
                64,
                stride=1
            )
        )


        # ----------------------------------------------------
        # Stage 3
        #
        # [N,64,16,16]
        #
        #       ↓
        #
        # [N,128,8,8]
        # ----------------------------------------------------

        self.stage3 = nn.Sequential(

            ResidualBlock(
                64,
                128,
                stride=2
            ),

            ResidualBlock(
                128,
                128,
                stride=1
            )
        )


        # ----------------------------------------------------
        # Global Average Pool
        #
        # [N,128,8,8]
        #
        #       ↓
        #
        # [N,128,1,1]
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        # ----------------------------------------------------
        # FC
        #
        # [N,128]
        #
        #    ↓
        #
        # [N,10]
        # ----------------------------------------------------

        self.fc = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc(x)

        return x


# ============================================================
# 5. Shape Check
# ============================================================

def shape_check():

    print("\n========== Shape Check ==========")

    model = SmallResNet().to(device)

    x = torch.randn(
        4,
        3,
        32,
        32
    ).to(device)

    print(
        "Input: ",
        x.shape
    )

    # --------------------------------------------------------
    # 手动经过每一部分
    # 方便观察 Shape
    # --------------------------------------------------------

    x = model.stem(x)

    print(
        "Stem:  ",
        x.shape
    )

    x = model.stage1(x)

    print(
        "Stage1:",
        x.shape
    )

    x = model.stage2(x)

    print(
        "Stage2:",
        x.shape
    )

    x = model.stage3(x)

    print(
        "Stage3:",
        x.shape
    )

    x = model.avgpool(x)

    print(
        "AvgPool:",
        x.shape
    )

    x = torch.flatten(
        x,
        1
    )

    print(
        "Flatten:",
        x.shape
    )

    x = model.fc(x)

    print(
        "Output:",
        x.shape
    )


# ============================================================
# 6. Parameter Count
# ============================================================

def parameter_count(model):

    print("\n========== Parameter Count ==========")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        total_params
    )

    print(
        "Trainable parameters:",
        trainable_params
    )


# ============================================================
# 7. CIFAR-10
# ============================================================

def get_small_cifar10():

    print("\n========== Loading CIFAR-10 ==========")

    # --------------------------------------------------------
    # 128 样本过拟合实验：
    #
    # 不使用数据增强。
    #
    # 因为我们的目标不是泛化，
    # 而是检查模型能不能记住固定的 128 张图片。
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.4914,
                0.4822,
                0.4465
            ),

            std=(
                0.2470,
                0.2435,
                0.2616
            )
        )
    ])


    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    # --------------------------------------------------------
    # 固定选择 128 张图片
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    )[:128]


    small_dataset = Subset(
        train_dataset,
        indices.tolist()
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    small_loader = DataLoader(
        small_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )


    print(
        "Small dataset size:",
        len(small_dataset)
    )

    return small_loader


# ============================================================
# 8. 128-sample Overfit Experiment
# ============================================================

def overfit_128():

    print("\n========== 128 Sample Overfit ==========")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader = get_small_cifar10()


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SmallResNet(
        num_classes=10
    ).to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # --------------------------------------------------------
    # Epoch
    #
    # 128 张图的目的就是尝试把训练集拟合得非常高。
    # 如果 30 epoch 不够，可以继续增加。
    # --------------------------------------------------------

    num_epochs = 30


    # --------------------------------------------------------
    # 保存训练曲线
    # --------------------------------------------------------

    loss_history = []

    acc_history = []


    # ========================================================
    # Training
    # ========================================================

    for epoch in range(num_epochs):

        # ----------------------------------------------------
        # Train Mode
        #
        # BatchNorm 使用当前 batch 的统计量
        # ----------------------------------------------------

        model.train()


        total_loss = 0.0

        correct = 0

        total = 0


        # ====================================================
        # 一个 epoch
        # ====================================================

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # ------------------------------------------------
            # Step 1
            #
            # 清空上一轮梯度
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # Step 2
            #
            # Forward
            # ------------------------------------------------

            logits = model(images)


            # ------------------------------------------------
            # Step 3
            #
            # Cross Entropy
            #
            # logits:
            #
            # [batch_size, 10]
            #
            # labels:
            #
            # [batch_size]
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )


            # ------------------------------------------------
            # Step 4
            #
            # Backward
            #
            # 计算：
            #
            # ∂L / ∂W
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # Step 5
            #
            # Update Parameters
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # 统计 Loss
            # ------------------------------------------------

            total_loss += (
                loss.item()
                *
                images.size(0)
            )


            # ------------------------------------------------
            # 预测类别
            # ------------------------------------------------

            predictions = logits.argmax(
                dim=1
            )


            # ------------------------------------------------
            # 统计正确数量
            # ------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        # ====================================================
        # Epoch Result
        # ====================================================

        train_loss = (
            total_loss
            /
            total
        )


        train_acc = (
            correct
            /
            total
        )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        loss_history.append(
            train_loss
        )

        acc_history.append(
            train_acc
        )


        # ----------------------------------------------------
        # 打印
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {train_acc * 100:.2f}%"
        )


    # ========================================================
    # Training finished
    # ========================================================

    print("\nTraining finished.")


    # --------------------------------------------------------
    # 最后在相同 128 张图片上检查一次
    # --------------------------------------------------------

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    final_acc = correct / total


    print(
        f"Final training accuracy: "
        f"{final_acc * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Plot Training Loss
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        range(1, num_epochs + 1),
        loss_history
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        "128-Sample Overfit - Loss"
    )

    plt.grid()

    plt.show()


    # --------------------------------------------------------
    # Plot Training Accuracy
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        range(1, num_epochs + 1),
        acc_history
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Accuracy"
    )

    plt.title(
        "128-Sample Overfit - Accuracy"
    )

    plt.grid()

    plt.show()


    return model


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Shape Check
    # --------------------------------------------------------

    shape_check()


    # --------------------------------------------------------
    # 2. Parameter Count
    # --------------------------------------------------------

    model = SmallResNet().to(device)

    parameter_count(
        model
    )


    # --------------------------------------------------------
    # 3. 128 Sample Overfit
    # --------------------------------------------------------

    trained_model = overfit_128()