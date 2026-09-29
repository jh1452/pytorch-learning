import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# ============================================================
# 1. 基本配置
# ============================================================

SEED = 42

BATCH_SIZE = 32

DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

TRAIN_SIZE = 45000
VAL_SIZE = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 固定随机种子
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. normalization 参数
# ============================================================

# 这里先使用比较简单的归一化方式：
#
# 原始 ToTensor() 之后：
# 像素范围约为 [0, 1]
#
# Normalize 后：
# x_new = (x - mean) / std
#
# 当 mean = 0.5，std = 0.5 时：
# [0, 1] -> [-1, 1]

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


# ============================================================
# 4. train_transform
# ============================================================

train_transform = transforms.Compose([

    # 随机水平翻转
    # 50% 概率左右翻转图片
    transforms.RandomHorizontalFlip(),

    # PIL Image -> Tensor
    #
    # shape:
    # H x W x C
    # ->
    # C x H x W
    #
    # pixel:
    # [0,255]
    # ->
    # [0,1]
    transforms.ToTensor(),

    # 标准化
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# 5. val_transform
# ============================================================

# Validation 不使用随机增强
#
# 因为验证集的目标是：
# 稳定、公平地评估模型

val_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# 6. 加载 CIFAR-10
# ============================================================

# 注意：
#
# train_base 和 val_base
# 都来自 CIFAR-10 官方训练集
#
# 区别只是 transform 不一样

train_base = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    transform=train_transform,
    download=True
)

 
test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    transform=val_transform,
    download=True   #这个为什么需要下载？
)


# CIFAR-10 类别名称
classes = train_base.classes

print("Classes:")
print(classes)


# ============================================================
# 7. 生成固定 train_indices / val_indices
# ============================================================

total_size = len(train_base)

print("\nOfficial training set size:")
print(total_size)


# 创建一个单独的随机数生成器
generator = torch.Generator()

generator.manual_seed(SEED)


# 生成 0 ~ 49999 的随机排列
indices = torch.randperm(
    total_size,
    generator=generator
)


# 前 5000 个做 Validation
val_indices = indices[:VAL_SIZE]


# 剩余 45000 个做 Train
train_indices = indices[VAL_SIZE:]


print("\nTrain indices shape:")
print(train_indices.shape)

print("Validation indices shape:")
print(val_indices.shape)


# ============================================================
# 8. 保存数据划分索引
# ============================================================

split_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_split_indices.pt"
)

torch.save(
    {
        "seed": SEED,
        "train_indices": train_indices,
        "val_indices": val_indices
    },
    split_path
)

print("\nSaved split indices to:")
print(split_path)


# ============================================================
# 9. 创建 train / val Dataset
# ============================================================

train_dataset = Subset(
    train_base,
    train_indices
)

val_dataset = Subset(
    val_base,
    val_indices
)


# ============================================================
# 10. 创建 DataLoader
# ============================================================

train_loader = DataLoader(
    dataset=train_dataset,

    batch_size=BATCH_SIZE,

    # 训练集通常打乱
    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(
    dataset=val_dataset,

    batch_size=BATCH_SIZE,

    # 验证集不需要打乱
    shuffle=False,

    num_workers=0
)


test_loader = DataLoader(
    dataset=test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 11. 打印数据集大小
# ============================================================

print("\n================ Dataset Size ================")

print(
    "Train size:",
    len(train_dataset)
)

print(
    "Validation size:",
    len(val_dataset)
)

print(
    "Test size:",
    len(test_dataset)
)


# ============================================================
# 12. 取一个 Batch
# ============================================================

images, labels = next(
    iter(train_loader)
)    


# ============================================================
# 13. 检查 shape / dtype / value range
# ============================================================

print("\n================ Batch Check ================")

print(
    "Images shape:",
    images.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Images dtype:",
    images.dtype
)

print(
    "Labels dtype:",
    labels.dtype
)

print(
    "Images min:",
    images.min().item()
)

print(
    "Images max:",
    images.max().item()
)

print(
    "First 10 labels:",
    labels[:10]
)


print("\nFirst 10 class names:")

for label in labels[:10]:

    print(
        classes[label.item()]
    )


# ============================================================
# 14. 反归一化
# ============================================================

def denormalize(images, mean, std):
    """
    把经过 Normalize 的图片恢复到接近原始 [0,1] 范围。

    Normalize:
        x_normalized = (x - mean) / std

    反归一化:
        x = x_normalized * std + mean
    """

    # 创建：
    #
    # [3]
    # ->
    # [1, 3, 1, 1]
    #
    # 这样可以和：
    #
    # [N, 3, H, W]
    #
    # 广播计算

    mean = torch.tensor(
        mean,
        dtype=images.dtype,
        device=images.device
    ).view(1, 3, 1, 1)        #把 Python 的元组 tuple 转成 PyTorch 的 Tensor。

    std = torch.tensor(
        std,
        dtype=images.dtype,
        device=images.device
    ).view(1, 3, 1, 1)

    images = images * std + mean

    # 防止出现略微超出 [0,1] 的情况
    images = images.clamp(
        0.0,
        1.0
    )

    return images


display_images = denormalize(
    images,
    MEAN,
    STD
)


# ============================================================
# 15. 保存 32 张样本网格
# ============================================================

# 32 张图片
#
# 每行 8 张
#
# 形成：
#
# 8 x 4 网格

grid = make_grid(
    display_images,
    nrow=8,
    padding=2
)


grid_path = os.path.join(
    OUTPUT_DIR,
    "cifar10_32_samples.png"
)


save_image(
    grid,
    grid_path
)


print("\nSaved sample grid to:")
print(grid_path)


# ============================================================
# 最后打印数据处理信息
# ============================================================

print("\n================ Data Summary ================")

print("Normalization:")
print("Mean:", MEAN)
print("Std :", STD)

print("\nTraining transform:")
print(train_transform)

print("\nValidation transform:")
print(val_transform)

print("\nSplit:")
print(
    f"Train = {len(train_dataset)}"
)

print(
    f"Validation = {len(val_dataset)}"
)

print(
    f"Test = {len(test_dataset)}"
)