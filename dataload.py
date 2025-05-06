import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 定义数据增强与预处理操作（训练 + 测试）
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 下载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# CIFAR-10中的标签索引：{'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
# 只保留cat和dog类：label为3和5，转为0和1
def filter_cat_dog(dataset):
    cat_dog_indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
    subset = Subset(dataset, cat_dog_indices)

    # 重设label为0（cat）或1（dog）
    subset.dataset.targets = np.array(dataset.targets)
    for idx in cat_dog_indices:
        orig_label = dataset.targets[idx]
        dataset.targets[idx] = 0 if orig_label == 3 else 1

    return subset

train_subset = filter_cat_dog(train_dataset)
test_subset = filter_cat_dog(test_dataset)

# 构建 DataLoader
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
