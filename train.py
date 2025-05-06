import torch
from torch import nn
import torchvision.models as models
from dataload import train_loader, test_loader
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math

LR_BASE = 0.0001          # 主干网络基础学习率（小）
LR_FC = 0.001             # FC层学习率（大）
EPOCH_NUM = 50
UNFREEZE_EPOCH = 10       # 第10轮解冻主干网络
LABEL_SMOOTHING = 0.1

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Dog_Cat_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Dog_Cat_Net, self).__init__()
        self.model = models.resnet50(pretrained=True)
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=fc_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

model = Dog_Cat_Net(num_classes=2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 冻结主干网络
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# Criterion with Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# 区分学习率：FC 层高学习率，主干低学习率
fc_params = []
backbone_params = []
for name, param in model.named_parameters():
    if "fc" in name:
        fc_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.Adam([
    {'params': backbone_params, 'lr': LR_BASE},
    {'params': fc_params, 'lr': LR_FC}
])

# ReduceLROnPlateau：动态调整学习率（监控 val loss）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def train(epoch):
    model.train()
    running_loss = 0.0
    count = 0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        count += 1
        if (batch_idx + 1) % (len(train_loader) // 5) == 0:
            print(f'[{time_since(start)}] Epoch {epoch+1}, Step {batch_idx+1}, Loss: {running_loss / count:.3f}')
    return running_loss / count

def test(epoch, acc_list):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (prediction == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    print(f'Accuracy on test set: ({correct}/{total}) {accuracy:.2f}%')
    acc_list.append(accuracy)
    with open("test.txt", "a") as f:
        f.write(f'Epoch {epoch+1} Accuracy: ({correct}/{total}) {accuracy:.2f}%\n')
    return avg_loss

if __name__ == '__main__':
    start = time.time()
    with open("test.txt", "a") as f:
        f.write('Start write!!!\n')

    acc_list = []
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(EPOCH_NUM):
        if epoch == UNFREEZE_EPOCH:
            print(f'🔓 Epoch {epoch+1}: Unfreezing backbone...')
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_loss = train(epoch)
        val_loss = test(epoch, acc_list)

        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)

        scheduler.step(val_loss)

    torch.save(model.state_dict(), 'Model.pth')

    # 可视化
    plt.figure(figsize=(12,5))

    # 1. Loss（按 epoch）
    plt.subplot(1,2,1)
    plt.title("Train vs Val Loss")
    plt.plot(range(1, EPOCH_NUM+1), train_loss_per_epoch, label="Train Loss", color='orange')
    plt.plot(range(1, EPOCH_NUM+1), val_loss_per_epoch, label="Val Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy（按 epoch）
    plt.subplot(1,2,2)
    plt.title("Test Accuracy")
    plt.plot(range(1, EPOCH_NUM+1), acc_list, marker='o', color='green', label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
