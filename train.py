import torch
from torch import nn
import torchvision.models as models
from dataload import train_loader, test_loader
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import numpy as np

LR_BASE = 0.0001  # 主干网络基础学习率（小）
LR_FC = 0.001  # FC层学习率（大）
EPOCH_NUM = 30
UNFREEZE_EPOCH = 5  # 第10轮解冻主干网络
LABEL_SMOOTHING = 0.05


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
            print(f'[{time_since(start)}] Epoch {epoch + 1}, Step {batch_idx + 1}, Loss: {running_loss / count:.3f}')
    return running_loss / count


def test(epoch, acc_list, precision_list, recall_list, f1_list):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (prediction == target).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(prediction.cpu().numpy())

    # 计算各项指标
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)

    # 计算precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='binary', zero_division=0)

    acc_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f'Accuracy on test set: ({correct}/{total}) {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

    with open("test.txt", "a") as f:
        f.write(f'Epoch {epoch + 1} Accuracy: ({correct}/{total}) {accuracy:.2f}%\n')
        f.write(f'Epoch {epoch + 1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n')

    # 只在最后一个epoch绘制混淆矩阵
    if epoch == EPOCH_NUM - 1:
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'],
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

    return avg_loss


if __name__ == '__main__':
    start = time.time()
    with open("test.txt", "a") as f:
        f.write('Start write!!!\n')

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(EPOCH_NUM):
        if epoch == UNFREEZE_EPOCH:
            print(f'🔓 Epoch {epoch + 1}: Unfreezing backbone...')
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_loss = train(epoch)
        val_loss = test(epoch, acc_list, precision_list, recall_list, f1_list)

        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)

        scheduler.step(val_loss)

    torch.save(model.state_dict(), 'Model.pth')

    # 可视化
    plt.figure(figsize=(18, 10))

    # 1. Loss（按 epoch）
    plt.subplot(2, 3, 1)
    plt.title("Train vs Val Loss")
    plt.plot(range(1, EPOCH_NUM + 1), train_loss_per_epoch, label="Train Loss", color='orange')
    plt.plot(range(1, EPOCH_NUM + 1), val_loss_per_epoch, label="Val Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy（按 epoch）
    plt.subplot(2, 3, 2)
    plt.title("Test Accuracy")
    plt.plot(range(1, EPOCH_NUM + 1), acc_list, marker='o', color='green', label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    # 3. Precision（按 epoch）
    plt.subplot(2, 3, 3)
    plt.title("Precision")
    plt.plot(range(1, EPOCH_NUM + 1), precision_list, marker='o', color='red', label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()

    # 4. Recall（按 epoch）
    plt.subplot(2, 3, 4)
    plt.title("Recall")
    plt.plot(range(1, EPOCH_NUM + 1), recall_list, marker='o', color='purple', label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend()

    # 5. F1-Score（按 epoch）
    plt.subplot(2, 3, 5)
    plt.title("F1-Score")
    plt.plot(range(1, EPOCH_NUM + 1), f1_list, marker='o', color='brown', label="F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()