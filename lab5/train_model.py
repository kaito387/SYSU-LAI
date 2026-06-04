"""
中药图片分类 - 基于PyTorch的CNN实现
数据集：5类中药（枸杞、金银花、槐花、党参、百合）
改进策略：轻量化网络 + 强数据增强 + 余弦退火调度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ========== 数据预处理与增强 ==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载训练集（带数据增强）
train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)

# 划分训练集和验证集 (80% / 20%)
n_total = len(train_dataset)
indices = list(range(n_total))
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * n_total)
train_indices = indices[:split]
val_indices = indices[split:]

train_subset = Subset(train_dataset, train_indices)
val_full_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=eval_transform)
val_subset = Subset(val_full_dataset, val_indices)

# 测试集
test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)

# DataLoader
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"类别: {class_names}")
print(f"训练集: {len(train_subset)}, 验证集: {len(val_subset)}, 测试集: {len(test_dataset)}")


# ========== CNN 网络结构 ==========
class ConvBlock(nn.Module):
    """卷积块：Conv -> BN -> ReLU (x2) -> MaxPool -> Dropout"""

    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class TCMClassifier(nn.Module):
    """中药分类CNN模型（轻量化设计）"""

    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),     # 224 -> 112
            ConvBlock(32, 64, dropout=0.1),     # 112 -> 56
            ConvBlock(64, 128, dropout=0.15),    # 56 -> 28
            ConvBlock(128, 128, dropout=0.2),   # 28 -> 14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ========== 训练函数 ==========
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


# ========== 主训练流程 ==========
def main():
    model = TCMClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{EPOCHS} | LR: {lr_now:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(BASE_DIR / "best_model.pth"))
            print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.4f})")

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")

    # ========== 测试集评估 ==========
    model.load_state_dict(torch.load(str(BASE_DIR / "best_model.pth"), weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"测试集准确率: {test_acc:.4f} | 测试集Loss: {test_loss:.4f}")

    # ========== 训练集最终准确率 ==========
    train_loss_final, train_acc_final = evaluate(model, train_loader, criterion, DEVICE)
    print(f"训练集最终准确率: {train_acc_final:.4f}")

    # ========== 绘制曲线 ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, EPOCHS + 1), history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    axes[0].plot(range(1, EPOCHS + 1), history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, EPOCHS + 1), history['train_acc'], 'b-', label='Train Accuracy', linewidth=1.5)
    axes[1].plot(range(1, EPOCHS + 1), history['val_acc'], 'r-', label='Val Accuracy', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"曲线图已保存至: {BASE_DIR / 'training_curves.png'}")

    # ========== 结果汇总 ==========
    print("\n" + "=" * 50)
    print("最终结果汇总")
    print("=" * 50)
    print(f"训练集准确率: {train_acc_final:.4f} ({train_acc_final*100:.2f}%)")
    print(f"验证集最佳准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
