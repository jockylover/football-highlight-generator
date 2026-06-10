import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss
import clip
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from model import FineTunedCLIP
from data_processing.video_utils import SoccerNetDataset
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def train():
    # 初始化美观样式
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.right': False,
        'axes.spines.top': False
    })

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载 CLIP 预处理函数
    _, clip_preprocess = clip.load("ViT-B/32", device=device)

    # 数据加载
    train_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\train\shot'
    train_non_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\train\non_shot'
    valid_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\valid\shot'
    valid_non_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\valid\non_shot'

    train_dataset = SoccerNetDataset(
        shot_dir=train_shot_dir,
        non_shot_dir=train_non_shot_dir,
        transform=clip_preprocess
    )
    valid_dataset = SoccerNetDataset(
        shot_dir=valid_shot_dir,
        non_shot_dir=valid_non_shot_dir,
        transform=clip_preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 训练准备
    model = FineTunedCLIP(freeze_clip=False, unfreeze_layers=2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))  # 处理类别不平衡
    num_epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 记录训练指标
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 训练循环
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            # 实时打印进度
            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {correct / total:.4f}")

        # 记录训练指标
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            all_preds = []
            all_labels = []

            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)

                # 计算准确率
                predictions = (outputs > 0.5).float()
                valid_correct += (predictions == labels).sum().item()
                valid_total += labels.size(0)

                # 保存预测结果用于ROC曲线
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_valid_loss += loss.item()

            # 记录验证指标
            epoch_valid_loss = total_valid_loss / len(valid_loader)
            epoch_valid_acc = valid_correct / valid_total
            valid_losses.append(epoch_valid_loss)
            valid_accuracies.append(epoch_valid_acc)

        # 更新学习率调度器
        scheduler.step()

        # 保存最佳模型
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), f"best_model_{timestamp}.pth")
            print("Best model saved!")

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Valid Loss: {epoch_valid_loss:.4f} | Valid Acc: {epoch_valid_acc:.4f}\n")

    # 可视化训练过程
    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(valid_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--', marker='s')
    plt.title('Training and Validation Loss', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='#2ca02c', linewidth=2, marker='o')
    plt.plot(valid_accuracies, label='Validation Accuracy', color='#d62728', linewidth=2, linestyle='--', marker='s')
    plt.title('Training and Validation Accuracy', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_curves_{timestamp}.png", bbox_inches='tight', transparent=True)
    plt.close()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', pad=20)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f"roc_curve_{timestamp}.png", bbox_inches='tight', transparent=True)
    plt.close()

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, np.array(all_preds) > 0.5)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Shot', 'Shot'], yticklabels=['Non-Shot', 'Shot'])
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"confusion_matrix_{timestamp}.png", bbox_inches='tight', transparent=True)
    plt.close()

    print("Training completed and visualizations saved.")


if __name__ == "__main__":
    train()