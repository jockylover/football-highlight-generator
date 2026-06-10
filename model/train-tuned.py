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
from model.model_tuned import ImprovedCLIPShotDetector, ZeroShotCLIPShotDetector
from data_processing.video_utils import SoccerNetDataset
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def train_improved_model():
    """训练改进的多模态模型"""
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

    # 数据加载（与原始代码相同的路径）
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
        batch_size=16,  # 减小batch size因为模型更复杂
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 训练准备
    model = ImprovedCLIPShotDetector(device=device, freeze_vision=False, freeze_text=True).to(device)

    # 使用不同的学习率对不同部分
    vision_params = []
    fusion_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip_model.visual' in name:
                vision_params.append(param)
            else:
                fusion_params.append(param)

    optimizer = AdamW([
        {'params': vision_params, 'lr': 1e-5},  # 视觉编码器用较小学习率
        {'params': fusion_params, 'lr': 1e-4}  # 融合层用较大学习率
    ], weight_decay=0.01)

    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    num_epochs = 15  # 减少epoch数，模型收敛可能更快
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 记录训练指标
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print("Starting training with improved multimodal CLIP model...")

    # 训练循环
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 计算准确率
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            # 实时打印进度
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {correct / total:.4f}")

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
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)

                # 计算准确率
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                valid_correct += (predictions == labels).sum().item()
                valid_total += labels.size(0)

                # 保存预测结果用于ROC曲线
                all_preds.extend(probs.cpu().numpy())
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
            torch.save(model.state_dict(), f"improved_clip_model_{timestamp}.pth")
            print("Best model saved!")

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Valid Loss: {epoch_valid_loss:.4f} | Valid Acc: {epoch_valid_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}\n")

    # 生成详细的分类报告
    all_preds_binary = np.array(all_preds) > 0.5
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds_binary,
                                target_names=['Non-Shot', 'Shot'], digits=4))

    # 可视化训练过程
    create_training_plots(train_losses, valid_losses, train_accuracies,
                          valid_accuracies, all_labels, all_preds, timestamp)

    print("Training completed and visualizations saved.")
    return model, timestamp


def evaluate_zero_shot_model():
    """评估零样本模型性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating Zero-Shot CLIP model on device: {device}")

    _, clip_preprocess = clip.load("ViT-B/32", device=device)

    # 加载验证数据
    valid_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\valid\shot'
    valid_non_shot_dir = r'E:\System Default\table\学习\大四下\paper\data\clips\valid\non_shot'

    valid_dataset = SoccerNetDataset(
        shot_dir=valid_shot_dir,
        non_shot_dir=valid_non_shot_dir,
        transform=clip_preprocess
    )

    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 创建零样本模型
    zero_shot_model = ZeroShotCLIPShotDetector(device=device)
    zero_shot_model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device).float()

            # 获取预测概率
            probs = zero_shot_model(images).squeeze(-1)
            predictions = (probs > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Zero-Shot Model Accuracy: {accuracy:.4f}")

    # 计算AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    print(f"Zero-Shot Model AUC: {roc_auc:.4f}")

    # 生成分类报告
    all_preds_binary = np.array(all_preds) > 0.5
    print("\nZero-Shot Classification Report:")
    print(classification_report(all_labels, all_preds_binary,
                                target_names=['Non-Shot', 'Shot'], digits=4))

    return all_labels, all_preds


def create_training_plots(train_losses, valid_losses, train_accuracies,
                          valid_accuracies, all_labels, all_preds, timestamp):
    """创建训练过程的可视化图表"""

    # 设置整体图形样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 创建包含4个子图的图形
    fig = plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(valid_losses, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--', marker='s')
    plt.title('Training and Validation Loss', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='#2ca02c', linewidth=2, marker='o')
    plt.plot(valid_accuracies, label='Validation Accuracy', color='#d62728', linewidth=2, linestyle='--', marker='s')
    plt.title('Training and Validation Accuracy', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ROC曲线
    plt.subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 混淆矩阵
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(all_labels, np.array(all_preds) > 0.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Shot', 'Shot'], yticklabels=['Non-Shot', 'Shot'],
                square=True, linewidths=0.5)
    plt.title('Confusion Matrix', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout(pad=3.0)
    plt.savefig(f"improved_training_results_{timestamp}.png", bbox_inches='tight',
                transparent=True, dpi=300)
    plt.show()


def compare_models():
    """比较原始模型、改进模型和零样本模型的性能"""
    print("=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    # 评估零样本模型
    zero_shot_labels, zero_shot_preds = evaluate_zero_shot_model()

    # 这里你可以加载原始模型进行比较
    # original_labels, original_preds = evaluate_original_model()

    print("\nComparison Summary:")
    print("1. Zero-shot CLIP model: Uses text-image similarity without training")
    print("2. Improved multimodal model: Fine-tunes vision-language alignment")
    print("3. Your original model: Only uses visual features (not multimodal)")


if __name__ == "__main__":
    # 训练改进的多模态模型
    model, timestamp = train_improved_model()

    # 评估零样本模型
    compare_models()

    print("\nTraining completed! Key improvements:")
    print("✅ Uses CLIP's multimodal capabilities")
    print("✅ Compares images with relevant text descriptions")
    print("✅ Includes zero-shot evaluation")
    print("✅ Better feature fusion strategy")