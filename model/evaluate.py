"""在 test split 上评估已训练模型，报告 precision/recall/F1/AP 与 PR 曲线。

用法（从仓库根目录）：
    python model/evaluate.py
依赖 config.py 中的 DEFAULT_MODEL_PATH 与 clip_dirs("test")。
"""
import os
import json
import datetime

import torch
from torch.utils.data import DataLoader
import clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, average_precision_score, precision_recall_curve
)

from model import FineTunedCLIP
from data_processing.video_utils import SoccerNetDataset
from config import clip_dirs, DEFAULT_MODEL_PATH, CLIP_BACKBONE, DEFAULT_THRESHOLD


def load_threshold(model_path, default=DEFAULT_THRESHOLD):
    thr_path = os.path.join(os.path.dirname(model_path), "best_threshold.json")
    try:
        with open(thr_path, "r", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except (OSError, KeyError, ValueError, TypeError):
        return default


def evaluate(model_path=DEFAULT_MODEL_PATH, split="test", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, clip_preprocess = clip.load(CLIP_BACKBONE, device=device)
    shot_dir, non_shot_dir = clip_dirs(split)
    dataset = SoccerNetDataset(shot_dir=shot_dir, non_shot_dir=non_shot_dir,
                               transform=clip_preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FineTunedCLIP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(-1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    threshold = load_threshold(model_path)
    preds = all_probs > threshold

    print(f"\n判定阈值: {threshold:.4f}")
    print(classification_report(all_labels, preds,
                                target_names=['Non-Shot', 'Shot'], digits=4))
    ap = average_precision_score(all_labels, all_probs)
    print(f"Average Precision (AP): {ap:.4f}")

    # PR 曲线
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall ({split})')
    plt.legend(loc='lower left')
    plt.savefig(f"pr_curve_{split}_{timestamp}.png", bbox_inches='tight')
    plt.close()
    print(f"PR 曲线已保存: pr_curve_{split}_{timestamp}.png")


if __name__ == "__main__":
    evaluate()
