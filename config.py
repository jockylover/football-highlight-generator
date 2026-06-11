"""集中配置：数据/模型路径与默认超参，支持环境变量覆盖。

换机器时只需设置环境变量（如 PAPER_ROOT，或单独设 DATA_ROOT / MODEL_DIR /
MODEL_PATH / SOCCERNET_DIR），无需改动任何源码。
"""
import os
from pathlib import Path

# 数据与模型的根目录
PAPER_ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"E:\System Default\table\学习\大四下\paper",
))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PAPER_ROOT / "data")))
CLIPS_ROOT = DATA_ROOT / "clips"
SOCCERNET_DIR = Path(os.environ.get("SOCCERNET_DIR", str(DATA_ROOT / "SoccerNet")))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PAPER_ROOT / "model")))
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(MODEL_DIR / "best_model_20250406-201708.pth"))

# CLIP backbone
CLIP_BACKBONE = "ViT-B/32"

# 推理默认判定阈值（可被模型同目录的 best_threshold.json 覆盖）
DEFAULT_THRESHOLD = 0.5


def clip_dirs(split):
    """返回某个 split 的 (正样本目录, 负样本目录)。split ∈ {train, valid, test}。"""
    base = CLIPS_ROOT / split
    return str(base / "shot"), str(base / "non_shot")
