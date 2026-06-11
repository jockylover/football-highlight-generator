import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class SoccerNetDataset(Dataset):
    def __init__(self, shot_dir, non_shot_dir, transform=None, cache_path=None):
        """
        shot_dir / non_shot_dir: 正/负样本片段目录
        transform: CLIP 预处理
        cache_path: 可选。校验过的样本清单缓存(json)。存在则直接加载，
                    避免每次启动都逐个 VideoCapture 打开 6000+ 片段。
        """
        self.transform = transform

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.samples = [tuple(item) for item in json.load(f)]
            print(f"从缓存加载样本清单: {cache_path} ({len(self.samples)} 条)")
            return

        valid_extensions = ('.mp4', '.mkv', '.avi')
        self.samples = []
        for directory, label in ((shot_dir, 1), (non_shot_dir, 0)):
            for root, _, files in os.walk(directory):
                for file in files:
                    if not file.lower().endswith(valid_extensions):
                        continue
                    path = os.path.join(root, file)
                    cap = cv2.VideoCapture(path)
                    ok = cap.isOpened() and cap.read()[0]  # 检查视频是否有有效帧
                    cap.release()
                    if ok:
                        self.samples.append((path, label))

        print(f"总计加载样本数: {len(self.samples)}")
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, ensure_ascii=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 直接 seek 到随机一帧，避免解码整段视频
        frame = None
        if total > 0:
            target = int(torch.randint(0, total, (1,)).item())
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                frame = None

        # 回退：随机 seek 失败(坏帧/无法定位)时退到第 0 帧
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"无法读取视频帧: {video_path}")
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
