import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class SoccerNetDataset(Dataset):
    def __init__(self, shot_dir, non_shot_dir, transform=None):
        self.samples = []
        valid_extensions = ('.mp4', '.mkv', '.avi')

        # 加载正样本
        for root, _, files in os.walk(shot_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    path = os.path.join(root, file)
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened() and cap.read()[0]:  # 检查视频是否有有效帧
                        self.samples.append((path, 1))
                    cap.release()

        # 加载负样本
        for root, _, files in os.walk(non_shot_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    path = os.path.join(root, file)
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened() and cap.read()[0]:  # 检查视频是否有有效帧
                        self.samples.append((path, 0))
                    cap.release()

        self.transform = transform
        print(f"总计加载样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # num_frames = len(frames)
        # if num_frames == 0:
        #     raise ValueError(f"视频中没有有效帧: {video_path}")
        #
        # # 随机选择三帧（如果帧数不足3，则补足重复帧）
        # if num_frames >= 3:
        #     indices = torch.randperm(num_frames)[:3]
        # else:
        #     indices = torch.randint(0, num_frames, (3,))
        #
        # selected_frames = [frames[i] for i in indices]
        # selected_frames = torch.stack(selected_frames)  # shape: (3, C, H, W)
        # return selected_frames, torch.tensor(label, dtype=torch.float32)

        # 随机选择一帧
        random_idx = torch.randint(0, len(frames), (1,))
        return frames[random_idx.item()], torch.tensor(label, dtype=torch.float32)