import os
import json
import torch
import cv2
import subprocess
from model import FineTunedCLIP
import clip
from PIL import Image
import ffmpeg
from config import DEFAULT_MODEL_PATH


class HighlightGenerator:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FineTunedCLIP().to(self.device)
        # map_location 兼容 CPU/GPU；weights_only 规避 PyTorch 2.6 默认值变更告警（权重是 state_dict）
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.preprocess = clip.load("ViT-B/32", device=self.device)[1]
        self.batch_size = batch_size
        # 推理阈值：优先读取与模型同目录的 best_threshold.json，否则回退到 0.5（与训练评估一致）
        self.threshold = self._load_threshold(model_path, default=0.5)

    @staticmethod
    def _load_threshold(model_path, default=0.5):
        thr_path = os.path.join(os.path.dirname(model_path), "best_threshold.json")
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                return float(json.load(f)["threshold"])
        except (OSError, KeyError, ValueError, TypeError):
            return default

    def detect_shots(self, video_path, threshold=None):
        """按秒采样帧并分批推理，返回被判为射门的秒位列表。"""
        if threshold is None:
            threshold = self.threshold

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0:
            cap.release()
            return []

        shot_times = []
        batch_tensors = []
        batch_secs = []

        def flush():
            if not batch_tensors:
                return
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch).squeeze(-1)
                probs = torch.sigmoid(logits)
            for sec, p in zip(batch_secs, probs.tolist()):
                if p > threshold:
                    shot_times.append(sec)
            batch_tensors.clear()
            batch_secs.clear()

        for sec in range(0, int(total_frames / fps), 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理（攒批，统一上 GPU 做一次前向）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_tensors.append(self.preprocess(Image.fromarray(frame)))
            batch_secs.append(sec)

            if len(batch_tensors) >= self.batch_size:
                flush()

        flush()
        cap.release()
        return sorted(shot_times)

    def merge_shot_times(self, shot_times, clip_duration):
        if not shot_times:
            return []
        merged_times = []
        start = shot_times[0]
        end = shot_times[0]
        for t in shot_times[1:]:
            if t - end < clip_duration:
                end = t
            else:
                merged_times.append((start, end))
                start = t
                end = t
        merged_times.append((start, end))
        return merged_times

    # def generate_highlight(self, video_path, shot_times, output_path, clip_duration=5.0):
    #     if not shot_times:
    #         print("No highlights detected, skipping highlight generation.")
    #         return
    #
    #     merged_times = self.merge_shot_times(shot_times, clip_duration)
    #     inputs = []
    #     # for start, end in merged_times:
    #     #     # 计算实际截取的起始时间和时长
    #     #     ss = max(0, start - clip_duration / 2)
    #     #     t = end - start + clip_duration
    #     #     inputs.append(
    #     #         ffmpeg.input(video_path, ss=ss, t=t)
    #     #     )
    #     # ffmpeg.concat(*[i.video for i in inputs], v=1, a=1).output(output_path).run()
    #     concat = ffmpeg.concat(*[i.video for i in inputs], *[i.audio for i in inputs], v=1, a=1)
    #     concat.output(output_path).run()
    def generate_highlight(self, video_path, shot_times, output_path, clip_duration=5.0):
        if not shot_times:
            print("No highlights detected, skipping highlight generation.")
            return

        merged_times = self.merge_shot_times(shot_times, clip_duration)
        inputs = []
        for start, end in merged_times:
            # 计算实际截取的起始时间和时长
            ss = max(0, start - clip_duration / 2)
            t = end - start + clip_duration
            inputs.append(
                ffmpeg.input(video_path, ss=ss, t=t)
            )

        video_streams = [i.video for i in inputs]
        audio_streams = [i.audio for i in inputs]

        # 拼接视频流
        video_concat = ffmpeg.concat(*video_streams, v=1, a=0).node
        # 拼接音频流
        audio_concat = ffmpeg.concat(*audio_streams, v=0, a=1).node

        # 合并视频和音频流，使用 [0] 获取流
        output = ffmpeg.output(video_concat[0], audio_concat[0], output_path)
        output.run()


if __name__ == "__main__":
    generator = HighlightGenerator()
    sample_clip = r"E:\System Default\table\学习\大四下\paper\data\clips\train\shot\2014-11-04 - 22-45 Arsenal 3 - 3 Anderlecht\1_720p\shot_0.mp4"
    shots = generator.detect_shots(sample_clip)
    generator.generate_highlight(sample_clip, shots, "highlight.mp4")