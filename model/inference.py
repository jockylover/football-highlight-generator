import torch
import cv2
import subprocess
from model import FineTunedCLIP
import clip
from PIL import Image
import ffmpeg


class HighlightGenerator:
    def __init__(self, model_path=r"E:\System Default\table\学习\大四下\paper\model\best_model_20250406-201708.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FineTunedCLIP().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.preprocess = clip.load("ViT-B/32", device=self.device)[1]

    def detect_shots(self, video_path, threshold=0.85, window_size=2.0):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shot_times = []

        for sec in range(0, int(total_frames / fps), 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                output = self.model(image)
            prob = torch.sigmoid(output).item()

            if prob > threshold:
                shot_times.append(sec)

        cap.release()
        return shot_times

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