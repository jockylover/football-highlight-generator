import os
import json
import cv2
import subprocess
import random
from glob import glob


# ------------------- 工具函数 -------------------
def parse_game_time(game_time_str):
    half, time_str = game_time_str.split(' - ')
    minutes, seconds = map(int, time_str.split(':'))
    return int(half), minutes * 60 + seconds


def load_labels(label_path):
    with open(label_path, 'r') as f:
        data = json.load(f)
    shot_times = {1: [], 2: []}
    for ann in data['annotations']:
        if ann['label'].lower() in ['shots on target', 'goal']:
            half, time_sec = parse_game_time(ann['gameTime'])
            shot_times[half].append(time_sec)
    return shot_times


# ------------------- 视频处理 -------------------
def extract_shot_clips_ffmpeg(video_path, shot_times, output_dir, clip_duration=5.0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    for idx, time_sec in enumerate(shot_times):
        if time_sec > total_duration:
            print(f"跳过无效时间点：{time_sec}秒（视频总时长 {total_duration:.1f}秒）")
            continue

        start_time = max(0, time_sec - clip_duration / 2)
        output_path = os.path.join(output_dir, f"shot_{idx}.mp4")
        subprocess.run([
            "ffmpeg", "-ss", str(start_time), "-i", video_path,
            "-t", str(clip_duration), "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-vf", "scale=iw:-2", "-y", output_path
        ], check=True)


def generate_negative_clips(video_path, shot_times, output_dir, num_negatives=10, clip_duration=5.0):
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    safe_times = [t for t in range(0, int(total_duration - clip_duration))
                  if all(abs(t - st) >= 10 for st in shot_times)]
    selected_times = random.sample(safe_times, min(num_negatives, len(safe_times)))

    for idx, t in enumerate(selected_times):
        output_path = os.path.join(output_dir, f"neg_{idx}.mp4")
        subprocess.run([
            "ffmpeg", "-ss", str(t), "-i", video_path,
            "-t", str(clip_duration), "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-y", output_path
        ], check=True)


# ------------------- 主流程 -------------------
def process_all_matches(root_dir, output_base_dir):
    for subdir, _, _ in os.walk(root_dir):
        match_dirs = [d for d in glob(os.path.join(subdir, "* - * * * *"))]
        for match_dir in match_dirs:
            label_path = os.path.join(match_dir, "Labels-v2.json")
            if not os.path.exists(label_path):
                continue

            half_shot_times = load_labels(label_path)
            match_name = os.path.basename(match_dir)

            for half_video in glob(os.path.join(match_dir, "[1-2]_*.mkv")):
                base_name = os.path.basename(half_video)
                half = int(base_name.split('_')[0])
                half_name = os.path.splitext(base_name)[0]

                shot_times = half_shot_times.get(half, [])

                # 正样本
                output_shot = os.path.join(output_base_dir, "shot", match_name, half_name)
                os.makedirs(output_shot, exist_ok=True)
                extract_shot_clips_ffmpeg(half_video, shot_times, output_shot)

                # 负样本
                output_neg = os.path.join(output_base_dir, "non_shot", match_name, half_name)
                os.makedirs(output_neg, exist_ok=True)
                generate_negative_clips(half_video, shot_times, output_neg)


# 测试运行
process_all_matches(
    root_dir="E:/System Default/table/paper/data/SoccerNet/train",
    output_base_dir="../data/clips/train"
)