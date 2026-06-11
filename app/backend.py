import sys
import os
import logging
import subprocess
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, render_template, make_response
from model.inference import HighlightGenerator
import uuid
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
LOG_FOLDER = "logs"
CORS(app)

# 创建必要的文件夹
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 存储处理状态的字典
processing_status = {}
cleanup_times = {}

# 全局模型单例：避免每次上传都重新加载 CLIP + 权重（数百 MB / 数秒）
_generator = None
_generator_lock = threading.Lock()
_inference_lock = threading.Lock()  # 串行化 GPU 推理，避免并发 OOM


def get_generator():
    """惰性加载并复用同一个 HighlightGenerator 实例。"""
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                logger.info("Loading HighlightGenerator (one-time)...")
                _generator = HighlightGenerator()
                logger.info("HighlightGenerator ready.")
    return _generator

# 配置
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
CLEANUP_AFTER_HOURS = 24  # 24小时后清理文件


def allowed_file(filename):
    """检查文件扩展名是否被允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(file_path):
    """获取文件信息"""
    try:
        if not os.path.exists(file_path):
            return None

        stat = os.stat(file_path)
        duration = get_video_duration(file_path)

        return {
            'size': stat.st_size,
            'duration': duration,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None


def cleanup_old_files():
    """清理旧文件"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=CLEANUP_AFTER_HOURS)

        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for file_path in Path(folder).glob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")

        # 清理处理状态
        to_remove = []
        for video_id, status_time in cleanup_times.items():
            if status_time < cutoff_time:
                to_remove.append(video_id)

        for video_id in to_remove:
            processing_status.pop(video_id, None)
            cleanup_times.pop(video_id, None)

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def schedule_cleanup():
    """定期清理任务"""
    cleanup_old_files()
    # 每小时清理一次
    threading.Timer(3600, schedule_cleanup).start()


# 启动清理任务
schedule_cleanup()


@app.route("/")
def home():
    """主页"""
    return render_template("index.html")


@app.route("/health")
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })


@app.route("/upload", methods=["POST"])
def upload_video():
    """上传视频文件"""
    try:
        # 检查请求中是否包含文件
        if "video" not in request.files:
            logger.warning("No video file in request")
            return jsonify({"error": "没有上传视频文件"}), 400

        video = request.files["video"]

        # 检查文件名
        if video.filename == '' or not video.filename:
            logger.warning("Empty filename")
            return jsonify({"error": "文件名不能为空"}), 400

        # 检查文件扩展名
        if not allowed_file(video.filename):
            logger.warning(f"Invalid file extension: {video.filename}")
            return jsonify({"error": f"不支持的文件格式，请上传 {', '.join(ALLOWED_EXTENSIONS)} 格式"}), 400

        # 生成唯一ID
        video_id = str(uuid.uuid4())
        safe_filename = secure_filename(video.filename)
        video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{safe_filename}")

        # 检查文件大小（在保存前）
        video.seek(0, 2)  # 移到文件末尾
        file_size = video.tell()
        video.seek(0)  # 重置文件指针

        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({"error": f"文件大小超过限制 ({MAX_FILE_SIZE // (1024 * 1024)}MB)"}), 400

        # 保存文件
        logger.info(f"Saving uploaded file: {video_path}")
        video.save(video_path)

        # 验证文件是否有效
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            logger.error(f"Failed to save file or file is empty: {video_path}")
            return jsonify({"error": "文件保存失败"}), 500

        # 初始化处理状态
        processing_status[video_id] = {
            "status": "uploaded",
            "stage": "文件上传成功",
            "progress": 0,
            "start_time": datetime.now(),
            "original_filename": safe_filename,
            "file_size": file_size
        }
        cleanup_times[video_id] = datetime.now()

        # 启动异步处理
        threading.Thread(
            target=process_video_async,
            args=(video_id, video_path),
            daemon=True
        ).start()

        logger.info(f"Video upload successful, ID: {video_id}")
        return jsonify({
            "video_id": video_id,
            "message": "文件上传成功，正在处理中",
            "file_size": file_size
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "服务器内部错误"}), 500


def process_video_async(video_id, video_path):
    """异步处理视频"""
    try:
        logger.info(f"Starting async processing for video {video_id}")

        # 更新状态
        processing_status[video_id].update({
            "status": "processing",
            "stage": "正在初始化AI模型",
            "progress": 10
        })

        # 复用全局生成器（首次调用时加载一次）
        generator = get_generator()

        # 检测射门场景
        processing_status[video_id].update({
            "stage": "正在分析视频内容",
            "progress": 30
        })

        # 串行化 GPU 推理，避免多请求并发导致显存溢出
        with _inference_lock:
            shot_times = generator.detect_shots(video_path)
        logger.info(f"Detected {len(shot_times)} shot scenes for video {video_id}")

        # 生成集锦
        processing_status[video_id].update({
            "stage": "正在生成集锦视频",
            "progress": 70
        })

        highlight_path = os.path.join(OUTPUT_FOLDER, f"highlight_{video_id}.mp4")
        generator.generate_highlight(video_path, shot_times, highlight_path)

        # 获取输出文件信息
        file_info = get_file_info(highlight_path)

        if file_info and os.path.exists(highlight_path):
            processing_status[video_id].update({
                "status": "completed",
                "stage": "处理完成",
                "progress": 100,
                "output_file": highlight_path,
                "output_info": file_info,
                "shot_count": len(shot_times),
                "completion_time": datetime.now()
            })
            logger.info(f"Video processing completed successfully for {video_id}")
        else:
            raise Exception("Failed to generate highlight video")

    except Exception as e:
        logger.error(f"Processing error for video {video_id}: {e}")
        processing_status[video_id].update({
            "status": "error",
            "stage": f"处理失败: {str(e)}",
            "progress": 0,
            "error_time": datetime.now()
        })


@app.route("/status/<video_id>")
def check_status(video_id):
    """检查处理状态"""
    try:
        if video_id not in processing_status:
            logger.warning(f"Video ID not found: {video_id}")
            return jsonify({"error": "视频ID不存在"}), 404

        status_info = processing_status[video_id].copy()

        # 转换datetime对象为字符串
        for key in ['start_time', 'completion_time', 'error_time']:
            if key in status_info and isinstance(status_info[key], datetime):
                status_info[key] = status_info[key].isoformat()

        if status_info["status"] == "completed":
            highlight_path = status_info.get("output_file")
            if highlight_path and os.path.exists(highlight_path):
                return jsonify({
                    "status": "completed",
                    "download_url": f"/download/{video_id}",
                    "duration": status_info["output_info"]["duration"],
                    "file_size": status_info["output_info"]["size"],
                    "shot_count": status_info.get("shot_count", 0),
                    "processing_time": status_info.get("completion_time"),
                    "stage": status_info["stage"]
                })
            else:
                logger.error(f"Output file missing for completed video {video_id}")
                return jsonify({"error": "输出文件丢失"}), 500

        elif status_info["status"] == "error":
            return jsonify({
                "status": "error",
                "message": status_info["stage"],
                "error_time": status_info.get("error_time")
            }), 500

        else:
            return jsonify({
                "status": status_info["status"],
                "stage": status_info["stage"],
                "progress": status_info["progress"]
            })

    except Exception as e:
        logger.error(f"Status check error for video {video_id}: {e}")
        return jsonify({"error": "状态检查失败"}), 500


@app.route("/download/<video_id>")
def download_highlight(video_id):
    """下载生成的集锦视频"""
    try:
        if video_id not in processing_status:
            logger.warning(f"Download requested for unknown video ID: {video_id}")
            return jsonify({"error": "视频ID不存在"}), 404

        status_info = processing_status[video_id]

        if status_info["status"] != "completed":
            return jsonify({"error": "视频处理尚未完成"}), 400

        highlight_path = status_info.get("output_file")

        if not highlight_path or not os.path.exists(highlight_path):
            logger.error(f"Highlight file not found: {highlight_path}")
            return jsonify({"error": "集锦视频文件不存在"}), 404

        # 检查Flask版本并使用正确的参数
        try:
            # 尝试使用新版本的参数名
            response = make_response(send_file(
                highlight_path,
                as_attachment=True,
                download_name=f"football_highlights_{video_id}.mp4",
                mimetype='video/mp4'
            ))
        except TypeError:
            # 如果失败，使用旧版本的参数名
            response = make_response(send_file(
                highlight_path,
                as_attachment=True,
                attachment_filename=f"football_highlights_{video_id}.mp4",
                mimetype='video/mp4'
            ))

        response.headers['Content-Disposition'] = f'attachment; filename="football_highlights_{video_id}.mp4"'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'

        logger.info(f"Serving download for video {video_id}")
        return response

    except Exception as e:
        logger.error(f"Download error for video {video_id}: {e}")
        return jsonify({"error": "下载失败"}), 500


@app.route("/list")
def list_videos():
    """列出所有处理的视频"""
    try:
        video_list = []
        for video_id, info in processing_status.items():
            video_info = {
                "video_id": video_id,
                "status": info["status"],
                "original_filename": info.get("original_filename", "unknown"),
                "file_size": info.get("file_size", 0),
                "start_time": info["start_time"].isoformat() if isinstance(info["start_time"], datetime) else info[
                    "start_time"]
            }

            if info["status"] == "completed":
                video_info.update({
                    "shot_count": info.get("shot_count", 0),
                    "output_size": info.get("output_info", {}).get("size", 0),
                    "completion_time": info["completion_time"].isoformat() if isinstance(info["completion_time"],
                                                                                         datetime) else info.get(
                        "completion_time")
                })

            video_list.append(video_info)

        return jsonify({
            "videos": video_list,
            "total": len(video_list)
        })

    except Exception as e:
        logger.error(f"List videos error: {e}")
        return jsonify({"error": "获取视频列表失败"}), 500


@app.route("/stats")
def get_stats():
    """获取系统统计信息"""
    try:
        total_videos = len(processing_status)
        completed_videos = sum(1 for info in processing_status.values() if info["status"] == "completed")
        processing_videos = sum(
            1 for info in processing_status.values() if info["status"] in ["uploaded", "processing"])
        failed_videos = sum(1 for info in processing_status.values() if info["status"] == "error")

        # 计算存储使用情况
        upload_size = sum(os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
                          for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)))
        output_size = sum(os.path.getsize(os.path.join(OUTPUT_FOLDER, f))
                          for f in os.listdir(OUTPUT_FOLDER) if os.path.isfile(os.path.join(OUTPUT_FOLDER, f)))

        return jsonify({
            "total_videos": total_videos,
            "completed_videos": completed_videos,
            "processing_videos": processing_videos,
            "failed_videos": failed_videos,
            "storage": {
                "uploads_mb": round(upload_size / (1024 * 1024), 2),
                "outputs_mb": round(output_size / (1024 * 1024), 2),
                "total_mb": round((upload_size + output_size) / (1024 * 1024), 2)
            }
        })

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": "获取统计信息失败"}), 500


def get_video_duration(file_path):
    """获取视频时长"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)

        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            logger.warning(f"ffprobe failed for {file_path}: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"Error getting video duration for {file_path}: {e}")
        return 0


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "服务器内部错误"}), 500


if __name__ == "__main__":
    logger.info("Starting Football Highlight Generator Server...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )