import ffmpeg


def clip_video(input_file, output_file, start_time, end_time):
    try:
        (
            ffmpeg
            .input(input_file, ss=start_time, to=end_time)
            .output(output_file)
            .run()
        )
        print("视频截取成功！")
    except ffmpeg.Error as e:
        print(f"截取视频时出错: {e.stderr.decode()}")


if __name__ == "__main__":
    input_video = r"E:\System Default\table\1_720p.mp4"
    output_video = r"E:\System Default\table\11_720p.mp4"
    start = 44 * 60
    end = 45 * 60
    clip_video(input_video, output_video, start, end)
