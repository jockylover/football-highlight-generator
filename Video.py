import argparse

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
    parser = argparse.ArgumentParser(description="按起止时间(秒)截取视频片段")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("start", type=float, help="起始时间(秒)")
    parser.add_argument("end", type=float, help="结束时间(秒)")
    args = parser.parse_args()
    clip_video(args.input, args.output, args.start, args.end)
