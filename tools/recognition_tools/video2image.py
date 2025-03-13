import cv2
import os
import numpy as np
def extract_frames(video_path, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 每秒提取一帧
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    extracted_count = 0
    crop_region = (1148, 208, 1172, 232)
    mini_map_region = (1060, 120, 1260, 320)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        merged_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(-2, 3):
            for j in range(-2, 3):
                # 计算实际裁剪坐标
                y_start = crop_region[1] + 38 * i
                y_end = crop_region[3] + 38 * i
                x_start = crop_region[0] + 38 * j
                x_end = crop_region[2] + 38 * j

                # 边界检查（防止越界）
                if (y_start >= 0 and y_end <= height and
                    x_start >= 0 and x_end <= width):
                    # 裁剪原图区域
                    cropped = frame[y_start:y_end, x_start:x_end]
                    # 将裁剪内容贴到黑底画布对应位置
                    merged_frame[y_start:y_end, x_start:x_end] = cropped
        frame_cropped = merged_frame[mini_map_region[1]:mini_map_region[3], mini_map_region[0]:mini_map_region[2]]
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame_cropped)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

video_path = '办公室.mp4'
output_dir = 'frames/office'
extract_frames(video_path, output_dir)