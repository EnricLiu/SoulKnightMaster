import cv2
import os
import time
# 打开视频文件
video_path = './元气骑士(3).mp4'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 设置截取间隔（以帧为单位）
frame_interval = 10  # 每隔30帧截取一张图

# 设置保存路径
save_path = './frames/skill_ready_bug'
os.makedirs(save_path, exist_ok=True)

# 设置截取区域（x, y, width, height）
region = (1120, 540, 120, 120)  # 示例区域

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # 截取特定区域
        x, y, w, h = region
        cropped_frame = frame[y:y+h, x:x+w]

        # 保存截取的图像
        save_filename = os.path.join(save_path, f'frame_{saved_count}_{time.time()}.jpg')
        cv2.imwrite(save_filename, cropped_frame)
        saved_count += 1
        print(f"Saved {save_filename}")

    frame_count += 1

# 释放视频捕获对象
cap.release()
print("Done.")