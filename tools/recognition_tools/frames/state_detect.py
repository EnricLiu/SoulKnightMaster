import cv2 as cv
from pathlib import Path
from PIL import Image
import numpy as np
import time

position_dict = {
    "blood": (51, 11, 230, 36),
    "Shield": (51, 45, 230, 70),
    "mana": (51, 78, 230, 103),
    "mini_map": (1060, 120, 1258, 321),
    "self": (1144, 204, 1176, 236)
}


def calculate_grayscale_sum(image):
    grayscale_sums = {}
    for name, coords in position_dict.items():
        region = image.crop(coords)
        gray_region = region.convert('L')
        pixel_sum = np.sum(np.array(gray_region))
        grayscale_sums[name] = pixel_sum
    return grayscale_sums


def process_folder(folder_path):
    folder = Path(folder_path)
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.png']]

    for image_path in image_files:
        print(f"{'=' * 40}\n处理文件：{image_path.name}\n{'=' * 40}")
        try:
            image = Image.open(image_path)
            results = calculate_grayscale_sum(image)

            # 处理self区域
            coords = position_dict["self"]
            region = image.crop(coords)
            region_np = np.array(region)
            R_channel = region_np[:, :, 0]  # 提取R通道
            img1 = cv.imread('./elon_mask.png', 0)  # 模板图像

            if img1.shape != R_channel.shape:
                print("警告：self区域与模板尺寸不匹配")
                continue

            self_total = np.sum((img1 / 255) * R_channel)
            print("self区域灰度总和：", self_total)
            results["self"] = self_total

            # 获取各区域值
            blood = results["blood"]
            shield = results["Shield"]
            mana = results["mana"]

            # 传送门状态判断
            if (blood == shield == mana) or (blood < 300000 or shield < 300000 or mana < 300000):
                print(f"blood: {blood}, shield: {shield}, mana: {mana}")
                print("当前状态：正在乘坐传送门")
                continue

            # 血量判断
            if blood > 410000:
                blood_status = "满血"
            elif blood > 380000:
                blood_status = "半血"
            elif blood > 360000:
                blood_status = "空血"
            else:
                blood_status = "异常血量"

            # 盾量判断
            shield_levels = [
                (570000, "6/6"),
                (530000, "5/6"),
                (490000, "4/6"),
                (460000, "3/6"),
                (430000, "2/6"),
                (390000, "1/6"),
                (360000, "0/6")
            ]
            shield_status = "0/6"
            for threshold, level in shield_levels:
                if shield > threshold:
                    shield_status = level
                    break

            # 蓝量判断
            if mana > 510000:
                mana_status = "满蓝"
            elif mana > 450000:
                mana_status = "半蓝"
            elif mana > 390000:
                mana_status = "空蓝"
            else:
                mana_status = "异常蓝量"

            # 状态判断
            state = "战斗状态" if results["self"] <= 37000 else "跑图状态"

            print(f"血量：{blood_status} | 盾量：{shield_status} | 蓝量：{mana_status}")
            print(f"当前状态：{state}")

            # 显示图像（调试用）
            region_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
            cv.imshow("region", region_cv)
            cv.waitKey(0)
            # time.sleep(0.1)

        except Exception as e:
            print(f"处理文件时出错：{str(e)}")
            continue


if __name__ == "__main__":
    target_folder = Path("../frames")  # 修改为实际路径
    process_folder(target_folder)