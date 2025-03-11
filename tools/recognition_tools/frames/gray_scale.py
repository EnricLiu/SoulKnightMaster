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
def calculate_grayscale_sum(image_path):
    image = Image.open(image_path)
    grayscale_sums = {}

    for name, coords in position_dict.items():
        region = image.crop(coords)
        gray_region = region.convert('L')
        pixel_sum = np.sum(np.array(gray_region))
        grayscale_sums[name] = pixel_sum
    return grayscale_sums

# 新增文件夹处理函数：
def process_folder(folder_path):
    folder = Path(folder_path)
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.png']]

    for image_path in image_files:
        print(f"{'='*40}\n处理文件：{image_path.name}\n{'='*40}")
        results = calculate_grayscale_sum(image_path)

        for name, total in results.items():
            image = Image.open(image_path)
            if name == "self":
                # 读取并验证
                img1 = cv.imread('./elon_mask.png', 0)  # 0表示灰度模式
                coords = position_dict["self"]
                region = image.crop(coords)
                # 1. 将PIL Image转为numpy数组
                region_np = np.array(region)
                R_channel = region_np[:, :, 0]
                # print(img1.shape, img2_gray.shape)
                assert img1.shape == R_channel.shape, "尺寸不一致"
                # cv.imshow("img1", img1)
                print("img1", img1)
                # cv.imshow("R_channel", R_channel)

                # 计算并输出结果
                total = np.sum((img1/255 )* R_channel)
                # cv.imshow("result", img1 * R_channel)
                print("总和:", total)
            print(f"{name:<15} 灰度总和：{total}")
        region_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        cv.imshow("region", region_cv)
        cv.waitKey(0)


# 主程序入口示例：
if __name__ == "__main__":
    target_folder = Path("../frames")
    process_folder(target_folder)





