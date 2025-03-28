import psutil
import subprocess
import time
import os

# ---------------------- 配置部分 ----------------------
SOFTWARE_PATH = "E:\\Program Files\\Netease\\MuMu Player 12\\shell\\MuMuPlayer.exe"
PROCESS_NAME = "MuMuPlayer.exe"

CHECK_INTERVAL = 20  # 检查间隔时间（秒）



# ----------------------------------------------------

def check_condition():
    """
    自定义条件检查函数
    返回 True 表示需要重启软件，否则返回 False
    """

    return True


def find_software_process():
    """查找与目标进程名匹配的所有进程"""
    return [proc for proc in psutil.process_iter(['name']) if proc.info['name'] == PROCESS_NAME]


def restart_software():
    """重启软件"""
    # 终止所有同名进程
    for proc in find_software_process():
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    # 启动新进程
    subprocess.Popen(SOFTWARE_PATH)
    print(f"已重启 {PROCESS_NAME}")


def main_loop():
    """主监控循环"""
    while True:
        try:
            # 查找正在运行的软件进程
            processes = find_software_process()

            if processes:
                # 如果找到运行中的进程
                if check_condition():
                    print("触发重启条件")
                    restart_software()
            else:
                # 如果进程未运行，直接启动
                print("进程未运行，正在启动...")
                restart_software()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"发生错误：{str(e)}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    print(f"开始监控进程 {PROCESS_NAME}...")
    main_loop()