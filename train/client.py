import time
import json
from math import pi
import numpy as np
from PIL import Image
from PIL.Image import Image as _Image

import psutil
import subprocess

from httb import Client as httb
from utils import Action
from utils import Position

class Client:
    def __init__(self, node_name, ip="127.0.0.1", port="55555", timeout=1):
        self.ip = ip
        self.port = port
        self.url = f"http://{self.ip}:{self.port}"
        self.node_name = node_name
        
        self.client = httb(timeout)

    def sync_action(self, action: Action):
        url = f"{self.url}/action"
        payload = [{
            "node":     self.node_name,
            "action":   action.__dict__(),
        }]
        res = self.client.post(url, json=payload)
        
        if res.status_code() != 200:
            if res.json().get("type", None) is None or res.json().get("msg", None) is None:
                raise res.text()
        
        res = res.json()
        if isinstance(res, list):
            for node in res:
                if node["node"] == self.node_name:
                    return node
        return res
    
    def deschedule(self):
        url = f"{self.url}/deschedule?node={self.node_name}"
        res = self.client.get(url)
        if res.status_code() != 200:
            if res.json().get("type", None) is None or res.json().get("msg", None) is None:
                raise res.text()
        return res.json()
    
    def schedule(self):
        url = f"{self.url}/schedule?node={self.node_name}"
        res = self.client.get(url)
        if res.status_code() != 200:
            if res.json().get("type", None) is None or res.json().get("msg", None) is None:
                raise res.text()
        return res.json()
    
    def fetch_fb(self) -> np.ndarray:
        res = self.client.get(f"{self.url}/fb?node={self.node_name}")
        return np.frombuffer(res.content(), dtype=np.uint8).reshape(720, 1280, 4)[:,:,:3]

    def fetch_status(self):
        res = self.client.get(f"{self.url}/status?node={self.node_name}")
        if res.status_code() != 200:
            if res.json().get("type", None) is None or res.json().get("msg", None) is None:
                raise res.text()
        return res.json()
    
    def sync_click(self, pos: Position):
        url = f"{self.url}/action"
        payload = [{
            "node": self.node_name,
            "action": {
                "type": "Click",
                "pos": {
                    "x": pos.x,
                    "y": pos.y,
                }
            },
        }]
        res = self.client.post(url, json=payload)
        # print(res.response)
        if res.status_code() != 200:
            if res.json().get("type", None) is None or res.json().get("msg", None) is None:
                raise res.text()
    
        res = res.json()
        if isinstance(res, list):
            for node in res:
                if node["node"] == self.node_name:
                    return node
        return res


class AutoPilotClient(Client):
    def __init__(self, node_name, config):
        super().__init__(node_name, config["ip"], config["port"])

        self.task_flow = config["task_flow"]
        self.mumu_exec = config["mumu_exec"]
        self.ckpts = {}
        self.load_ckpts(config["ckpts"])

    def dhash(self, img: _Image|np.ndarray, resize=(32, 18)):
        if isinstance(img, np.ndarray):
            if resize is None:
                pixels = img.dot([0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                img = Image.fromarray(img)
                
        if isinstance(img, Image.Image):
            if resize is not None:
                img = img.resize(resize)
            pixels = np.array(img.convert('L')) 
        
        diff_h = pixels[:, 1:] > pixels[:, :-1]
        diff_v = pixels[1:, :] > pixels[:-1, :]
        hash_bits = np.concatenate([diff_h.flatten(), diff_v.flatten()]).astype(np.uint8)
        return hash_bits
    
    def judge_by_dhash(self, hash1: np.ndarray, hash2: np.ndarray, threshold: int):
        hamming = np.bitwise_xor(hash1, hash2).sum()
        return hamming <= threshold, hamming

    def load_ckpts(self, ckpt_config):
        for key, value in ckpt_config.items():
            image = Image.open(value["image"]).convert("L")
            threshold = value["threshold"]
            crop = value.get("crop")
            if crop is not None: 
                image = np.array(image)[crop[0]:crop[1], crop[2]:crop[3]]
                image = Image.fromarray(image)
            self.ckpts[key] = (threshold, self.dhash(image), crop)

    def _detect_ckpt(self, ckpt_name: str, timeout: int):
        lowest_distance = 114514
        thresh, target_dhash, crop = self.ckpts[ckpt_name]

        timeout = time.time() + timeout
        while time.time() < timeout:
            img = self.fetch_fb()
            if crop is not None: img = img[crop[0]:crop[1], crop[2]:crop[3]]
            # Image.fromarray(img).show()
            # exit()
            dhash = self.dhash(img)
            is_same, distance = self.judge_by_dhash(dhash, target_dhash, thresh)
            lowest_distance = min(lowest_distance, distance)
            if is_same: return True, distance
            time.sleep(0.25)
        else:
            return False, lowest_distance

    def _detect_ckpt_to_raw(self, ckpt_name: str, img: np.ndarray) -> bool:
        thresh, target_dhash, crop = self.ckpts[ckpt_name]
        if crop is not None: img = img[crop[0]:crop[1], crop[2]:crop[3]]
        dhash = self.dhash(img)
        return self.judge_by_dhash(dhash, target_dhash, thresh)

    def _take_action(self, action: dict):
        act_type = action["action"]

        is_good = False
        match act_type:
            case "detect":
                ckpt = action["ckpt"]
                timeout = action["timeout"]
                is_match, _distance = self._detect_ckpt(ckpt, timeout)
                if not is_match: 
                    print(f"[{ckpt}] Failed to detect, distance: {_distance}")
                is_good = is_match
            case "click":
                x = action["pos"]["x"]
                y = action["pos"]["y"]
                res = self.sync_click(Position(x, y))
                is_good = res.get("success", False)
            case "move":
                direction = action["direction"] * pi / 180
                duration = action["duration"]
                res = self.sync_action(Action(direction, False, False, False))
                if res.get("success", False):
                    time.sleep(duration)
                    is_good = True
            case "stop_move":
                res = self.sync_action(Action(None, False, False, False))
                is_good = res.get("success", False)
            case "wait":
                time.sleep(action["time"])
                is_good =True

        return act_type, is_good
    
    def _reboot(self):
        print("[DEBUG] adb kill server")
        subprocess.Popen(f"adb kill-server")
        for proc in [proc for proc in psutil.process_iter(['name']) if proc.info['name'] in ["MuMuPlayer.exe", "MuMuPlayerService.exe", "MuMuVMMHeadless.exe", "MuMuVMMSVC.exe"]]:
            proc.terminate()
            print("[DEBUG] terminated!")
        time.sleep(3)
        subprocess.Popen(f"{self.mumu_exec} -v 2")
        print("[DEBUG] re-opened!")
    
    def check_status(self):
        res = self.fetch_status()
        match res.get("status", "DEAD"):
            case "DEAD":
                res = self.deschedule()
                if not res.get("success", False): 
                    raise Exception(f"[EnvStep] Client[{self.node_name}] DEAD and not reboot😭!")
                print(f"[DEBUG] DEScheduled!")
                res = self._reboot()
                print(f"[DEBUG] WAITING for boot...")
                time.sleep(10)
                print(f"[DEBUG] START TASK!")
                res = self.schedule()
                if not res.get("success", False): 
                    raise Exception(f"[EnvStep] Client[{self.node_name}] DEAD and not reboot😭!")
                print(f"[DEBUG] Scheduled!")
                self.try_task("emu_reboot", timeout=120)
                res = self.fetch_status()
                if res.get("status", "DEAD") not in ["RUNNING", "IDLE"]: 
                    raise Exception(f"[EnvStep] Client[{self.node_name}] DEAD and not reboot😭!")
            case "DISCONNECTED" | "STOPPED": 
                raise Exception(f"[EnvStep] Client[{self.node_name}] not connected😡!")

    def try_task(self, task_name, timeout=45, max_retry=3):
        task = self.task_flow[task_name]
        start = time.time()

        move_flag = False
        retry_cnt = 0
        p_action = 0
        while time.time() - start < timeout:
            action = task[p_action]

            print(f"[{p_action}] Taking Action: {action}")
            act_type, is_good = self._take_action(action)
            print(f"[{p_action}] Action {'Passed!' if is_good else 'Failed!'}")

            if act_type == "move":
                move_flag = True
            elif move_flag:
                self._take_action({"action": "stop_move"})
                self._take_action({"action": "stop_move"})
                move_flag = False

            if is_good:
                retry_cnt = 0
                p_action += 1
                if p_action >= len(task):
                    return True
            else:
                retry_cnt += 1
                if retry_cnt >= max_retry: 
                    return False
        else: 
            return False
        
if __name__ == "__main__":
    import json
    config = json.load(open("./configs/client.json"))
    client = AutoPilotClient("SKM_16449", config)
    
    print(client.sync_click(Position(1,2)))
    # print(client.sync_action(Action(None, False, False, False)))
    
    # client.try_task("restart")
    
    # while True:
    #     res, distance = client._detect_ckpt("portal", timeout=0.25)
    #     print(res, distance)
    #     time.sleep(0.25)
    # client.sync_action("SKM_16448", Action(None, False, False, False))
    
    # client.sync_action(Action(pi/2, False, False, False))
    # time.sleep(1)
    # client.sync_action(Action(0, False, False, False))
    # time.sleep(5.5)
    # client.sync_action(Action(-pi/2, False, False, False))
    # time.sleep(3.5)
    # client.sync_action(Action(pi, False, False, False))
    # time.sleep(4)
    # client.sync_action(Action(-pi/4, False, False, False))
    # time.sleep(1.5)
    # client.sync_action(Action(None, False, False, False))
