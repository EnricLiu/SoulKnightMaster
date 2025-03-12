import time
import json
from math import pi
import numpy as np
from PIL import Image

from httb import Client as httb
from action import Action
from position import Position

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
        return res.status_code()
    
    def fetch_fb(self) -> np.ndarray:
        res = self.client.get(f"{self.url}/fb?node={self.node_name}")
        return np.frombuffer(res.content(), dtype=np.uint8).reshape(720, 1280, 4)[:,:,:3]

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
        return res.status_code()


class AutoPilotClient(Client):
    ckpts = ["choose_character", "character_start", "character_started", "choose_mode", "portal"]

    def __init__(self, node_name, config):
        super().__init__(node_name, config["ip"], config["port"])

        self.task_flow = config["task_flow"]
        self.ckpts = {}
        self.load_ckpts(config["ckpts"])

    def dhash(self, img: Image, resize=(32, 18)):
        img = img.resize(resize, Image.Resampling.LANCZOS)
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
            self.ckpts[key] = (threshold, self.dhash(image))

    def _detect_ckpt(self, ckpt_name: str, timeout: int):
        lowest_distance = 114514
        thresh, target_dhash = self.ckpts[ckpt_name]

        timeout = time.time() + timeout
        while time.time() < timeout:
            image = Image.fromarray(self.fetch_fb())
            dhash = self.dhash(image)
            is_same, distance = self.judge_by_dhash(dhash, target_dhash, thresh)
            lowest_distance = min(lowest_distance, distance)
            if is_same:
                return True, distance
            time.sleep(0.1)
        else:
            return False, lowest_distance


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
                status_code = self.sync_click(Position(x, y))
                is_good = status_code == 200
            case "move":
                direction = action["direction"] * pi / 180
                duration = action["duration"]
                status_code = self.sync_action(Action(direction, False, False, False))
                if status_code != 200:
                    is_good = False
                else:
                    time.sleep(duration)
                    is_good = True
            case "stop_move":
                status_code = self.sync_action(Action(None, False, False, False))
                is_good = status_code == 200
            case "wait":
                time.sleep(action["time"])
                is_good =True

        return act_type, is_good


    def try_task(self, task_name, timeout=45, max_retry=3):
        task = self.task_flow[task_name]
        start = time.time()

        move_flag = False
        retry_cnt = 0
        p_ensure = 0
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
                move_flag = False

            if is_good:
                if act_type == "detect": p_ensure = p_action
                p_action += 1
                if p_action >= len(task):
                    return True
            else:
                p_action = p_ensure
                retry_cnt += 1
                if retry_cnt >= max_retry: 
                    return False
        else: 
            return False
        
if __name__ == "__main__":
    import json
    config = json.load(open("./configs/client.json"))
    client = AutoPilotClient("SKM_16448", config)
    
    client.try_task("restart")
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
