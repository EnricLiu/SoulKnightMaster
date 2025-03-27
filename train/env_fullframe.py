from pathlib import Path
import math, json

import wandb
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
from torch.nn.functional import avg_pool2d
from torchvision import transforms

from utils import Action
from utils import Position
from client import AutoPilotClient
from minimap import SoulKnightMinimap

CKPT_PATH_COMBINE = Path("./pretrain/bn/1741430443-ln=skill-loss=8.477-e3.pth")
STATE_CONFIG = {
    "region_coords": {
        "blood":    (51, 11, 230, 36),
        "shield":   (51, 45, 230, 70),
        "mana":     (51, 78, 230, 103),
        "mini_map": (1060, 120, 1258, 321),
        "self":     (1144, 204, 1176, 236)
    },
    "thresholds": {
        "blood":    [410000, 380000, 360000],
        "shield":   [570000, 530000, 490000, 460000, 430000, 390000],
        "mana":     [510000, 450000, 390000],
        "self_combat": 37000
    }
}

WANDB_CFG = json.load(open("./configs/wandb.json"))
TRAIN_PARAMS = {
    "n_steps":          128,
    "batch_size":       2,
    "learning_rate":    3e-2,
    "n_epochs":         1,
    "ent_coef":         0.01,
}

    

class SoulKnightEnv(gym.Env):
    def __init__(self, device, autopilot_config: dict, minimap_config: dict, name:str, logger=None):
        print("[EnvInit] initing parent...")
        super(SoulKnightEnv, self).__init__()
        
        print("[EnvInit] making client...")
        self.name = name
        self.logger = logger
        self.device = device
        self.client = AutoPilotClient(name, autopilot_config)
        self.minimap = SoulKnightMinimap(minimap_config)
        # self.observation_space = gym.spaces.Box(
        #     low =   -float("inf"),
        #     high =   float("inf"),
        #     shape = (10, 3, 720, 1280),
        #     dtype = np.float32
        # )
        
        self.observation_space = gym.spaces.Dict({
            "frames": gym.spaces.Box(
                low =   -float("inf"),
                high =   float("inf"),
                shape = (10, 3, 180, 320),
                dtype = np.float32
            ),
            "minimap": gym.spaces.Box(
                low =   -float("inf"),
                high =   float("inf"),
                shape = (10, 3, 5, 5),
                dtype = np.float32
            ),
            "health": gym.spaces.Box(
                low =   -float("inf"),
                high =   float("inf"),
                shape = (10, 3),
                dtype = np.float32
            ),
        })
        
        print("[EnvInit] making obs frames...")
        self.obs = {
            "frames":   torch.zeros((10, 3, 180, 320), dtype=torch.float32).to("cpu"),
            "minimap":  torch.zeros((10, 3, 5, 5), dtype=torch.float32).to("cpu"),
            "health":   torch.zeros((10, 3), dtype=torch.float32).to("cpu"),
        }
        # is_move[0-1], move_angle[0-255], is_attack[0-1], is_skill[0-1]
        # self.action_space = gym.spaces.MultiDiscrete([2, 256, 2, 2])
        self.action_space = gym.spaces.Box(
            low =   -1000000,
            high =   1000000,
            shape = (4,),
            dtype = np.float32
        )
        self.running_timer = 0
        self.in_portal = False
        self.last_target_dir = 0
        self.idle_cnt = 0
        
        self.steps = 0
        self.rewards = np.zeros((128), dtype=np.float32)
        print("[EnvInit] Finish!")
        
    def show_reward(self) -> float:
        self.steps += 1
        if self.steps % 128 == 0:
            print(f"reward: {self.rewards.mean().item():.4f}")
            
    
    def reset(self, *args, **kwargs):
        """ 重置环境并返回初始观测 """
        self.client.sync_action(Action(None, False, False, False))
        # 1. 获取原始游戏画面
        task_res = self.client.try_task("restart", timeout=120, max_retry=3)
        self.minimap.reset()
        obs, raw_frame = self.fetch_obs()
        print(raw_frame.shape)
        # 3. 初始化状态跟踪
        self.prev_state = self._extract_game_state(raw_frame)
        self.running_duration = 0

        return obs, {"suceess": task_res}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """ 执行动作并返回环境反馈 """
        # 1. 执行游戏动作
        action: Action = Action.from_raw(action)
        self.client.sync_action(action)
        # 2. 获取新状态
        obs, raw_frame = self.fetch_obs()
        self.minimap.update(self.minimap.crop_minimap(raw_frame))
        new_state = self._extract_game_state(raw_frame)
        # 3. 计算奖励
        reward = self._calculate_reward(new_state, action, raw_frame)
        # select buff
        if self._check_portal_state(raw_frame):
            self.client.sync_click(Position(640, 640))
            self.client.sync_click(Position(640, 640))
        # 4. 判断回合终止
        terminated = self._check_done_condition(raw_frame)
        truncated = False                                                           # TODO: impl truncated
        # 5. 更新状态跟踪
        self.prev_state = new_state
        self.running_duration += 1
        # houyao
        self.show_reward()
        self.rewards = np.roll(self.rewards, 1, axis=0)
        self.rewards[-1] = reward
        print({
                "action/reward": reward,
                "action/target": self.last_target_dir,
                "action/angle": action._angle if action is not None else None,
            })
        if self.logger is not None:
            self.logger({
                "action/reward": reward,
                "action/target": self.last_target_dir,
                "action/angle": action._angle if action is not None and action._angle is not None else 0,
            }, commit=True)
                
        return obs, reward, terminated, truncated, {}
    

    def fetch_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """fetch_fb and update the frames dequeue(obs)
        
        Returns:
            tuple[]:
            (features, raw_frame): np.ndarray (1, 10, 720, 1280, 3), np.ndarray (720, 1280, 3)
        """
        raw_frame = self.client.fetch_fb()  # [720, 1280, 3]
        
        # minimap
        new_minimap = self.minimap.parse(self.minimap.crop_minimap(raw_frame))                  # [3, 5, 5]
        self.obs["minimap"] = torch.roll(self.obs["minimap"], shifts=1, dims=0)
        self.obs["minimap"][-1] = torch.from_numpy(np.stack(new_minimap))
        
        # health
        new_health = self._extract_health(raw_frame)
        self.obs["health"] = torch.roll(self.obs["health"], shifts=1, dims=0)
        self.obs["health"][-1] = torch.from_numpy(new_health)
        
        # frames
        new_frame = raw_frame.astype(np.float32)
        new_frame = torch.from_numpy(new_frame).to(self.device).permute(2, 0, 1).unsqueeze(0)   # [1, 3, 720, 1280]
        new_frame = avg_pool2d(new_frame, kernel_size=4, stride=4)[0]                           # [3, 180, 320]
        self.obs["frames"] = torch.roll(self.obs["frames"], shifts=1, dims=0)
        self.obs["frames"][-1] = new_frame.to("cpu")
        
        
        return self.obs, raw_frame
    

    def _preprocess_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """ 图像预处理流水线（关键修改点） """
        data_transform = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        processed_frame = data_transform["train"](raw_frame)

        return processed_frame
    
    def _extract_health(self, frame: np.ndarray) -> np.ndarray:
        state = {}
        for name, coords in STATE_CONFIG["region_coords"].items():
            region = frame[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            gray_region = region.dot([0.2989, 0.5870, 0.1140]).astype(np.uint8)
            state[name] = np.mean(gray_region)
            
        return np.array([state[k] for k in ["blood", "shield", "mana"]], dtype=np.float32)

    def _extract_game_state(self, frame: np.ndarray) -> dict:
        """ 从原始帧提取游戏状态（用于奖励计算） """
        state = {}

        # 计算各区域灰度值
        for name, coords in STATE_CONFIG["region_coords"].items():
            region = frame[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            gray_region = region.dot([0.2989, 0.5870, 0.1140]).astype(np.uint8)
            state[name] = np.sum(gray_region)
        def get_mana_percent(value):
            min_mana = 390000
            max_mana = 510000
            # 将值限制在区间内并计算百分比
            return max(0, min(100, (value - min_mana) / (max_mana - min_mana) * 100))


        # 状态判断逻辑
        state["in_portal"]      = self._check_portal_state(frame)
        state["combat_state"]   = state["self"] <= STATE_CONFIG["thresholds"]["self_combat"]
        state["blood_level"]    = self._quantize_value(state["blood"],  STATE_CONFIG["thresholds"]["blood"])
        state["shield_level"]   = self._quantize_value(state["shield"], STATE_CONFIG["thresholds"]["shield"])
        state["mana_level"]     = get_mana_percent(state["mana"])

        return state

    def _calculate_reward(self, current_state: dict, action: Action, raw_frame: np.ndarray) -> float:
        """ 基于状态变化的奖励计算 """
        reward = 0.0

        # # 传送门奖励（优先级最高）
        # if current_state["in_portal"]:
        #     if not self.prev_state["in_portal"]:
        #         self.minimap.reset()
        #         return 100
        #     return 0
        #
        # res, distance = self.client._detect_ckpt_to_raw("dead", raw_frame)
        # if res: reward -= 200 # dead :(
        #
        # # 血量变化
        # blood_diff = current_state["blood_level"] - self.prev_state["blood_level"]
        # if current_state["blood_level"] == 0:
        #     pass
        # elif blood_diff < 0:
        #     if self.prev_state["blood_level"] == 2 and current_state["blood_level"] == 1:
        #         reward -= 50
        # elif blood_diff > 0:
        #     reward += abs(blood_diff) * 10
        #
        # # 盾量变化
        # shield_diff = current_state["shield_level"] - self.prev_state["shield_level"]
        # reward += shield_diff * (-10 if shield_diff < 0 else 5)
        #
        # # 蓝量变化
        # mana_diff = current_state["mana_level"] - self.prev_state["mana_level"]
        # if mana_diff < 0:  # 蓝量减少
        #     reward -= mana_diff * 5 * (1 - current_state["mana_level"] / 100)
        # else:  # 蓝量增加
        #     reward += mana_diff * 2 * (1 - current_state["mana_level"] / 100)
        
        # 状态转换奖励
        # if not current_state["combat_state"] and self.prev_state["combat_state"]:
        #     reward += 10
        #     self.running_duration = 0
        #
        # if current_state["combat_state"] and not self.prev_state["combat_state"]:
        #     reward += 50
        #     self.running_duration = 0
        #
        # if current_state["combat_state"]:
        #     reward -= np.power(0.1 * self.running_duration, 1/4)
            
        if not current_state["combat_state"]:
            # minimap direction
            target_dir = self.minimap.navigate()
            if target_dir is None:
                target_dir = self.last_target_dir
            else: 
                self.last_target_dir = target_dir
            # img = self.minimap._render()
            # cv2.imshow("minimap", img)
            # key = cv2.waitKey(50)
            if self.steps % 128 == 0: print(f"[{self.name}] GOOGLE_MAP: ", target_dir)
            
            if action._angle is not None:
                self.idle_cnt = 0
                K = 3
                # action_angle = action._angle + math.pi/4
                action_angle = action._angle
                diff = (target_dir - action_angle - np.pi) % (2 * np.pi) - np.pi
                reward += np.exp(- K * np.abs(diff)) * 10 - diff ** 4 / 5
            else:
                self.idle_cnt += 1
                reward -= 5 + np.power(0.5 * self.idle_cnt, 1/4)
            # idle
            # reward -= 1
            self.running_duration += 1
            # if self.running_duration >= 100:  # 假设每步0.125秒，1.25秒=100步
            #     reward -= 0.01
            #     self.running_duration = 0
        return reward * 0.1

    def _quantize_value(self, value: float, thresholds: list) -> int:
        """ 将连续值量化为离散等级 """
        for i, th in enumerate(thresholds):
            if value > th:
                return len(thresholds) - i
        return 0

    def _check_portal_state(self, raw_frame: np.ndarray) -> bool:
        """ 判断是否在传送门中 """
        res, distance = self.client._detect_ckpt_to_raw("portal", raw_frame)
        return res

    def _check_done_condition(self, raw_frame: np.ndarray) -> bool:
        """ 判断回合是否结束 """
        res, distance = self.client._detect_ckpt_to_raw("dead", raw_frame)
        return res