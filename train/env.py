from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image
import gymnasium as gym

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms

from utils import Action
from model import MainModel
from utils import Position, SoulKnightMinimap
from client import AutoPilotClient

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

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.effi_net = MainModel().effi_net
        checkpoint = torch.load(CKPT_PATH_COMBINE, weights_only=False)
        self.effi_net.load_state_dict(checkpoint.state_dict(), strict=False)
    def forward(self, x):
        return self.effi_net(x)
    
if __name__ == "__main__":
    model = FeatureExtractor().to("cuda")
    model.eval()
    with torch.no_grad():
        res = model(torch.randn(10, 3, 1280, 720).to("cuda"))
        print(res.cpu().numpy().shape)

class SoulKnightEnv(gym.Env):
    def __init__(self, device, autopilot_config: dict, minimap_config: dict):
        super(SoulKnightEnv, self).__init__()
        
        self.device = device
        self.client = AutoPilotClient("SKM_16448", autopilot_config)
        self.minimap = SoulKnightMinimap(minimap_config)
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(10, 1280),
            dtype=np.float32
        )
        
        self.features = torch.zeros((1, 10, 1280), dtype=torch.float32).to("cpu")
        # is_move[0-1], move_angle[0-255], is_attack[0-1], is_skill[0-1]
        self.action_space = gym.spaces.MultiDiscrete([2, 256, 2, 2])
        self.running_timer = 0
        self.in_portal = False
        
        self.steps = 0
        self.rewards = np.zeros((128), dtype=np.float32)
        
    def show_reward(self) -> float:
        self.steps += 1
        if self.steps % 128 == 0:
            print(f"reward: {self.rewards.mean().item():.4f}")
            
    
    def reset(self, *args, **kwargs):
        """ 重置环境并返回初始观测 """
        # 1. 获取原始游戏画面
        task_res = False
        task_res = self.client.try_task("restart", timeout=60, max_retry=3)
        self.minimap.reset()
        features, raw_frame = self.fetch_features()
        # 3. 初始化状态跟踪
        self.prev_state = self._extract_game_state(raw_frame)
        self.running_duration = 0

        return features, {"suceess": task_res}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """ 执行动作并返回环境反馈 """
        # 1. 执行游戏动作
        action = Action.from_raw(action)
        self.client.sync_action(action)
        # 2. 获取新状态
        features, new_frame = self.fetch_features()
        self.minimap.update(self.minimap.crop_minimap(new_frame))
        new_state = self._extract_game_state(new_frame)
        # 3. 计算奖励
        reward = self._calculate_reward(new_state, action)
        # select buff
        if self._check_portal_state(new_state):
            self.client.sync_click(Position(640, 640))
            self.client.sync_click(Position(640, 640))
        # 4. 判断回合终止
        terminated = self._check_done_condition(new_state)
        truncated = False                                                           # TODO: impl truncated
        # 5. 更新状态跟踪
        self.prev_state = new_state
        self.running_duration += 1
        # houyao
        self.show_reward()
        self.rewards = np.roll(self.rewards, 1, axis=0)
        self.rewards[-1] = reward

        return features, reward, terminated, truncated, {}
    

    def fetch_features(self) -> tuple[np.ndarray, np.ndarray]:
        """fetch_fb and extract features, then update the feature dequeue
        
        Returns:
            tuple[]:
            (features, raw_frame): np.ndarray (1, 10, 1280), np.ndarray (1, 1280)
        """
        
        new_feature, raw_frame = self.fetch_fb_to_feature() # (1, 1280)
        self.features = torch.roll(self.features, shifts=1, dims=1)
        self.features[:, -1, :] = new_feature.to("cpu")
        return self.features, raw_frame
        
    
    def fetch_fb_to_feature(self) -> tuple[np.ndarray, np.ndarray]:
        new_frame = self.client.fetch_fb()
        processed_frame = self._preprocess_frame(new_frame)
        with torch.no_grad():
            feature = self.feature_extractor(Tensor(processed_frame).unsqueeze(0).to(self.device))
        return feature, new_frame
    

    def _preprocess_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """ 图像预处理流水线（关键修改点） """
        data_transform = {
            "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        processed_frame = data_transform["train"](raw_frame)

        return processed_frame
    

    def _extract_game_state(self, frame: np.ndarray) -> dict:
        """ 从原始帧提取游戏状态（用于奖励计算） """
        pil_image = Image.fromarray(frame)
        state = {}

        # 计算各区域灰度值
        for name, coords in STATE_CONFIG["region_coords"].items():
            region = pil_image.crop(coords)
            gray_region = region.convert('L')
            state[name] = np.sum(np.array(gray_region))
        def get_mana_percent(value):
            min_mana = 390000
            max_mana = 510000
            # 将值限制在区间内并计算百分比
            return max(0, min(100, (value - min_mana) / (max_mana - min_mana) * 100))


        # 状态判断逻辑
        state["in_portal"]      = self._check_portal_state(state)
        state["combat_state"]   = state["self"] <= STATE_CONFIG["thresholds"]["self_combat"]
        state["blood_level"]    = self._quantize_value(state["blood"],  STATE_CONFIG["thresholds"]["blood"])
        state["shield_level"]   = self._quantize_value(state["shield"], STATE_CONFIG["thresholds"]["shield"])
        state["mana_level"]     = get_mana_percent(state["mana"])

        return state

    def _calculate_reward(self, current_state: dict, action: Action) -> float:
        """ 基于状态变化的奖励计算 """
        reward = 0.0

        # 传送门奖励（优先级最高）
        if current_state["in_portal"] and not self.prev_state["in_portal"]:
            self.minimap.reset()
            return 500

        # 血量变化
        blood_diff = current_state["blood_level"] - self.prev_state["blood_level"]
        if current_state["blood_level"] == 0:
            reward -= 2000
        elif blood_diff < 0:
            if self.prev_state["blood_level"] == 2 and current_state["blood_level"] == 1:
                reward -= 20
        elif blood_diff > 0:
            reward += abs(blood_diff) * 10

        # 盾量变化
        shield_diff = current_state["shield_level"] - self.prev_state["shield_level"]
        reward += shield_diff * (-1 if shield_diff < 0 else 2)

        # 蓝量变化
        mana_diff = current_state["mana_level"] - self.prev_state["mana_level"]
        if mana_diff < 0:  # 蓝量减少
            reward += mana_diff * 2 * (1 - current_state["mana_level"] / 100)
        else:  # 蓝量增加
            reward += mana_diff * 0.8 * (1 - current_state["mana_level"] / 100)
        
        # 状态转换奖励
        if not current_state["combat_state"] and self.prev_state["combat_state"]:
            reward += 10
            self.running_duration = 0
            
        if current_state["combat_state"] and not self.prev_state["combat_state"]:
            reward += 50
            self.running_duration = 0
            
        if not current_state["combat_state"]:
            # minimap direction
            target_dir = self.minimap.navigate()
            # img = self.minimap._render()
            # cv2.imshow("minimap", img)
            # key = cv2.waitKey(50)
            if self.steps % 128 == 0: print("GOOGLE_MAP: ", target_dir)
            
            if target_dir is not None and action._angle is not None:
                action_angle = action._angle + math.pi/4
                reward += 0.5 - ((target_dir - action_angle + math.pi) % (2 * math.pi) - math.pi) ** 2 / 10
            
            # idle
            reward -= 0.2 * math.sqrt(self.running_duration)
            self.running_duration += 1
            # if self.running_duration >= 100:  # 假设每步0.125秒，1.25秒=100步
            #     reward -= 0.01
            #     self.running_duration = 0

        return reward * 0.01

    def _quantize_value(self, value: float, thresholds: list) -> int:
        """ 将连续值量化为离散等级 """
        for i, th in enumerate(thresholds):
            if value > th:
                return len(thresholds) - i
        return 0

    def _check_portal_state(self, state: dict) -> bool:
        """ 判断是否在传送门中 """
        portal_cond1 = (state["blood"] == state["shield"] == state["mana"])
        portal_cond2 = (state["blood"] < 3e5 or state["shield"] < 3e5 or state["mana"] < 3e5)
        return portal_cond1 or portal_cond2

    def _check_done_condition(self, state: dict) -> bool:
        """ 判断回合是否结束 """
        return state["blood_level"] == 0  # 空血时结束