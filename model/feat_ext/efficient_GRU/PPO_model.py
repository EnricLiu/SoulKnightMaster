import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from train import MainModel
from pathlib import Path
import gym
import numpy as np
from typing import Tuple, Dict, Any
from torchinfo import summary
from torchvision import transforms
import time
from PIL import Image

CKPT_PATH_COMBINE = Path("../ckpt/1741430443-ln=skill-loss=8.477-e3.pth")
STATE_CONFIG = {
    "region_coords": {
        "blood": (51, 11, 230, 36),
        "Shield": (51, 45, 230, 70),
        "mana": (51, 78, 230, 103),
        "mini_map": (1060, 120, 1258, 321),
        "self": (1144, 204, 1176, 236)
    },
    "thresholds": {
        "blood": [410000, 380000, 360000],
        "shield": [570000, 530000, 490000, 460000, 430000, 390000],
        "mana": [510000, 450000, 390000],
        "self_combat": 37000
    }
}

class GameEnv(gym.Env):
    def __init__(self):
        super(GameEnv, self).__init__()

        # 定义观测空间（处理后的图像输入）
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 10, 3, 1280, 720),
            dtype=np.uint8
            # 经过预处理的图像尺寸
        )

        # 定义动作空间（根据游戏需求）

        # 根据你的动作空间修改action_space
        self.action_space = gym.spaces.Discrete(4)
        self.running_timer = 0  # 跑图状态计时
        self.in_portal = False
    def reset(self):
        """ 重置环境并返回初始观测 """
        # 1. 获取原始游戏画面
        raw_frame = self._capture_game_screen()  # 需实现实际截图逻辑

        # 2. 预处理图像作为观测输入
        processed_frame = self._preprocess_frame(raw_frame)
        self.current_frame = processed_frame

        # 3. 初始化状态跟踪
        self.prev_state = self._extract_game_state(raw_frame)
        self.running_duration = 0

        return processed_frame

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """ 执行动作并返回环境反馈 """
        # 1. 执行游戏动作
        self._execute_action(action)  # 需实现实际动作执行逻辑
        # 2. 获取新状态
        new_raw_frame = self._capture_game_screen()
        new_frame = self._preprocess_frame(new_raw_frame)
        new_state = self._extract_game_state(new_raw_frame)
        # 3. 计算奖励
        reward = self._calculate_reward(new_state)
        # 4. 判断回合终止
        done = self._check_done_condition(new_state)
        # 5. 更新状态跟踪
        self.prev_state = new_state
        self.current_frame = new_frame
        self.running_duration += 1

        return new_frame, reward, done, {}

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

        # 状态判断逻辑
        state["in_portal"] = self._check_portal_state(state)
        state["combat_state"] = state["self"] <= STATE_CONFIG["self_combat"]
        state["blood_level"] = self._quantize_value(state["blood"], STATE_CONFIG["thresholds"]["blood"])
        state["shield_level"] = self._quantize_value(state["Shield"], STATE_CONFIG["thresholds"]["shield"])
        state["mana_level"] = self._quantize_value(state["mana"], STATE_CONFIG["thresholds"]["mana"])

        return state

    def _calculate_reward(self, current_state: dict) -> float:
        """ 基于状态变化的奖励计算 """
        reward = 0.0

        # 传送门奖励（优先级最高）
        if current_state["in_portal"] and not self.prev_state["in_portal"]:
            return 5.0

        # 血量变化
        blood_diff = current_state["blood_level"] - self.prev_state["blood_level"]
        if current_state["blood_level"] == 0:
            reward -= 100
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
        if mana_diff < 0:
            if self.prev_state["mana_level"] == 2 and current_state["mana_level"] == 1:
                reward -= 1
            elif self.prev_state["mana_level"] == 1 and current_state["mana_level"] == 0:
                reward -= 20
        elif mana_diff > 0:
            reward += abs(mana_diff) * 5

        # 状态转换奖励
        if not current_state["combat_state"] and self.prev_state["combat_state"]:
            reward += 1
            self.running_duration = 0
        elif not current_state["combat_state"]:
            self.running_duration += 1
            if self.running_duration >= 200:  # 假设每步0.1秒，20秒=200步🐮🐮🐮🐮🐮
                reward -= 1
                self.running_duration = 0

        return reward

    def _quantize_value(self, value: float, thresholds: list) -> int:
        """ 将连续值量化为离散等级 """
        for i, th in enumerate(thresholds):
            if value > th:
                return len(thresholds) - i
        return 0

    def _check_portal_state(self, state: dict) -> bool:
        """ 判断是否在传送门中 """
        portal_cond1 = (state["blood"] == state["Shield"] == state["mana"])
        portal_cond2 = (state["blood"] < 3e5 or state["Shield"] < 3e5 or state["mana"] < 3e5)
        return portal_cond1 or portal_cond2

    def _check_done_condition(self, state: dict) -> bool:
        """ 判断回合是否结束 """
        return state["blood_level"] == 0  # 空血时结束

    # 以下需根据实际游戏接口实现
    def _capture_game_screen(self) -> np.ndarray:#🐮🐮🐮🐮🐮
        """ 游戏截图实现（） """

        return np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)

    def _execute_action(self, action: int):#🐮🐮🐮🐮🐮
        """ 实际动作执行逻辑（） """
        print(f"Executing action: {action}")

class SoulKnightMasterPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU,  *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=net_arch, activation_fn=activation_fn,  *args, **kwargs)

        # 定义主模型结构
        self.action_net = MainModel()
        self.main_model = MainModel()
        # try:
            # 安全加载模型参数（添加weights_only=True）
        checkpoint = torch.load(CKPT_PATH_COMBINE)
        self.action_net.load_state_dict(checkpoint.state_dict(), strict=True)
        self.features_extractor = self.main_model.effi_net
        self.features_extractor.load_state_dict(checkpoint.state_dict(), strict=False)

        # except Exception as e:
        #     print(f"加载预训练权重失败: {str(e)}")

        self.value_net = nn.Sequential(
                    nn.Linear(1280, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 1)
                )

    def forward(self, obs, deterministic=False):
        action_logits = self.action_net(obs)
        with torch.no_grad():
            features = self.features_extractor(obs[:, -1, :, :, :])
        value = self.value_net(features)

        return action_logits, value

    def _get_action_dist_from_latent(self, latent_pi):
        # 返回动作分布
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    def _get_value_dist_from_latent(self, latent_vf):
        # 返回价值估计
        return latent_vf

if __name__ == "__main__":
    env = make_vec_env(lambda: GameEnv(), n_envs=1)
    start = time.time()
    model = PPO(SoulKnightMasterPolicy, env, verbose=2, n_steps=2, batch_size=2, learning_rate=1e-3, n_epochs=10,)
    model.learn(total_timesteps=1000)
    model.save("ppo_soul_knight")
    # for name, param in model.policy.named_parameters():
    #     print(name)
    print("Time:", time.time() - start)
    # summary(model.policy, (1, 10, 3, 1280, 720))
