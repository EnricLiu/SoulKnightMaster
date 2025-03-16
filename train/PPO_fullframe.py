import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from torch import nn
from torch.nn.functional import one_hot, avg_pool2d
from pathlib import Path
import time, json

from model_fullframe import ActorModel as MainModel, ValueModel, ActorModel
from env_fullframe import SoulKnightEnv

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATH_COMBINE = Path("./pretrain/bn/1741430443-ln=skill-loss=8.477-e3.pth")
STATE_CONFIG = {
    "region_coords": {
        "blood": (51, 11, 230, 36),
        "shield": (51, 45, 230, 70),
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

class SoulKnightMasterPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU,  *args, **kwargs):
        print("[SKMPolicyInit] initing parent...")
        super().__init__(observation_space, action_space, lr_schedule, net_arch=net_arch, activation_fn=activation_fn,  *args, **kwargs)

        # 定义主模型结构
        # self.action_net = MainModel().branches
        print("[SKMPolicyInit] initing actor model...")
        self.action_net = ActorModel()
        print("[SKMPolicyInit] initing value model...")
        self.value_net = ValueModel()
        
        # try:
            # 安全加载模型参数（添加weights_only=True）
        print("[SKMPolicyInit] loading pretrained weights...")
        # checkpoint = torch.load(CKPT_PATH_COMBINE)
        # self.action_net.load_state_dict(checkpoint.state_dict(), strict=False)
        # except Exception as e:
        #     print(f"加载预训练权重失败: {str(e)}")
        print("[SKMPolicyInit] Finish!")

    def forward(self, obs, deterministic=False):
        action = self.action_net(obs)
        values = self.predict_values(obs)               # [n_envs, 1]
        
        distribution = self._get_action_dist_from_latent(action)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        action = self.action_net(obs)
        distribution = self._get_action_dist_from_latent(action)
        log_prob = distribution.log_prob(actions)
        values = self.predict_values(obs)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy
    
    def predict_values(self, obs) -> torch.Tensor:
        frames, minimaps, healths = obs["frames"], obs["minimap"], obs["health"]
        n_envs, seq_len, C, H, W = frames.shape                             # [n_envs, 10, 3, 180, 320]
        frames = frames.view(n_envs * seq_len, C, H, W)
        frames = avg_pool2d(frames, kernel_size=2, stride=2)                # [n_envs, 10, 3,  90, 160]
        frames = frames.view(n_envs, seq_len, C, H//2, W//2)
        
        values = self.value_net((frames, minimaps[:,-1], healths[:,-1]))    # [n_envs, 1]

        return values
    
    def _get_action_dist_from_latent(self, latent_pi):
        # 返回动作分布
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    def _get_value_dist_from_latent(self, latent_vf):
        # 返回价值估计
        return latent_vf

if __name__ == "__main__":
    import json
    log_path = Path("./logs/")
    log_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading Config")
    client_config = json.load(open("./configs/client.json"))
    minimap_config = json.load(open("./configs/minimap.json"))
    print("Making Env")
    env = make_vec_env(lambda: SoulKnightEnv("cuda", client_config, minimap_config), n_envs=1)
    print("Making Model")
    try:
        start = round(time.time())
        model = \
            PPO(SoulKnightMasterPolicy, env,
                tensorboard_log=str(log_path.absolute()), verbose=2, 
                n_steps=192, batch_size=2, learning_rate=1e-1, n_epochs=1, ent_coef=0.01
            )
        print("Start Learning")
        model.learn(
            total_timesteps =1_000_000,
            reset_num_timesteps = True,
            tb_log_name = "SoulKnightMaster_FullFrame"
        )
    except Exception as e:
        # print(e)
        raise e
    finally:
        train_duration = round(round(time.time()) - start)
        print("Model Saving")
        # if model is not None:
        #     model.save(f"./ckpt/fullframe/{start}_step={train_duration}")
