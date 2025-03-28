import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from torch import nn
from torch.nn.functional import one_hot, avg_pool2d
from pathlib import Path
import time, json

from model_fullframe import ActorModel as FeatureExtractor, ValueModel
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
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=[512, 512], activation_fn=nn.ReLU,  *args, **kwargs):
        print("[SKMPolicyInit] initing parent...")
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            share_features_extractor=False,
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs={
                "out_feature_dim": 1024
            },
            activation_fn=activation_fn,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={
                # "params":   filter(lambda p: p.requires_grad, self.parameters()),
                "eps":      1e-5,
            },
            *args, **kwargs
        )


    # def forward(self, obs, deterministic=False):
    #     latent_pi = self.predict(obs)
    #     values = self.predict_values(obs)               # [n_envs, 1]
        
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     actions = distribution.get_actions(deterministic=deterministic)
    #     log_prob = distribution.log_prob(actions)
    #     # print("Raw entropy:", distribution.entropy().mean().item())
    #     # print("Action logits stats:", actions.mean().item(), actions.std().item())
    #     return actions, values, log_prob
    
    # def evaluate_actions(self, obs, actions):
    #     latent_pi = self.predict(obs)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     log_prob = distribution.log_prob(actions)
    #     values = self.predict_values(obs)
    #     entropy = distribution.entropy()
        
    #     return values, log_prob, entropy
    
    # def _predict(self, obs, deterministic: bool = False) -> torch.Tensor:
    #     frames, minimaps, healths = obs["frames"], obs["minimap"], obs["health"]
    #     latent_pi = self.action_net((frames, minimaps, healths))        # [n_envs, action_net_mid_feature_dim]
    #     actions = self.mlp_head(latent_pi)
    #     return actions
    
    # def predict_values(self, obs) -> torch.Tensor:
    #     frames, minimaps, healths = obs["frames"], obs["minimap"], obs["health"]
    #     # n_envs, seq_len, C, H, W = frames.shape                             # [n_envs, 10, 3, 180, 320]
    #     # frames = frames.view(n_envs * seq_len, C, H, W)
    #     # frames = avg_pool2d(frames, kernel_size=2, stride=2)                # [n_envs, 10, 3,  90, 160]
    #     # frames = frames.view(n_envs, seq_len, C, H//2, W//2)
        
    #     values = self.value_net((frames, minimaps[:,-1], healths[:,-1]))    # [n_envs, 1]

    #     return values
    
    # def _get_action_dist_from_latent(self, latent_pi):
    #     return self.action_dist.proba_distribution(mean_actions=latent_pi, log_std=self.log_std)


import wandb
from wandb.integration.sb3 import WandbCallback
WANDB_CFG = json.load(open("./configs/wandb.json"))
TRAIN_PARAMS = {
    "n_steps":          256,
    "batch_size":       2,
    "learning_rate":    3e-4,
    "n_epochs":         1,
    "ent_coef":         0.01,
}

if not wandb.login(key=WANDB_CFG["secret"], relogin=True, timeout=5):
    print("Failed to login to wandb")
    exit()
    
RUN = wandb.init(
    entity=WANDB_CFG["entity"],
    project="SoulKnightMaster_FullFrame",
    config=TRAIN_PARAMS,
    sync_tensorboard=True,
)


if __name__ == "__main__":
    import json
    log_path = Path("./logs/")
    log_path.mkdir(parents=True, exist_ok=True)
    
    
    print("Loading Config")
    client_config = json.load(open("./configs/client.json"))
    minimap_config = json.load(open("./configs/minimap.json"))
    
    print("Making Env")
    def make_sk_env(name):
        def _init():
            return SoulKnightEnv("cuda", client_config, minimap_config, name=name, logger=RUN.log)
        return _init

    env = DummyVecEnv([make_sk_env(name) for name in ["SKM_16448"]])
    # env = make_vec_env(lambda: SoulKnightEnv("cuda", client_config, minimap_config), n_envs=2)
    
    print("Making Model")
    model = None
    try:
        model = PPO(SoulKnightMasterPolicy, env,
                    tensorboard_log=f"runs/{RUN.name}", verbose=2, device=DEVICE,
                    **TRAIN_PARAMS)
        # model = PPO.load("./ckpt/fullframe_map_health/charmed-darkness-101-2025_03_18-15_28_09.zip",
        #                  env, tensorboard_log=f"runs/{RUN.name}", verbose=2,  **TRAIN_PARAMS)
        
        step_cnt = 0
        model_saving_callback = CheckpointCallback(8192, f"./ckpt/fullframe_map_health/{RUN.name}-{time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())}")
        callbacks = CallbackList([model_saving_callback, WandbCallback(2)])
        
        print("Start Learning")
        model.learn(
            total_timesteps = 1_000_000,
            reset_num_timesteps = True,
            tb_log_name = "SoulKnightMaster_FullFrame",
            callback=callbacks
        )
    except Exception as e:
        print(e)
        raise e
    finally:
        if model is not None:
            model.save(f"./ckpt/fullframe_map_health/{RUN.name}-{time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())}")
            print("Model Saved")
        RUN.finish()
