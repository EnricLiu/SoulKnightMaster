import numpy as np
import torch
from torch import nn
unlocked_map = np.array([[[2, 0, 1],
                [1, 3, 0],
                [0, 1, 2],
                [1, 0, 0],
                [0, 0, 1]],
                [[2, 0, 1],
                [1, 3, 0],
                [0, 1, 2],
                [2, 0, 0],
                [0, 0, 1]]])
C, H, W = unlocked_map.shape
full_map = np.zeros((2,5,5))
full_map[:,:H, :W] = unlocked_map
# print(full_map)
stats_dict = {"map_grid": torch.tensor(full_map).unsqueeze(0),
                "position": torch.tensor([0, 1]).unsqueeze(0),
                "hp_mp_shield": torch.tensor([4, 100, 6]).unsqueeze(0)}


class StatsBranch(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 小地图处理子分支 ---
        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 输入 (1,5,5)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (16,2,2)
            nn.Conv2d(16, 32, kernel_size=3),  # 输出 (32,0,0) → 需调整padding
            nn.AdaptiveAvgPool2d(1),  # 输出 (32,1,1)
            nn.Flatten(),
            nn.Linear(32, 64)
        )

        # --- 位置编码子分支 ---
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        # --- 定值特征子分支 ---
        self.stats_encoder = nn.Sequential(
            nn.Linear(3, 16),  # HP, MP, Shield
            nn.ReLU()
        )

        # --- 合并层 ---
        self.combined = nn.Sequential(
            nn.Linear(64 + 16 + 16, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def forward(self, stats_dict: dict) -> torch.Tensor:
        # 输入字典包含：map_grid, position, hp_mp_shield
        # map_grid: (B, 1, 5, 5)
        # position: (B, 4) → [sin(x), cos(x), sin(y), cos(y)]
        # hp_mp_shield: (B, 3)

        # 处理小地图
        map_features = self.map_cnn(stats_dict["map_grid"])

        # 处理位置
        pos_features = self.pos_encoder(stats_dict["position"])

        # 处理定值特征
        stats_features = self.stats_encoder(stats_dict["hp_mp_shield"])

        # 合并所有特征
        combined = torch.cat([map_features, pos_features, stats_features], dim=1)
        return self.combined(combined)


