import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from minimap import SoulKnightMinimap
from torch.nn.functional import avg_pool2d


class ValueModel(nn.Module):
    def __init__(self, minimap_config, gru_hs=128, hardcode_hs=128, minimap_out_dim=64, health_out_dim=32, num_layers=2):
        super().__init__()

        self.minimap = SoulKnightMinimap(minimap_config)
        
        self.effi_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.effi_net.classifier[1].in_features # 1280
        self.effi_net.classifier = nn.Identity()  # Remove original classifier
        self.gru = nn.GRU(in_features, gru_hs, batch_first=True, num_layers=num_layers)
        
        # --- 小地图处理子分支 ---
        self.map_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, minimap_out_dim)
        )

        # --- 定值特征子分支 ---
        self.health_encoder = nn.Sequential(
            nn.Linear(3, health_out_dim),  # HP, MP, Shield
            nn.ReLU()
        )

        # --- 合并层 ---
        self.hardcoded_head = nn.Sequential(
            nn.Linear(minimap_out_dim + health_out_dim, 128),
            nn.LayerNorm(hardcode_hs),
            nn.ReLU()
        )
        
        self.classifier = \
            nn.Sequential(
                nn.Linear(gru_hs + hardcode_hs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

    def forward(self, x) -> torch.Tensor:
        # 输入字典包含：map_grid, position, hp_mp_shield
        # map_grid: (B, 1, 5, 5)
        # position: (B, 4) → [sin(x), cos(x), sin(y), cos(y)]
        # hp_mp_shield: (B, 3)
        frame, minimap, health = x # frame: (B, 10, 3, 90, 160), minimap: (B, 2, 5, 5), health: (B, 3)
        print(frame,  minimap, health)
        bs, seq_len, C, H, W = frame.shape
        frame = frame.view(bs * seq_len, C, H, W)
        frame_features = self.effi_net(frame)
        frame_features = frame_features.view(bs, seq_len, -1)
        frame_features, _ = self.gru(frame_features)
        frame_features = frame_features[:, -1, :]
        
        graph_features = self.map_cnn(minimap)
        health_features = self.health_encoder(health)
        hardcoded_features = self.hardcoded_head(torch.cat([graph_features, health_features], dim=1))
        
        features = torch.cat([frame_features, hardcoded_features], dim=1)
        print(features.shape) # [1, 256]
        out = self.classifier(features)
        
        return out
        

if __name__ == "__main__":
    import json
    H, W = (5, 5)
    MINIMAP_CFG = json.load(open("./configs/minimap.json"))
    curr_pos = (1,0)
    pos_input = np.zeros((1, 1, H, W), dtype=np.float32)
    pos_input[:, :, curr_pos[0], curr_pos[1]] = 1
    
    curr_graph = np.random.randint(0, 15, (1, 2, 3, 4)).astype(np.uint8)
    graph_input = np.zeros((1, curr_graph.shape[1], H, W), dtype=np.float32)
    graph_input[:, :, :curr_graph.shape[2], :curr_graph.shape[3]] = curr_graph
    
    map_input = np.concatenate([pos_input, graph_input], axis=1)
    frame_input = torch.zeros((2, 10, 3, 90, 160))
    map_input = torch.zeros((2, 3, 5, 5))
    health_input = torch.zeros((2, 3))
    input = (frame_input, map_input, health_input)
    model = ValueModel(MINIMAP_CFG)
    print(model(input))