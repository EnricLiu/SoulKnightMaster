import torch
from torch import nn
from stable_baselines3.common.policies import BaseFeaturesExtractor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, EfficientNet_V2_S_Weights, efficientnet_v2_s
from torch.nn.functional import one_hot


class ActorModel(BaseFeaturesExtractor):
    # def __init__(self, gru_hs=256, gru_layers=1, hardcode_hs=128, minimap_out_dim=64, health_out_dim=16,
    #              train_label=["move", "angle", "attack", "skill"]):
    
    def __init__(self, obs_space, gru_hs=768, gru_layers=2, hardcode_hs=128, minimap_out_dim=64, health_out_dim=16, out_feature_dim=1024,
                train_label=["move", "angle", "attack", "skill"]):
        print("[ActorInit] initing parent...")
        super(ActorModel, self).__init__(obs_space, out_feature_dim)

        self.train_label = train_label
        print("[ActorInit] loading model...")
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.effi_net = efficientnet_v2_s(weights=weights)
        gru_in_dim = self.effi_net.classifier[1].in_features  # 1280 for EfficientNet V2 S
        self.effi_net.classifier = nn.Identity()  # Remove original classifier
        
        # frozen [0]-[5], [6: 0-9]
        for i in range(6):
            for param in self.effi_net.features[i].parameters():
                param.requires_grad = False
        for i in range(10):
            for param in self.effi_net.features[6][i].parameters():
                param.requires_grad = False

        # self.branches = nn.ModuleDict({
        #     label: nn.ModuleDict({
        #         'gru': nn.GRU(gru_in_dim + hardcode_hs, gru_hs, batch_first=True, num_layers=gru_layers),
        #         'classifier': nn.Sequential(
        #             nn.Linear(gru_hs , 1024),
        #             nn.BatchNorm1d(1024),
        #             nn.ReLU(),
        #             nn.Dropout(0.5),
        #             nn.Linear(1024, 768),
        #             nn.BatchNorm1d(768),
        #             nn.ReLU(),
        #             nn.Dropout(0.3),
        #             nn.Linear(768, 384)
        #         )
        #     }) for label in ["move", "angle", "attack", "skill"]
        # })
        
        self.gru = nn.GRU(gru_in_dim + hardcode_hs, gru_hs, batch_first=True, num_layers=gru_layers)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hs , 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, out_feature_dim)
        )
        
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
            nn.Linear(minimap_out_dim + health_out_dim, hardcode_hs),
            nn.LayerNorm(hardcode_hs),
            nn.ReLU()
        )
        print("[ActorInit] Finish!")

    def forward(self, obs):
        # frame: (B, 10, 3, 90, 160), minimap: (B, 10, 3, 5, 5), health: (B, 10, 3)
        frame, minimap, health = obs["frames"], obs["minimap"], obs["health"]
        bs, seq_len, C, H, W = frame.shape
        frame = frame.view(bs * seq_len, C, H, W)
        frame_features = self.effi_net(frame)                   # [bs * seq_len, gru_in_dim]
        frame_features = frame_features.view(bs, seq_len, -1)   # [bs,  seq_len, gru_in_dim]
        
        bs, seq_len, C, H, W = minimap.shape
        minimap = minimap.view(bs * seq_len, C, H, W)           # [bs * seq_len, 3, 5, 5]
        # minimap = minimap[:,-1,:,:,:]                           # [bs, 3, 5, 5]
        
        bs, seq_len, dim = health.shape
        health = health.view(bs * seq_len, dim)                 # [bs * seq_len, 3]
        # health = health[:,-1,:]                                 # [bs, 3]
        
        # actions = []
        # for idx, label in enumerate(self.train_label):
        #     graph_features = self.map_cnn(minimap)                      # [bs * seq_len, minimap_out_dim]
        #     graph_features = graph_features.view(bs, seq_len, -1)       # [bs, seq_len, minimap_out_dim]
        #     health_features = self.health_encoder(health)               # [bs * seq_len, health_out_dim]
        #     health_features = health_features.view(bs, seq_len, -1)     # [bs, seq_len, health_out_dim]
        #     hardcoded_features = \
        #         self.hardcoded_head(torch.cat([graph_features, health_features], dim=2))   # [bs, seq_len, hardcode_hs]

        #     gru_in = torch.cat([frame_features, hardcoded_features], dim=2)     # [bs, seq_len, gru_in_dim + hardcode_hs]
        #     features, _ = self.branches[label]['gru'](gru_in)                   # [bs, seq_len, gru_hs]
        #     action = self.branches[label]['classifier'](features[:,-1,:])       # [bs, 1]
        #     actions.append(action.squeeze(-1))
                    
        #     # graph_features = self.map_cnn(minimap)                      # [bs, minimap_out_dim]
        #     # health_features = self.health_encoder(health)               # [bs, health_out_dim]
        #     # hardcoded_features = \
        #     #     self.hardcoded_head(torch.cat([graph_features, health_features], dim=1))
            
        #     # features, _ = self.branches[label]['gru'](frame_features)   # [bs, seq_len, gru_hs]
        #     # action = self.branches[label]['classifier'](torch.cat([features[:,-1,:], hardcoded_features], dim=1))
        #     # actions.append(action.squeeze(-1))

        # actions = torch.stack(actions, dim=1)
        # actions[:,1] *= 3
        # return actions
        
        graph_features = self.map_cnn(minimap)                      # [bs * seq_len, minimap_out_dim]
        graph_features = graph_features.view(bs, seq_len, -1)       # [bs, seq_len, minimap_out_dim]
        health_features = self.health_encoder(health)               # [bs * seq_len, health_out_dim]
        health_features = health_features.view(bs, seq_len, -1)     # [bs, seq_len, health_out_dim]
        hardcoded_features = \
            self.hardcoded_head(torch.cat([graph_features, health_features], dim=2))   # [bs, seq_len, ha

        gru_in = torch.cat([frame_features, hardcoded_features], dim=2)     # [bs, seq_len, gru_in_dim + 
        features, _ = self.gru(gru_in)                                      # [bs, seq_len, gru_hs]
        out_feature = self.classifier(features[:,-1,:])                     # [bs, 1024]
        
        return out_feature


class ValueModel(BaseFeaturesExtractor):
    def __init__(self, obs_space, gru_hs=256, gru_layers=1, hardcode_hs=128, minimap_out_dim=64, health_out_dim=16):
        super().__init__()

        self.effi_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.effi_net.classifier[1].in_features  # 1280
        self.effi_net.classifier = nn.Identity()  # Remove original classifier
        
        for i in range(7):
            for param in self.effi_net.features[i].parameters():
                param.requires_grad = False
        
        # self.gru = nn.GRU(in_features, gru_hs, batch_first=True, num_layers=gru_layers)
        gru_hs = in_features

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
                nn.Linear(gru_hs + hardcode_hs, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )

    def forward(self, x) -> torch.Tensor:
        frame, minimap, health = x  # frame: (B, 10, 3, 180, 320), minimap: (B, 3, 5, 5), health: (B, 3)

        frame = frame[:,-1,:,:,:]
        bs, C, H, W = frame.shape
        frame_features = self.effi_net(frame)
        frame_features = frame_features.view(bs, -1)
        # frame_features, _ = self.gru(frame_features)
        # frame_features = frame_features[:, -1, :]

        graph_features = self.map_cnn(minimap)
        health_features = self.health_encoder(health)

        hardcoded_features = self.hardcoded_head(torch.cat([graph_features, health_features], dim=1))

        features = torch.cat([frame_features, hardcoded_features], dim=1)
        out = self.classifier(features)

        return out


if __name__ == "__main__":
    # data = torch.zeros(1, 10, 3, 360, 640)
    effi_net = ActorModel()
    
    # for name, param in effi_net.effi_net.features.named_parameters():
    #     print(name)
        
