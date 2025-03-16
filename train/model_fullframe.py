import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, EfficientNet_V2_S_Weights, efficientnet_v2_s
from torch.nn.functional import one_hot


class ActorModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, hardcode_hs=128, minimap_out_dim=64, health_out_dim=32,
                 train_label=["move", "angle", "attack", "skill"]):
        print("[ActorInit] initing parent...")
        super(ActorModel, self).__init__()

        self.train_label = train_label
        print("[ActorInit] loading model...")
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.effi_net = efficientnet_v2_s(weights=weights)
        in_features = self.effi_net.classifier[1].in_features  # 1280 for EfficientNet V2 S
        print(f"Input size: {in_features}")
        self.effi_net.classifier = nn.Identity()  # Remove original classifier

        self.branches = nn.ModuleDict({
            label: nn.ModuleDict({
                'gru': nn.GRU(in_features, hidden_size, batch_first=True, num_layers=num_layers),
                'classifier': nn.Sequential(
                    nn.Linear(hidden_size + hardcode_hs, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 1)
                )
            }) for label in ["move", "angle", "attack", "skill"]
        })
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

        print("[ActorInit] Finish!")

    def forward(self, x):
        x = x["frames"]
        print("x shape: ", x.shape)

        exit()
        frame, minimap, health = x  # frame: (B, 10, 3, 90, 160), minimap: (B, 2, 5, 5), health: (B, 3)
        bs, seq_len, C, H, W = frame.shape
        frame = frame.view(bs * seq_len, C, H, W)
        batch_size, sequence_length, C, H, W = frame.shape
        # Reshape for EfficientNet: [batch_size * sequence_length, C, H, W]
        frame = frame.view(batch_size * sequence_length, C, H, W)
        features = self.effi_net(frame)  # [batch_size * sequence_length, 1280]
        # Reshape for GRU: [batch_size, sequence_length, 1280]
        features = features.view(batch_size, sequence_length, -1)
        out_list = []
        # GRU processing
        for idx, label in enumerate(self.train_label):
            gru = self.branches[label]['gru']
            classifier = self.branches[label]['classifier']
            gru_out, _ = gru(features)  # [batch_size, sequence_length, hidden_size]
            print(f"GRU output shape: {gru_out.shape}")
            # Reshape for classifier: [batch_size * sequence_length, hidden_size]
            # gru_out = gru_out.reshape(batch_size * sequence_length, -1)
            # actornet与valuenet出现差异，原actornet是将十个帧的输出直接reshape，在valuenet中则是取最后一个帧的输出，先更改actornet为valuenet一致的
            gru_out = gru_out[:, -1, :]
            print(f"GRU output shape: {gru_out.shape}")
            exit()
            graph_features = self.map_cnn(minimap)
            health_features = self.health_encoder(health)

            hardcoded_features = self.hardcoded_head(torch.cat([graph_features, health_features], dim=1))
            features = torch.cat([gru_out, hardcoded_features], dim=1)
            out = classifier(features)  # [batch_size * sequence_length, 1]
            # Reshape back: [batch_size, sequence_length, 1]
            out = out.view(batch_size, sequence_length, 1)
            # out_list[:, :, idx] = out.squeeze(-1)
            out_list.append(out.squeeze(-1))

        # pred_action = torch.stack(out_list, dim=2)[:, -1, :] #[bs, 4]
        move = (out_list[0] > 0.5).to(torch.int)
        out_list = [
            out_list[0] > 0.5,  # move
            ((out_list[1] + 1) * 128) % 256,  # angle
            out_list[2] > 0.5,  # attack
            out_list[3] > 0.5,  # skill
        ]

        out_list = list(map(lambda x: x.to(torch.long), out_list))
        out_list = [
            one_hot(out_list[0], 2),
            one_hot(out_list[1], 256),
            one_hot(out_list[2], 2),
            one_hot(out_list[3], 2),
        ]

        pred_action = torch.cat(out_list, dim=1)
        return pred_action

    def activate_branch(self, label):
        # 冻结所有分支
        for branch in self.branches.values():
            for param in branch.parameters():
                param.requires_grad = False
        # 激活当前分支
        for param in self.branches[label].parameters():
            param.requires_grad = True


class ValueModel(nn.Module):
    def __init__(self, gru_hs=128, hardcode_hs=128, minimap_out_dim=64, health_out_dim=32, num_layers=2):
        super().__init__()

        self.effi_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.effi_net.classifier[1].in_features  # 1280
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
        frame, minimap, health = x  # frame: (B, 10, 3, 90, 160), minimap: (B, 2, 5, 5), health: (B, 3)

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
        out = self.classifier(features)

        return out


if __name__ == "__main__":
    weights = EfficientNet_B0_Weights.DEFAULT
    data = torch.zeros(1, 10, 3, 360, 640)
    effi_net = ActorModel()
    effi_net(data)