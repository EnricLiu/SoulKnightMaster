import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights,efficientnet_v2_s


class MainModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, train_label=[]):
        super(MainModel, self).__init__()
        self.train_label = train_label
        
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.effi_net = efficientnet_v2_s(weights=weights)
        in_features = self.effi_net.classifier[1].in_features  # 1280 for EfficientNet V2 S
        print(f"Input size: {in_features}")
        in_features_7 = self.effi_net.features[6][-1].out_channels
        # print(f"Input size: {in_features}")
        # print(f"Input size 7: {in_features_7}")
        self.effi_net.classifier = nn.Identity()  # Remove original classifier

        self.branches = nn.ModuleDict({
            label: nn.ModuleDict({
                'gru': nn.GRU(in_features, hidden_size, batch_first=True, num_layers=num_layers),
                'classifier': nn.Sequential(
                    nn.Linear(hidden_size, 1024),
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

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.shape
        # Reshape for EfficientNet: [batch_size * sequence_length, C, H, W]
        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.effi_net(x)  # [batch_size * sequence_length, 1280]
        # Reshape for GRU: [batch_size, sequence_length, 1280]
        features = features.view(batch_size, sequence_length, -1)
        out_list = []
        # GRU processing
        for idx, label in enumerate(self.train_label):
            gru = self.branches[label]['gru']
            classifier = self.branches[label]['classifier']
            gru_out, _ = gru(features)  # [batch_size, sequence_length, hidden_size]
            # Reshape for classifier: [batch_size * sequence_length, hidden_size]
            gru_out = gru_out.reshape(batch_size * sequence_length, -1)
            out = classifier(gru_out)  # [batch_size * sequence_length, 1]
            # Reshape back: [batch_size, sequence_length, 1]
            out = out.view(batch_size, sequence_length, 1)
            # out_list[:, :, idx] = out.squeeze(-1)
            out_list.append(out.squeeze(-1))

        return torch.stack(out_list, dim=2)
    
    def activate_branch(self, label):
        # 冻结所有分支
        for branch in self.branches.values():
            for param in branch.parameters():
                param.requires_grad = False
        # 激活当前分支
        for param in self.branches[label].parameters():
            param.requires_grad = True


class BranchesModel(nn.Module):
    def __init__(self, in_features=1280, hidden_size=256, num_layers=1, train_label=["move", "angle", "attack", "skill"]):
        super(BranchesModel, self).__init__()
        self.train_label = train_label
        self.branches = nn.ModuleDict({
            label: nn.ModuleDict({
                'gru': nn.GRU(in_features, hidden_size, batch_first=True, num_layers=num_layers),
                'classifier': nn.Sequential(
                    nn.Linear(hidden_size, 1024),
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
        
    def forward(self, features):
        batch_size, seq_len, feature_dim = features.shape
        out_list = []
        for idx, label in enumerate(self.train_label):
            gru = self.branches[label]['gru']
            classifier = self.branches[label]['classifier']
            gru_out, _ = gru(features)  # [batch_size, seq_len, hidden_size]
            # Reshape for classifier: [batch_size * seq_len, hidden_size]
            gru_out = gru_out.reshape(batch_size * seq_len, -1)
            out = classifier(gru_out)  # [batch_size * seq_len, 1]
            # Reshape back: [batch_size, seq_len, 1]
            out = out.view(batch_size, seq_len, 1)
            # out_list[:, :, idx] = out.squeeze(-1)
            out_list.append(out.squeeze(-1))
            
        # pred_action = torch.stack(out_list, dim=2)[:, -1, :] #[bs, 4]
        
        out_list = [
            out_list[0] > 0.5,                 # move
            ((out_list[1] + 1) * 128) % 256,   # angle
            out_list[2] > 0.5,                 # attack
            out_list[3] > 0.5,                 # skill
        ]
        
        pred_action = torch.stack(out_list, dim=-1) # [bs, 2+256+2+2]
        return pred_action
    
class ValueModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=1):
        super(ValueModel, self).__init__()
        self.value_net = nn.ModuleDict({
            'gru': nn.GRU(input_size=1280, hidden_size=256, num_layers=1, batch_first=True),
            'classifier': nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
        })
        
    def forward(self, obs):
        # print(obs.shape)
        gru_out, _ = self.value_net['gru'](obs)                         # [bs, seq_len, hidden_size]
        value = self.value_net['classifier'](gru_out)                   # [bs, 1]
        return value