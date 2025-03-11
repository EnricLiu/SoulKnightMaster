from pathlib import Path
import time
import polars as pl
import torch
from torch import nn
from SoulKnightMaster.model.feat_ext.dataset_test import ImageDataset, ImageDataset_Combine
from torchinfo import summary

import wandb
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, r2_score, mean_squared_error
from dataset_test import SequenceImageDataset

import os
import numpy as np
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights,efficientnet_v2_s
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
# train_label = {0: "move", 1: "angle", 2: "attack", 3: "skill", 4: "weapon"}
train_label = ["move", "angle", "attack", "skill"]

# torch.autograd.set_detect_anomaly(True)
################################################
# 登录到 W&B
wandb.login(key="")

# 初始化 W&B 项目
wandb.init(project="")
################################################
class MainModel(nn.Module):
    def __init__(self, hidden_size=256,num_layers=1):
        super(MainModel, self).__init__()
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
        for idx, label in enumerate(train_label):
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


def freeze_model_weights(model, freeze_until_layer=15):
    for i in range(6):
        for name, param in model.features[i].named_parameters():
            param.requires_grad = False
    freeze_until_layer = max(0, min(freeze_until_layer, 15))
    for j in range(freeze_until_layer):
        for name, param in model.features[6][j].named_parameters():
            param.requires_grad = False


# 训练过程
def train(model, device, dataset,
          ckpt_path: Path, save_state_dict: bool,
          val_percent=0.2, split_seed=0, batch_size=0, save_interval: int = None, use_amp=False, num_epochs=10, sequence_length=10):

    def save_ckpt(_loss, _epoch):
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if ckpt_path is None: return
        target = model
###############################################################
        wandb.unwatch(model)
###############################################################
        if save_state_dict: target = model.state_dict()
        torch.save(target, os.path.join(ckpt_path, f'{train_id}-ln={train_label[train_number]}-loss={_loss:.3f}-e{_epoch}.pth'))
        print(f'Checkpoint {_epoch} saved!')
############################################################
        wandb.watch(
            model,
            criterion=criterion,
            log="all",
            log_freq=100
        )
##########################################################
    train_id = round(time.time())

    model.to(device)

####################monitor######################
    wandb.watch(
        model,
        log="all",
        log_freq=100
    )
################################################
    criterions = [None] * len(train_label)
    def criterion(_train_number):
        if criterions[_train_number] is not None:
            return criterions[_train_number]

        if _train_number == 1:
            ret = nn.MSELoss()              # angle是回归任务
        else:
            ret = nn.BCEWithLogitsLoss()    # 其他是二分类

        criterions[_train_number] = ret
        return ret

    total_sequences = len(dataset)
    n_val = int(total_sequences * val_percent)
    n_train = total_sequences - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, total_sequences))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    loader_args = dict(batch_size=batch_size, num_workers=0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    best_loss = float("inf")
    cplt_epoches = 0
    train_losses, val_losses = [], []

    layer_num = 14
    freeze_model_weights(model.effi_net, freeze_until_layer=layer_num)
    print(f"frozen until layer: {layer_num}")
    optimizers = {}
    for train_number in range(len(train_label)):
        current_label = train_label[train_number]
        model.activate_branch(current_label)
        optimizers[train_number] = optim.Adam(  # 逐个添加键值对
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001
        )

    try:
    # with open(ckpt_path / "train_id.txt", "w") as f:
        for epoch in range(num_epochs):
            model.train()
            with tqdm(total=n_train - sequence_length + 1, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
                train_loss = {
                    "move":     0.0,
                    "angle":    0.0,
                    "attack":   0.0,
                    "skill":    0.0,
                }
                for i, images in enumerate(train_loader):
                    image = images["image"].to(device)  # [batch_size, sequence_length, C, H, W]
                    action = images["action"].to(device)  # [batch_size, sequence_length, 5]
                    action = action[:, :, :4]  # [batch_size, sequence_length, 4]
                    batch_loss = {
                        "move":     0.0,
                        "angle":    0.0,
                        "attack":   0.0,
                        "skill":    0.0,
                    }


                    for train_number in range(len(train_label)):
                        current_label = train_label[train_number]
                        # print(f"\n=== Training {current_label} ===")
                        output = model(image)

                        # 激活当前分支并设置优化器
                        model.activate_branch(current_label)
                        optimizer = optimizers[train_number]
                        optimizer.zero_grad()
                        loss = criterion(train_number)(output[:, sequence_length - 1, train_number].to(device),
                                         action[:, sequence_length - 1, train_number])
                        # print(f"output: {output[:, sequence_length - 1, train_number]}")
                        # print(f"action: {action[:, sequence_length - 1, train_number]}")
                        # print(f"loss: {loss}")

                        loss.backward()
                        optimizer.step()
                        batch_loss[current_label] += loss.item()

                    pbar.update(batch_size)
                    pbar.set_postfix(**{'loss (batch)': batch_loss})

                    train_loss = {k: v + batch_loss[k] for k, v in train_loss.items()}

            train_loss = {k: v / len(train_loader) * batch_size for k, v in train_loss.items()}

            print(f"Epoch {epoch}, Loss: {train_loss}, Total: {sum(train_loss.values())}")

            ###################################################################
            wandb.log({"epoch": epoch, "train_loss": train_loss})
            ###################################################################
            model.eval()
            val_loss = {
                "move": 0.0,
                "angle": 0.0,
                "attack": 0.0,
                "skill": 0.0,
            }
            with tqdm(total=n_val - sequence_length + 1, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
                for i, images in tqdm(enumerate(val_loader)):
                    image = images["image"].to(device)  # [batch_size, sequence_length, C, H, W]
                    action = images["action"].to(device)  # [batch_size, sequence_length, 1]
                    batch_loss = {
                        "move": 0.0,
                        "angle": 0.0,
                        "attack": 0.0,
                        "skill": 0.0,
                    }
                    with torch.no_grad():
                        output = model(image)
                    for train_number in [0, 1, 2, 3]:
                        current_label = train_label[train_number]
                        print(f"\n=== Training {current_label} ===")

                        with torch.no_grad():
                            loss = criterion(train_number)(output[:, sequence_length - 1, train_number].to(device),
                                             action[:, sequence_length - 1, train_number])
                            batch_loss[current_label] += loss.item()

                            pbar.update(len(image))
                            pbar.set_postfix(**{'loss (batch)': batch_loss})

                    val_loss = {k: v + batch_loss[k] for k, v in val_loss.items()}

            val_loss = {k: v / 2 * batch_size for k, v in val_loss.items()}
            print(f"Epoch {epoch}, Loss: {val_loss}, Total: {sum(val_loss.values())}")
            val_total = sum(val_loss.values())

            if val_total < best_loss and epoch > 0:
                best_loss = val_loss
                save_ckpt(val_loss, epoch)
            elif save_interval is not None and epoch % save_interval == 0:
                best_loss = min(val_loss, val_loss)
                save_ckpt(val_loss, epoch)
            else:
                print()
            #######################################################################################
            wandb.log({
                "val_loss": val_loss,
                "best_val_loss": best_loss,
                "epoch": epoch
            })

    ########################################################################################

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if ckpt_path and cplt_epoches != 0:
            print("Saving last checkpoint...")
            save_ckpt(val_loss, cplt_epoches)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Finished training")
        return train_id, train_losses, val_losses, cplt_epoches


if __name__ == "__main__":
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 初始化模型、损失函数和优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    image_path = Path(r'../tmp/datasets/record_20241231-16_59_52-_out')
    action_df = pl.read_csv(r'../tmp/datasets/record_20241231-16_59_52-_out/dataset.csv')
    # image_path = Path(r'../tmp/datasets/fake_dataset')
    # action_df = pl.read_csv(r'../tmp/datasets/fake_dataset/fake_dataset.csv')
    sequence_length = 10
    dataset = SequenceImageDataset(
        image_dir=image_path,
        action_df=action_df,
        sequence_length=sequence_length,
        transform=data_transform["train"]
    )
    model = MainModel(hidden_size=256)

    for name, param in model.named_parameters():
        print(name)
    exit()
    #
    # print("dataset shape:", dataset.__len__())
    # sample = dataset[0]
    # print("Image shape:", sample["image"].shape)
    # print("Actions:", sample["action"])
    # dataset, _ = random_split(dataset, [10, len(dataset) - 10])

    train_id, train_loss, val_loss, cplt_epoches = train(
        model, device, dataset,
        batch_size=2, val_percent=0.5, num_epochs=10,
        ckpt_path=Path("../ckpt"), save_state_dict=False, save_interval=None,
        use_amp=False, sequence_length=sequence_length
        )

    # summary(model, input_size=(2, 3, 720, 1280))


