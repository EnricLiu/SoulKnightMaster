from pathlib import Path
import time
import polars as pl
import torch
from torch import nn
from dataset import ImageDataset, ImageDataset_Combine
from torchinfo import summary
import os
import numpy as np
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.cuda.amp as amp

train_number = 0
train_label = {0: "move", 1: "angle", 2: "attack", 3: "skill", 4: "weapon"}


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.effi_net = models.efficientnet_v2_s(weights=weights)
        in_features = self.effi_net.classifier[1].in_features# 获取全连接层的输入特征数
        self.effi_net.classifier = nn.Identity()  # 去掉原来的全连接层，提取特征

        self.classify = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        features = self.effi_net(x)  # 提取特征

        # 二分类输出
        out_classify = self.classify(features)
        # out_binary_1 = torch.sigmoid(self.fc_binary_1(features))  # 第1项
        # out_binary_3 = torch.sigmoid(self.fc_binary_3(features))  # 第3项
        # out_binary_4 = torch.sigmoid(self.fc_binary_4(features))  # 第4项
        # out_binary_5 = torch.sigmoid(self.fc_binary_5(features))  # 第5项

        # # 回归输出（-4到4之间，使用 tanh 缩放值到范围内）
        # out_regression_2 = 4 * torch.tanh(self.fc_regression_2(features))  # 第2项
        pred_action = out_classify
        return pred_action


# 定义自定义损失函数
def custom_loss(outputs, targets):
    criteria_binary = nn.BCEWithLogitsLoss()  # 二分类任务使用二元交叉熵
    criteria_regression = nn.MSELoss()  # 回归任务使用均方误差损失

    # 分别计算损失
    # loss_binary_1 = criteria_binary(outputs[0], targets[:, 0].unsqueeze(1))  # 第1项
    # loss_regression_2 = criteria_regression(outputs[1], targets[:, 1].unsqueeze(1))  # 第2项
    # loss_binary_3 = criteria_binary(outputs[2], targets[:, 2].unsqueeze(1))  # 第3项
    # loss_binary_4 = criteria_binary(outputs[3], targets[:, 3].unsqueeze(1))  # 第4项
    # loss_binary_5 = criteria_binary(outputs[4], targets[:, 4].unsqueeze(1))  # 第5项
    #
    # # 加权求和损失
    # total_loss = (
    #         loss_binary_1 * 0.4 +
    #         loss_regression_2 * 0.4 +
    #         loss_binary_3 * 0.1 +
    #         loss_binary_4 * 0.09 +
    #         loss_binary_5 * 0.01
    # )
    # if targets.dim() == 1:
    #     targets = targets.unsqueeze(1)
    if train_number == 1:
        loss = criteria_regression(outputs, targets.unsqueeze(1))
    else:
        loss = criteria_binary(outputs, targets.unsqueeze(1))
    return loss


# 冻结部分模型
def freeze_model_weights(model, freeze_until_layer=15):
    print(f"model_features: {model.features}")
    for i in range(6):
        for name, param in model.features[i].named_parameters():
            param.requires_grad = False
    freeze_until_layer = max(0, min(freeze_until_layer, 15))
    for j in range(freeze_until_layer):
        for name, param in model.features[6][j].named_parameters():
            param.requires_grad = False


# 训练过程
def train(model, device, dataset, optimizer,criterion,
          ckpt_path: Path, save_state_dict: bool,
          val_percent=0.2, split_seed=0, batch_size=0, save_interval: int = None, use_amp=False, num_epochs=10, ):

    def save_ckpt(_loss, _epoch):
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if ckpt_path is None: return
        target = model

        if save_state_dict: target = model.state_dict()
        torch.save(target, os.path.join(ckpt_path, f'{train_id}-ln={train_label[train_number]}-loss={_loss:.3f}-e{_epoch}.pth'))
        print(f'Checkpoint {_epoch} saved!')

    train_id = round(time.time())

    model.to(device)
    model.train()


    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed))
    loader_args = dict(batch_size=batch_size, num_workers=0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    best_loss = float("inf")
    cplt_epoches = 0
    train_losses, val_losses = [], []

    scaler = amp.GradScaler(enabled=use_amp)
    layer_num = 15
    try:
        for epoch in range(num_epochs):
            if epoch // 20 == 0 and epoch != 0:
                freeze_model_weights(model.effi_net, freeze_until_layer=layer_num)
                layer_num -= 1
                print(f"freeze layer {layer_num}")
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
                train_loss = 0
                for images in train_loader:
                    image = images["image"].to(device)
                    action = images["action"][:, train_number].to(device)
                    optimizer.zero_grad()
                    with amp.autocast(enabled=use_amp):
                        output = model(image)
                        loss = criterion(output, action)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                    pbar.update(batch_size)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

            train_loss /= n_train
            train_loss = train_loss*batch_size

            print(f"Epoch {epoch}, Total Loss: {train_loss}")


            model.eval()
            val_loss = 0
            with torch.no_grad():
                all_y_true = []  # 存储所有批次的真实标签
                all_y_pred = []  # 存储所有批次的预测结果
                for i, images in tqdm(enumerate(val_loader)):
                    image = images["image"].to(device)
                    action = images["action"][:, train_number].to(device)


                    output = model(image)
                    # output_concat = torch.cat(output, dim=1)
                    # print(action.shape)
                    # print(action.type())
                    # for i, elem in enumerate(action):
                    #     print(f"action[{i}]: type = {type(elem)}, content = {elem}")
                    # for i, elem in enumerate(output_concat):
                    #     print(f"output_concat[{i}]: type = {type(elem)}, content = {elem}")
                    loss = criterion(output, action)
                    val_loss += loss.item()
                    #
                    # action_cpu = [t.cpu() for t in action]
                    # action_numpy = [t.numpy() for t in action_cpu]
                    # all_y_true.extend(action_numpy)
                    # output_cpu = [t.cpu() for t in output_concat]
                    # output_numpy = [t.numpy() for t in output_cpu]
                    # all_y_pred.extend(output_numpy)


            val_loss /= len(val_set)
            val_loss = val_loss*batch_size
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")


            if val_loss < best_loss and epoch > 0:
                best_loss = val_loss
                save_ckpt(val_loss, epoch)
            elif save_interval is not None and epoch % save_interval == 0:
                best_loss = min(val_loss, val_loss)
                save_ckpt(val_loss, epoch)
            else:
                print()


    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if ckpt_path and cplt_epoches != 0:
            print("Saving last checkpoint...")
            save_ckpt(val_loss, cplt_epoches)

    except Exception as e:
        print(f"Error: {e}")
        raise e

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
    # device = "cpu"
    print("device:", device)
    image_path = Path(r'./datasets/merge/merged_20250227-00_37_48-_out')
    action_df = pl.read_csv(r'./datasets/merge/merged_20250227-00_37_48-_out/dataset.csv')
    # model = MainModel()
    model = torch.load("../ckpt/1740588758-ln=move-loss=0.346-e5.pth")
    criterion = custom_loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    dataset = ImageDataset(image_dir=image_path, action_df=action_df, transform=data_transform["train"])
    # print("dataset shape:", dataset.__len__())
    # sample = dataset[0]
    # print("Image shape:", sample["image"].shape)
    # print("Actions:", sample["action"])
    # dataset, _ = random_split(dataset, [10, len(dataset) - 10])

    train_id, train_loss, val_loss, cplt_epoches = train(
            model, device, dataset, optimizer,criterion=criterion,
            batch_size=32, val_percent=0.2, num_epochs=100,
            ckpt_path=Path("../ckpt"), save_state_dict=False, save_interval=None, use_amp=False
        )

    # summary(model, input_size=(2, 3, 720, 1280))
