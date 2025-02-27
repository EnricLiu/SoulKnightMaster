import time
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from dataset import ImageDataset
from single_train import MainModel, custom_loss

def inference(model: MainModel, device:torch.device, input_data:torch.Tensor) -> torch.Tensor|list[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        output = model(input_data.to(device))
    return output


def dataset_infer(model:MainModel, device:torch.device, dataset:ImageDataset, interval:range=None, bs:int=48) -> np.ndarray:
    model.eval()
    def load_data(r:range=None):
        images, true_actions = [], []
        _bs = bs
        if r is None: r = range(len(dataset))
        for idx in r:
            test_data: list[str, torch.Tensor] = dataset.__getitem__(idx)
            true_actions.append(test_data["action"])
            images.append(test_data["image"])
            _bs -= 1
            if _bs == 0:
                _bs = bs
                yield torch.from_numpy(np.asarray(images)), np.asarray(true_actions)
    
    preds = []
    actions = []
    for images, true_action in tqdm((load_data(interval))):
        output = inference(model, device, images)
        if type(output) == list:
            output = np.asarray(list(map(lambda x: x.cpu().numpy().flatten(), output)))
            print(output.shape)
            if len(preds) == 0: preds = output
            else: preds = np.concatenate([preds, output], axis=1)
        else:
            preds.extend(output.cpu().numpy().flatten())
            
        np.asarray(actions.append(true_action))
    return preds, actions
        


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = Path("../ckpt/1740501980-loss=0.189-e11.pth")
    # model = MainModel().to("cuda")
    # model.load_state_dict(torch.load(CKPT_PATH))
    model = torch.load(CKPT_PATH)

    image_path = Path('./datasets/origin/record_20241231-16_59_52-_out')
    action_df = pl.read_csv('./datasets/origin/record_20241231-16_59_52-_out/dataset.csv')
    print("loading dataset...")
    dataset = ImageDataset(
        image_dir = image_path,
        action_df = action_df,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    )
    print("finish loading!")
    # for i in range(50):
    # idx = np.random.randint(0, len(dataset))
    # test_data: list[str, torch.Tensor] = dataset.__getitem__(idx)
    # true_action = test_data["action"]
    # start = time.perf_counter()
    # result = inference(model, DEVICE, test_data["image"].unsqueeze(0))
    # true_action = torch.tensor([[torch.tensor([[x]], dtype=torch.float32) for x in true_action]]).to(DEVICE)
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    y_true = []  # 示例数据
    y_pred = []

    for i in tqdm(range(len(dataset))):
        test_data: list[str, torch.Tensor] = dataset.__getitem__(i)
        true_action = test_data["action"]
        result = inference(model, DEVICE, test_data["image"].unsqueeze(0))
        # print(true_action)
        # print(result)
        # print(int(true_action[0]) == int(result[0].cpu().numpy()>0.5))
        # if int(true_action[0]) == int(result[0].cpu().numpy()>0.5):
        #     cnt+=1
        if int(true_action[2]) == int(result[0].cpu().numpy()>0.5):
            cnt1+=1
        # if int(true_action[3]) == int(result[3].cpu().numpy()>0.5):
        #     cnt2+=1
        # if int(true_action[4]) == int(result[4].cpu().numpy()>0.5):
        #     cnt3+=1
        # if int(true_action[0]) == int(result[0].cpu().numpy()>0.5):
        #     cnt+=1
        # 假设你已有真实标签 y_true、预测标签 y_pred 以及标签名称列表 class_names

        y_true.append(int(true_action[2]))
        y_pred.append(int(result[0].cpu().numpy()>0.5))
    class_names = ["False", "True"]
    # 使用 wandb.sklearn 记录混淆矩阵


    def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
        """
        绘制混淆矩阵
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            classes: 类别名称列表
            save_path: 保存图片的路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)

        # 设置标题和标签
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=f'./pictures/{CKPT_PATH.stem}.png')


    print(f"cnt1:{cnt1/len(dataset)}")
    # print(f"cnt1:{cnt1/len(dataset)}")
    # print(f"cnt2:{cnt2/len(dataset)}")
    # print(f"cnt3:{cnt3/len(dataset)}")
    # print(true_action)
    # print(result)
    # custom_loss(result, true_action.to(DEVICE))
    # stop = time.perf_counter()

    # print("inference time:", stop - start)
    # print("true action:", true_action)
    # print("predicted action:", result)
    # print(f"loss: {custom_loss(result, true_action)}")
