import torch
import time
import numpy as np
import polars as pl
from pathlib import Path
from torchvision import transforms
from old import MainModel, custom_loss
from SoulKnightMaster.model.feat_ext.dataset_test import ImageDataset
from tqdm import tqdm
import wandb
import wandb.sklearn

# 初始化 wandb 运行（替换为你的项目名称）
wandb.login(key="")
wandb.init(project="your_project_name")





def inference(model, device, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data.to(device))
    return output


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = Path("../ckpt/1740456584-loss=0.173-e6.pth")
    # model = MainModel().to("cuda")
    # model.load_state_dict(torch.load(CKPT_PATH))
    model = torch.load(CKPT_PATH)

    image_path = Path('../tmp/datasets/record_20241231-16_59_52-_out')
    action_df = pl.read_csv('../tmp/datasets/record_20241231-16_59_52-_out/dataset.csv')
    print("loading dataset...")
    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        }
    dataset = ImageDataset(image_dir=image_path, action_df=action_df, transform=data_transform["train"])
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
        if int(true_action[0]) == int(result[0].cpu().numpy()>0.5):
            cnt+=1
        # if int(true_action[2]) == int(result[2].cpu().numpy()>0.5):
        #     cnt1+=1
        # if int(true_action[3]) == int(result[3].cpu().numpy()>0.5):
        #     cnt2+=1
        # if int(true_action[4]) == int(result[4].cpu().numpy()>0.5):
        #     cnt3+=1
        # if int(true_action[0]) == int(result[0].cpu().numpy()>0.5):
        #     cnt+=1
        # 假设你已有真实标签 y_true、预测标签 y_pred 以及标签名称列表 class_names

        y_true.append(int(true_action[0]))
        y_pred.append(int(result[0].cpu().numpy()>0.5))
    class_names = ["False", "True"]
    # 使用 wandb.sklearn 记录混淆矩阵
    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels=class_names)
    wandb.log({"accuracy": cnt/len(dataset)})
    print(f"cnt:{cnt/len(dataset)}")
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
