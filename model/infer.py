import torch
import time
import numpy as np
import polars as pl
from pathlib import Path
from dataset_test import ImageDataset
from resnet import ResNet, resnet101

def inference(model, device, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data.to(device))
    return output

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = Path("./ckpt/1735575662-loss=0.039-e1.pth")
    model = resnet101(num_classes=5).to("cuda")
    model.load_state_dict(torch.load(CKPT_PATH))
    # model = torch.load(CKPT_PATH)
    
    image_path = Path('../tools/replay_recorder/datasets/record_20241223-18_48_58-_out')
    action_df = pl.read_csv('../tools/replay_recorder/datasets/record_20241223-18_48_58-_out/dataset.csv')
    print("loading dataset...")
    dataset = ImageDataset(image_dir=image_path, action_df=action_df, transform=None)
    print("finish loading!")
    

    start = time.perf_counter()
    for i in range(50):
        idx = np.random.randint(0, len(dataset))
        test_data: list[str, torch.Tensor] = dataset.__getitem__(idx)
        true_action = test_data["action"]
        result = inference(model, DEVICE, test_data["image"].unsqueeze(0))
    stop = time.perf_counter()
    
    print("inference time:", stop - start)
    # print("true action:", true_action)
    # print("predicted action:", result)