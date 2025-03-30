import torch
import time
import numpy as np
import polars as pl
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from efficient_GRU import MainModel, SequenceImageDataset

from torch.utils.data import DataLoader, Subset
sequence_length = 10
batch_size = 1

def buffered_loader(_loader: DataLoader, _buffer_set: Subset):
    if not hasattr(buffered_loader, "data_buffer"):
        buffered_loader.data_buffer = {
            "image": torch.zeros((sequence_length + batch_size - 1, 3, 720, 1280)),
            "action": torch.zeros((sequence_length + batch_size - 1, 5)),
        }
        # 初始填充（仅执行一次）
        for i in range(sequence_length - 1):
            buffered_loader.data_buffer["image"][i] = _buffer_set[i]["image"]
            buffered_loader.data_buffer["action"][i] = torch.from_numpy(_buffer_set[i]["action"])

    # print("🤣")

    # print("🐮")

    for data in _loader:
        bs = data["image"].shape[0]
        # print("😘")
        for i in range(bs):
            buffered_loader.data_buffer["image"][sequence_length - 1 + i] = data["image"][i]
            buffered_loader.data_buffer["action"][sequence_length - 1 + i] = data["action"][i]
        # print(f"buffered_loader.data_buffer['action']{buffered_loader.data_buffer['action']}")
        _image = torch.stack(
            [buffered_loader.data_buffer["image"][idx: idx + sequence_length] for idx in range(bs)])
        _action = torch.stack(
            [buffered_loader.data_buffer["action"][idx: idx + sequence_length] for idx in range(bs)])
        buffered_loader.data_buffer["image"] = torch.cat(
            (buffered_loader.data_buffer["image"][-(sequence_length - 1):],
             torch.zeros((batch_size, 3, 720, 1280))))
        buffered_loader.data_buffer["action"] = torch.cat(
            (buffered_loader.data_buffer["action"][-(sequence_length - 1):],
             torch.zeros((batch_size, 5))))
        # print(buffered_loader.data_buffer["action"])
        yield {"image": _image,
               "action": _action,
               }
def inference(model, device, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data.to(device))
    return output

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CKPT_PATH_COMBINE = Path("../ckpt/1741313562-ln=skill-loss=6.551-e5.pth")
    model_combine = torch.load(CKPT_PATH_COMBINE, map_location=DEVICE).to(DEVICE)
    model_combine.eval()
    image_path = Path('../tmp/datasets/record_20241231-16_59_52-_out/sampled_data')
    action_df = pl.read_csv('../tmp/datasets/record_20241231-16_59_52-_out/sampled_data/dataset.csv')
    print("loading dataset...")
    data_transform = {
        "train": transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        }
    dataset = SequenceImageDataset(image_dir=image_path, action_df=action_df, transform=data_transform["train"], sequence_length=10)
    print("finish loading!")
    for i in range(10):
        idx = np.random.randint(0, len(dataset))
        test_data: list[str, torch.Tensor] = dataset.__getitem__(idx)
        image = test_data["image"].unsqueeze(0)
        true_action = test_data["action"]
        start = time.perf_counter()
        result = inference(model_combine, DEVICE, image)
        print(f'true_action: {true_action[9, :]}')
        print(f'result: {result[:, 9, :]}')
    # print(f'true_action: {true_action}')
    # print(f'result: {result}')
        # exit()



    stop = time.perf_counter()

    print("inference time:", stop - start)

