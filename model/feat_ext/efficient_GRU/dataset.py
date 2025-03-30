from pathlib import Path
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class SequenceImageDataset(Dataset):
    def __init__(self, image_dir: Path, action_df: pl.DataFrame, sequence_length: int,
                 transform: transforms.Compose = None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = action_df
        self.sequence_length = sequence_length

        self.image_paths = sorted(
            self.image_dir.glob("*.npy"),
            key=lambda x: int(x.stem)
        )
        self.cached_images = []
        # Sort dataframe to match image order
        self.df = self.df.with_columns(pl.col("img_dir").str.extract(r"(\d+)").cast(pl.Int32).alias("img_num"))
        self.df = self.df.sort("img_num")
        self.df = self.df.fill_null(0.0)
        self.df = self.df.with_columns(pl.col("angle") / 3.141592653589793 + 0.25)
        self.actions = self.df.select(["move", "angle", "attack", "skill", "weapon"]).to_numpy()
        # print(self.actions)
        self.col_map = ["move", "angle", "attack", "skill", "weapon"]
        self.cached_images = [None] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths) - self.sequence_length + 1

    def __getitem__(self, idx):
        images = []
        for i in range(idx, idx + self.sequence_length):
            if self.cached_images[i] is None:
                img_path = self.image_paths[i]
                if img_path.suffix == '.npy':
                    # print(f"load npy:{img_path}")
                    image = np.load(img_path)
                    image = Image.fromarray(image)
                else:
                    image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = np.array(image, dtype=np.uint8)
                    image = torch.tensor(image,dtype=torch.float32)  # 指定数据类型为 torch.float32
            else:
                image = self.cached_images[i]
            images.append(image)
        images = torch.stack(images, dim=0)  # [sequence_length, C, H, W]

        raw_actions = self.actions[idx:idx + self.sequence_length]  # [sequence_length, 5]
        actions = np.zeros_like(raw_actions, dtype=np.float32)

        # 对每一行动作数据进行处理
        for t in range(self.sequence_length):
            row = dict(zip(self.col_map, raw_actions[t]))
            for k, v in row.items():
                if v is None:
                    continue
                match k:
                    case "angle":
                        actions[t, self.col_map.index(k)] = float(v)
                    case x if x in self.col_map:
                        actions[t, self.col_map.index(k)] = 1.0 if v else 0.0

        return {"image": images, "action": actions}



if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.Resize((720, 1280)),
    #     transforms.ToTensor()
    # ])
    transform = None

    # df = pl.read_csv(r'../tmp\datasets\record_20241231-16_59_52-_out\dataset.csv')
    # dataset = SequenceImageDataset(image_dir=Path(r'../tmp\datasets\record_20241231-16_59_52-_out'), action_df=df,
    #                                transform=transform, sequence_length=10)
    
    df = pl.read_csv(r'../pre_train_resnet/datasets/merge/merged_20250226-14_00_28-_out/dataset.csv')
    dataset = SequenceImageDataset(image_dir=Path(r'../pre_train_resnet/datasets/merge/merged_20250226-14_00_28-_out'), action_df=df,
                                   transform=transform, sequence_length=10)
    
    #
    # sample = dataset[0][0]
    # print(sample["image"].shape)
    # print(sample["action"].shape)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # for images in dataloader:
    print("dataset len: ", len(dataset))
    for i in range(200):
        image = dataset.__getitem__(i)
        print(image["action"])



