import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import polars as pl
from pathlib import Path
import numpy as np
from torchvision.transforms import Compose

class ImageDataset(Dataset):
    def __init__(self, image_dir: Path, action_df: pl.DataFrame, transform: Compose = None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = action_df
        self.image_paths = [img for img in self.image_dir.iterdir() if img.suffix in ['.png', '.jpg', '.jpeg', '.npy']]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        match img_path.suffix:
            case '.npy':
                image = np.load(img_path)
                # if self.transform is not None:
                #     image = self.transform.ToTensor()(image)
                # else:
                #     image = torch.tensor(image)
            case '.png', '.jpg', '.jpeg':
                image = Image.open(img_path).convert('RGB')
                # if self.transform is not None:
                #     image = self.transform.ToTensor()(image)
                # else:
                #     image = torch.tensor(image)
            case _:
                raise NotImplemented

        if self.transform is not None:
            if img_path.suffix == '.npy': image = Image.fromarray(image)
            image = self.transform(image)
        # image = torch.tensor(np.asarray(image)[np.newaxis, :], dtype=torch.float32).transpose(3, 1)
        image = torch.tensor(image, dtype=torch.float32)
        action = np.zeros(5, dtype=np.float32)
        col_map = ["move", "angle", "attack", "skill", "weapon"]
        row = self.df.filter(pl.col("img_dir") == img_path.name).row(0, named=True)

        if not row: raise ValueError(f"Image {img_path} not found in action_df")

        for k, v in row.items():
            if v is None: continue
            match k:
                case "angle":
                    # action[col_map.index(k)] = float(v) / 360 + 0.5
                    action[col_map.index(k)] = float(v)/3.141592653589793 + 0.25
                case x if x in col_map:
                    action[col_map.index(k)] = 1.0 if v else 0.0

        return {"image": image, "action": action}
class ImageDataset_Combine(Dataset):
    def __init__(self, image_dir: Path, action_df: pl.DataFrame, transform: Compose = None,flow_dir: Path = None):
        self.image_dir = image_dir
        self.flow_dir = flow_dir
        self.transform = transform
        self.df = action_df
        self.image_paths = [img for img in self.image_dir.iterdir() if img.suffix in ['.png', '.jpg', '.jpeg', '.npy']]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        match img_path.suffix:
            case '.npy':
                image = np.load(img_path)
                # if self.transform is not None:
                #     image = self.transform.ToTensor()(image)
                # else:
                #     image = torch.tensor(image)
            case '.png', '.jpg', '.jpeg':
                image = Image.open(img_path).convert('RGB')
                # if self.transform is not None:
                #     image = self.transform.ToTensor()(image)
                # else:
                #     image = torch.tensor(image)
            case _:
                raise NotImplemented


        flow_idx = int(img_path.stem) - 1
        flow_path = self.flow_dir / f'{flow_idx}.npy'
        flow = np.load(flow_path)

        if self.transform is not None:
            if img_path.suffix == '.npy': image = Image.fromarray(image)
            image = self.transform(image)
        # image = torch.tensor(np.asarray(image)[np.newaxis, :], dtype=torch.float32).transpose(3, 1)
        combined = np.concatenate((image, flow), axis=-1)
        image = torch.tensor(combined, dtype=torch.float32)

        action = np.zeros(5, dtype=np.float32)
        col_map = ["move", "angle", "attack", "skill", "weapon"]
        row = self.df.filter(pl.col("img_dir") == img_path.name).row(0, named=True)

        if not row: raise ValueError(f"Image {img_path} not found in action_df")

        for k, v in row.items():
            if v is None: continue
            match k:
                case "angle":
                    # action[col_map.index(k)] = float(v) / 360 + 0.5
                    action[col_map.index(k)] = float(v)/3.141592653589793 + 0.25
                case x if x in col_map:
                    action[col_map.index(k)] = 1.0 if v else 0.0

        return {"image": image, "action": action}


# Example usage
if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.Resize((720, 1280)),
    #     transforms.ToTensor()
    # ])
    transform = None
    df = pl.read_csv(r'./tmp\datasets\record_20241231-16_59_52-_out\dataset.csv')
    dataset = ImageDataset_Combine(image_dir=Path(r'./tmp\datasets\record_20241231-16_59_52-_out'), action_df=df,
                           transform=transform, flow_dir=Path(r'E:\大学资料\大三上资料\竞赛\软件创新大赛\初赛\SoulKnightMaster\SoulKnightMaster\model\feat_ext\pre_train_resnet\output'))
    sample = dataset[0]
    print(sample["image"].shape)
    print(sample["action"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # for images in dataloader:
    for i in range(len(dataset)):
        image = dataset.__getitem__(i)
        print(image["image"])
        print(image["action"])
        exit()
