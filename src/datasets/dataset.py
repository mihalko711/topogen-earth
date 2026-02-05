import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import polars as pl
import torchvision.transforms as T


class EuroSATDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None, filt_class=None, return_label=False):
        self.df = pl.read_csv(csv_path).drop(pl.read_csv(csv_path).columns[0])
        self.image_root = Path(image_root)
        self.transform = transform  # ← внешний трансформ
        self.return_label = return_label

        if filt_class is not None:
            self.df = self.df.filter(pl.col("Label") == filt_class)
            
    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        img_path = self.image_root / row["Filename"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)  # ← применяем внешний трансформ

        if self.return_label:
            return image, row["Label"], row["ClassName"]

        return image


def get_train_transform():
    """Get training transforms with augmentation"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomChoice([
            T.Lambda(lambda x: x),                          # 0°
            T.Lambda(lambda x: x.rotate(90, expand=False)),   # 90°
            T.Lambda(lambda x: x.rotate(180, expand=False)),  # 180°
            T.Lambda(lambda x: x.rotate(270, expand=False)),  # 270°
        ]),
        T.ColorJitter(brightness=0.05, contrast=0.05),
        T.RandomApply([T.GaussianBlur(3, (0.1, 0.3))], p=0.2),
        T.Resize((64, 64)),  # Для надёжности (изображения EuroSAT уже 64×64)
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1, 1]
    ])


def get_val_transform():
    """Get validation transforms without augmentation"""
    return T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])