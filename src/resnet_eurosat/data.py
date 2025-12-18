import torch
import polars as pl
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class EuroSATDataset(Dataset):
    def __init__(self, csv_path, image_root, filt_class=None, return_label=False):
        self.df = pl.read_csv(csv_path).drop(pl.read_csv(csv_path).columns[0])
        self.image_root = Path(image_root)
        self.return_label = return_label

        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        if filt_class is not None:
            self.df = self.df.filter(pl.col("Label")==filt_class)
            
    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        img_path = self.image_root / row["Filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.return_label:
            return image, row["Label"], row["ClassName"]

        return image