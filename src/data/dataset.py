from pathlib import Path

import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DeepGlobePolarsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        split: str = "train",
        crop_size: int = 256,
        subset_size: int | None = None,
        seed: int = 42,
        cache_metadata: bool = True,
    ):
        self.root_dir = root_dir
        self._cache = cache_metadata

        df = pl.read_csv(Path(root_dir) / csv_path)
        df = df.filter(pl.col("split") == split)

        if subset_size is not None and subset_size < len(df):
            df = df.sample(n=subset_size, seed=seed)

        self.data = df.to_dicts()

        self.transform = v2.Compose([
            v2.RandomCrop(size=(crop_size, crop_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]

        img_path = Path(self.root_dir) / row["sat_image_path"]
        mask_path = Path(self.root_dir) / row["mask_path"]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}
