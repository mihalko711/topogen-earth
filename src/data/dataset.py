from pathlib import Path

import numpy as np
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


class OpenEarthMapDataset(Dataset):
    CLASS_RGB = torch.tensor([
        [0, 0, 0],       # 0 unknown
        [128, 0, 0],     # 1 Bareland
        [0, 255, 36],    # 2 Grass
        [148, 148, 148], # 3 Pavement
        [255, 255, 255], # 4 Road
        [34, 97, 38],    # 5 Tree
        [0, 69, 255],    # 6 Water
        [75, 181, 73],   # 7 Cropland
        [222, 31, 7],    # 8 buildings
    ], dtype=torch.uint8)

    def __init__(
        self,
        root_dir: str = "",
        split: str = "train",
        crop_size: int = 256,
        subset_size: int | None = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        image_dir = self.root_dir / "images" / split
        self.image_paths = sorted([p for p in image_dir.glob("*.tif")])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No .tif files found in {image_dir}. "
                f"Check that root_dir='{root_dir}' and split='{split}' are correct."
            )

        if subset_size is not None and subset_size < len(self.image_paths):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.image_paths), size=subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(idx)]

        self.transform = v2.Compose([
            v2.RandomCrop(size=(crop_size, crop_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = Path(str(img_path).replace("/images/", "/label/"))

        image = Image.open(img_path).convert("RGB")

        mask_arr = np.array(Image.open(mask_path), dtype=np.int64)
        mask_rgb = self.CLASS_RGB[mask_arr].numpy()
        mask = Image.fromarray(mask_rgb, "RGB")

        image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}
