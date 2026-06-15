import os

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.config import TrainingConfig
from src.models.model import create_unet, flow_matching_loss
from src.training.utils import save_checkpoint, visualize_and_save


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        cfg: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        self.scaler = GradScaler(self.device)
        os.makedirs(cfg.save_dir, exist_ok=True)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        pbar = tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc=f"Epoch {epoch}",
        )

        for i, batch in pbar:
            clean_images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            batch_size = clean_images.shape[0]

            t = torch.rand((batch_size, 1, 1, 1), device=self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = flow_matching_loss(self.model, clean_images, masks, t)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            pbar.set_postfix({"loss": loss.item()})

            if i % self.cfg.viz_interval == 0:
                visualize_and_save(
                    self.model,
                    masks,
                    clean_images,
                    epoch,
                    i,
                    self.cfg.save_dir,
                    self.device,
                    num_steps=self.cfg.num_steps_generation,
                    crop_size=self.cfg.crop_size,
                )

    def run(self):
        for epoch in range(self.cfg.num_epochs):
            self.train_one_epoch(epoch)
            save_checkpoint(self.model, self.optimizer, epoch, self.cfg.save_dir)
