import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

from .config import UNetConfig


def create_unet(cfg: UNetConfig | None = None) -> UNet2DModel:
    if cfg is None:
        cfg = UNetConfig()
    return UNet2DModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
    )


@torch.no_grad()
def generate_steps(
    model: UNet2DModel,
    mask: torch.Tensor,
    num_steps: int = 10,
    device: str = "cuda",
    crop_size: int = 256,
) -> list[torch.Tensor]:
    model.eval()
    x = torch.randn((1, 3, crop_size, crop_size), device=device)

    mask = mask.to(device)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 5:
        mask = mask.squeeze(0)

    history = [x.clone().cpu()]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.tensor([i * dt], device=device)
        model_input = torch.cat([x, mask], dim=1)
        velocity = model(model_input, t).sample
        x = x + velocity * dt
        history.append(x.clone().cpu())

    return history


def flow_matching_loss(
    model: UNet2DModel,
    clean_images: torch.Tensor,
    masks: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(clean_images)
    x_t = (1 - t) * noise + t * clean_images
    model_input = torch.cat([x_t, masks], dim=1)
    pred_velocity = model(model_input, t.flatten()).sample
    target = clean_images - noise
    return F.mse_loss(pred_velocity, target)
