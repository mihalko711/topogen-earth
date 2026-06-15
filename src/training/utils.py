import os

import matplotlib.pyplot as plt
import torch

from src.models.model import generate_steps


def visualize_and_save(
    model,
    masks,
    clean_images,
    epoch,
    step,
    save_dir,
    device,
    num_steps=20,
    crop_size=128,
):
    model.eval()
    with torch.no_grad():
        history = generate_steps(
            model, masks[0:1], num_steps=num_steps, device=device, crop_size=crop_size
        )
    model.train()

    mask_disp = (masks[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
    target_disp = (
        (clean_images[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
    )

    ncols = (len(history) + 2) // 4
    fig, axes = plt.subplots(5, ncols, figsize=(ncols * 2.5, 5 * 3))

    axes[0][0].imshow(mask_disp)
    axes[0][0].set_title("Input Mask")
    axes[0][0].axis("off")
    axes[0][1].imshow(target_disp)
    axes[0][1].set_title("Target Image")
    axes[0][1].axis("off")

    for idx, img in enumerate(history):
        display_img = (img.squeeze().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        r = (idx + 2) // ncols
        c = (idx + 2) % ncols
        axes[r][c].imshow(display_img)
        axes[r][c].set_title(f"t={idx / (len(history) - 1):.1f}")
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_step_{step}.png"))
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"--- Checkpoint saved: {ckpt_path} ---")
