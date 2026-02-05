import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# --- Scheduler ---
class DiffusionScheduler(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        super().__init__()
        self.timesteps = timesteps
        if beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod).float())

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def add_noise(self, x_0, eps, t):
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps

# --- Lightweight U-Net ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.SiLU()
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.shortcut(x))

class LightConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.start = nn.Conv2d(in_channels + cond_channels, 64, 3, padding=1)

        self.res1 = ResidualBlock(64, 128, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128, 256, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(256, 256, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.res3 = ResidualBlock(256 + 256, 128, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.res4 = ResidualBlock(128 + 128, 64, time_dim)

        self.final = nn.Conv2d(64, in_channels, 1)

    def forward(self, r_t, t, condition):
        t = t.float() / 1000.0   # нормализация времени
        t_emb = self.time_mlp(t.unsqueeze(-1))

        x = self.start(torch.cat([r_t, condition], dim=1))

        s1 = self.res1(x, t_emb)
        x = self.pool1(s1)
        s2 = self.res2(x, t_emb)
        x = self.pool2(s2)

        x = self.bottleneck(x, t_emb)

        x = self.up2(x)
        x = self.res3(torch.cat([x, s2], dim=1), t_emb)
        x = self.up1(x)
        x = self.res4(torch.cat([x, s1], dim=1), t_emb)

        return self.final(x)

# --- Wrapper Class ---
class ConditionalDiffusionVAE:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, x_hat, timesteps=None):
        self.model.eval()
        B, C, H, W = x_hat.shape
        r_t = torch.randn((B, C, H, W), device=self.device)

        total_steps = timesteps if timesteps else self.scheduler.timesteps

        for t in reversed(range(total_steps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(r_t, t_batch, x_hat)

            alpha_t = self.scheduler.alphas[t].view(-1,1,1,1)
            alpha_bar_t = self.scheduler.alphas_cumprod[t].view(-1,1,1,1)
            sqrt_one_minus_bar = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)

            mean = (1 / torch.sqrt(alpha_t)) * (
                r_t - ((1 - alpha_t) / sqrt_one_minus_bar) * eps_pred
            )

            if t > 0:
                alpha_bar_prev = self.scheduler.alphas_cumprod[t-1].view(-1,1,1,1)
                beta_t = self.scheduler.betas[t].view(-1,1,1,1)
                sigma = torch.sqrt((1 - alpha_bar_prev)/(1 - alpha_bar_t) * beta_t)
                r_t = mean + sigma * torch.randn_like(r_t)
            else:
                r_t = mean

        return r_t, torch.clamp(x_hat + r_t, -1, 1)

# --- Factory Functions ---
def create_diffusion_vae(som_vae_model, device='cuda', timesteps=1000):
    scheduler = DiffusionScheduler(timesteps=timesteps).to(device)
    model = LightConditionalUNet(time_dim=256).to(device)
    return ConditionalDiffusionVAE(model, scheduler, device)

def train_diffusion_vae(som_vae_model, train_dataset, val_dataset,
                        epochs=10, batch_size=32, lr=2e-4,
                        device='cuda', timesteps=1000):

    diffusion_vae = create_diffusion_vae(som_vae_model, device, timesteps)
    optimizer = torch.optim.Adam(diffusion_vae.model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        diffusion_vae.model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for x_0, _, _ in pbar:
            x_0 = x_0.to(device)

            with torch.no_grad():
                x_hat, _, _, _, _, _ = som_vae_model(x_0)

            residual = x_0 - x_hat
            t = torch.randint(0, timesteps, (x_0.shape[0],), device=device)

            eps = torch.randn_like(residual)
            r_t = diffusion_vae.scheduler.add_noise(residual, eps, t)

            eps_pred = diffusion_vae.model(r_t, t, x_hat)
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    return diffusion_vae, None
