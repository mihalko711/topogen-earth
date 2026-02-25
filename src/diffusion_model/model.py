import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Positional Embedding for Time ---
def get_time_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Create sinusoidal time embeddings following the approach from Attention Is All You Need
    """
    half_dim = embedding_dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


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


# --- Lightweight U-Net for SOM-VAE Refinement ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_c)
        )
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_c)
        self.silu = nn.SiLU()
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.silu(self.bn1(self.conv1(x)))
        time_emb_out = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb_out
        h = self.bn2(self.conv2(h))
        return self.silu(h + self.shortcut(x))


class SOMVAERefinerUNet(nn.Module):
    """U-Net for refining SOM-VAE reconstructions using DiffuseVAE Formulation-1:
    - Input: noisy original image x_t (from x_0) + condition (x̂_q reconstruction)
    - Output: predicted noise ε for recovering x_0
    """
    def __init__(self, in_channels=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ← КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: вход = 2 * in_channels (x_t + condition)
        self.start = nn.Conv2d(in_channels * 2, 64, 3, padding=1)

        self.res1 = ResidualBlock(64, 128, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = ResidualBlock(128, 256, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(256, 256, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.res3 = ResidualBlock(256 + 256, 128, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.res4 = ResidualBlock(128 + 128, 64, time_dim)

        # Output: noise prediction
        self.final = nn.Conv2d(64, in_channels, 1)

    def forward(self, x_noisy, t, condition=None):
        """Forward pass for conditional denoising
        
        Args:
            x_noisy: Noisy version of ORIGINAL image x_t [B, C, H, W]
            t: Timestep [B]
            condition: SOM-VAE reconstruction x̂_q [B, C, H, W] (guidance signal)
        """
        if condition is None:
            raise ValueError("Condition (x̂_q reconstruction) is required for DiffuseVAE Formulation-1!")
        
        # ← КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: конкатенация шумного входа и условия
        x = torch.cat([x_noisy, condition], dim=1)  # [B, 2*C, H, W]
        
        # Time embedding
        t_emb = get_time_embedding(t, self.time_dim)
        t_emb = self.time_embed(t_emb)

        x = self.start(x)

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


# --- Diffusion Refiner for SOM-VAE (DiffuseVAE Formulation-1) ---
class SOMVAEDiffusionRefiner:
    """Diffusion refiner following DiffuseVAE Formulation-1 (Section 3.4.1):
    - Forward process: q(x₁:ₜ | x₀) — noise ORIGINAL x₀ directly
    - Reverse process: p(x₀:ₜ | x̂₀) — condition on VAE reconstruction x̂₀
    - Target: recover x₀ from noisy x_t given guidance x̂₀
    """
    
    def __init__(self, diffusion_model, scheduler, device):
        self.model = diffusion_model
        self.scheduler = scheduler
        self.device = device

    def train_step(self, som_vae_model, x_0, optimizer):
        """Single training step using DiffuseVAE Formulation-1
        
        From paper Section 3.4.1:
        "q(x₁:ₜ | z, x₀) ≈ q(x₁:ₜ | x₀)"  ← noise x₀ directly
        "p(x₀:ₜ | z) ≈ p(x₀:ₜ | x̂₀)"      ← condition reverse process on x̂₀
        
        Args:
            som_vae_model: Pretrained frozen SOM-VAE model
            x_0: Original input [B, C, H, W] (NOT reconstruction!)
            optimizer: Optimizer for diffusion model parameters
        """
        self.model.train()
        
        # 1. Get SOM-VAE reconstruction as CONDITION (guidance signal)
        with torch.no_grad():
            _, x_hat_q, _, _, _, _ = som_vae_model(x_0)
            x_hat_q = x_hat_q.detach()  # [B, C, H, W]

        # 2. ← КРИТИЧЕСКИ ВАЖНО: зашумляем ОРИГИНАЛ x_0 (Formulation-1 из основного текста статьи!)
        t = torch.randint(0, self.scheduler.timesteps, (x_0.shape[0],), device=self.device)
        eps = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, eps, t)  # x_t = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε

        # 3. Модель получает:
        #    - зашумлённый оригинал x_t
        #    - timestep t
        #    - условие x̂_q (реконструкция как "подсказка" для обратного процесса)
        eps_pred = self.model(x_t, t, condition=x_hat_q)

        # 4. Лосс: предсказываем шум для восстановления ОРИГИНАЛА x_0
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample_ddim(self, som_vae_model, x_0, num_inference_steps=50, eta=0.0, start_from_t=None):
        """DDIM sampling for Formulation-1:
        Start from noisy version of x̂_q, denoise to x_0 conditioned on x̂_q
        
        Note: Although we start from noisy x̂_q at inference, the model was trained
        to recover x_0 (not x̂_q) given condition x̂_q — this is the key insight of DiffuseVAE.
        """
        self.model.eval()
        
        # Get SOM-VAE reconstruction as condition
        with torch.no_grad():
            _, x_hat_q, _, _, _, _ = som_vae_model(x_0)
            x_hat_q = x_hat_q.detach()  # [B, C, H, W]

        B, C, H, W = x_hat_q.shape
        
        # ← СТАРТУЕМ С ЗАШУМЛЁННОЙ РЕКОНСТРУКЦИИ (практичный выбор для рефининга)
        if start_from_t is None:
            start_from_t = self.scheduler.timesteps - 1
        
        t_start = torch.full((B,), start_from_t, device=self.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_hat_q, torch.randn_like(x_hat_q), t_start)
        
        # DDIM timesteps
        timesteps = torch.linspace(start_from_t, 0, num_inference_steps, device=self.device).long()
        
        # DDIM sampling loop (using alphas_cumprod — critical!)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            # ← ПЕРЕДАЁМ УСЛОВИЕ (реконструкцию) в модель!
            eps_pred = self.model(x_t, t_batch, condition=x_hat_q)
            
            # Correct DDIM formulas using alphas_cumprod
            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            alpha_bar_t_next = self.scheduler.alphas_cumprod[t_next]
            
            # Predict x_0 from current noisy state
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / (torch.sqrt(alpha_bar_t) + 1e-8)
            
            # DDIM update (deterministic or stochastic based on eta)
            if eta > 0:
                # Stochastic mode (adds noise for diversity)
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next)
                )
                noise = torch.randn_like(x_t)
                x_t = (
                    torch.sqrt(alpha_bar_t_next) * pred_x0 +
                    torch.sqrt(1 - alpha_bar_t_next - sigma ** 2) * eps_pred +
                    sigma * noise
                )
            else:
                # Deterministic mode (recommended for refinement)
                x_t = (
                    torch.sqrt(alpha_bar_t_next) * pred_x0 +
                    torch.sqrt(1 - alpha_bar_t_next) * eps_pred
                )
        
        # Final step: ensure output is in valid image range
        x_t = torch.clamp(x_t, -1.0, 1.0)  # adjust to [0.0, 1.0] if your data uses that range
        
        return x_t

def create_somvae_diffusion_refiner(som_vae_model, device='cuda', timesteps=1000):
    """Factory function to create a diffusion refiner for a pretrained SOM-VAE
    
    Args:
        som_vae_model: Pretrained SOM-VAE model (will be frozen)
        device: Device to put the model on
        timesteps: Number of diffusion timesteps
    """
    # Freeze the SOM-VAE model
    for param in som_vae_model.parameters():
        param.requires_grad = False
    
    # Get the number of output channels from the SOM-VAE decoder
    in_channels = som_vae_model.decoder[-2].out_channels if hasattr(som_vae_model.decoder[-2], 'out_channels') else 3
    
    # Create the diffusion model (for refining quantized reconstructions)
    diffusion_model = SOMVAERefinerUNet(in_channels=in_channels).to(device)
    
    # Create scheduler
    scheduler = DiffusionScheduler(timesteps=timesteps).to(device)
    
    # Create refiner wrapper
    refiner = SOMVAEDiffusionRefiner(diffusion_model, scheduler, device)
    
    return refiner