import torch
from tqdm.auto import tqdm


@torch.no_grad()
def sample_ddim(refiner, som_vae_model, x_0, num_inference_steps=50, start_from_t=None):
    """
    DDIM sampling for refining SOM-VAE reconstructions using DiffuseVAE Formulation-1
    
    Args:
        refiner: Trained SOM-VAE diffusion refiner (with conditional architecture)
        som_vae_model: Pretrained frozen SOM-VAE model
        x_0: Original input [B, C, H, W] (used only to get x_hat_q via SOM-VAE)
        num_inference_steps: Number of DDIM steps (20-50)
        start_from_t: Starting timestep (None = full diffusion, или значение вроде 300)
    """
    refiner.model.eval()
    som_vae_model.eval()  # ← КРИТИЧЕСКИ ВАЖНО: переключить в eval режим!
    device = refiner.device
    
    # Получаем реконструкцию ПОСЛЕ квантования (второй выход = индекс 1)
    with torch.no_grad():
        outputs = som_vae_model(x_0)
        x_hat_q = outputs[1].detach()  # ← ИНДЕКС 1 — это x_hat_q (после квантования)
    
    B, C, H, W = x_hat_q.shape
    
    # Инициализация: зашумляем РЕКОНСТРУКЦИЮ (не чистый шум!)
    if start_from_t is None:
        start_from_t = refiner.scheduler.timesteps - 1
    
    t_start = torch.full((B,), start_from_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_hat_q)
    x_t = refiner.scheduler.add_noise(x_hat_q, noise, t_start)
    
    # DDIM sampling loop
    timesteps = torch.linspace(start_from_t, 0, num_inference_steps, device=device).long()
    
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # ✅ ПЕРЕДАЁМ УСЛОВИЕ (реконструкцию ПОСЛЕ квантования)
        eps_pred = refiner.model(x_t, t_batch, condition=x_hat_q)
        
        # Правильные накопленные альфы
        alpha_bar_t = refiner.scheduler.alphas_cumprod[t]
        alpha_bar_t_next = refiner.scheduler.alphas_cumprod[t_next]
        
        # Предсказать x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / (torch.sqrt(alpha_bar_t) + 1e-8)
        
        # DDIM update
        x_t = torch.sqrt(alpha_bar_t_next) * pred_x0 + torch.sqrt(1 - alpha_bar_t_next) * eps_pred
    
    # Клиппинг для стабильности
    x_t = torch.clamp(x_t, -1.0, 1.0)  # подстрой под свой диапазон нормализации
    
    return x_t


@torch.no_grad()
def refine_with_diffusion(refiner, som_vae_model, x_0, num_inference_steps=50):
    """Main inference method: x → z_q → x̂_q →(diffusion)→ x̃
    
    Args:
        refiner: Trained SOM-VAE diffusion refiner
        som_vae_model: Pretrained frozen SOM-VAE model
        x_0: Original input [B, C, H, W]
        num_inference_steps: Number of diffusion steps (20-50 recommended)
    """
    refiner.model.eval()
    
    device = refiner.device
    
    with torch.no_grad():
        # Step 1: x → z_q (get quantized latent from SOM-VAE)
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = som_vae_model(x_0)
        x_hat_q = x_hat_q.detach()  # Quantized reconstruction: x̂_q = decoder(z_q)

    B, C, H, W = x_hat_q.shape
    
    # Step 2: Apply diffusion refinement to x̂_q
    # Initialize with x̂_q and gradually denoise
    x_t = torch.randn((B, C, H, W), device=device)

    # Define timesteps for reverse diffusion
    timesteps = torch.linspace(refiner.scheduler.timesteps - 1, 0, num_inference_steps, device=device).long()

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict noise at current step
        eps_pred = refiner.model(x_t, t_batch)

        if t > 0:
            # DDIM update (faster than DDPM)
            alpha_t = refiner.scheduler.alphas[t]
            alpha_t_prev = refiner.scheduler.alphas[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0, device=device)
            
            # Predict x_0 from x_t
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            
            # Update x_t to next step
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * eps_pred
        else:
            # Final prediction
            pred_x0 = (x_t - torch.sqrt(1 - refiner.scheduler.alphas[t]) * eps_pred) / torch.sqrt(refiner.scheduler.alphas[t])
            x_t = pred_x0

    # The final result x_t is the refined version of the quantized reconstruction
    return x_t, x_hat_q


@torch.no_grad()
def reconstruct_with_diffusion_refiner(refiner, som_vae_model, x_0, num_inference_steps=50):
    """Complete reconstruction pipeline: x → z_q → x̂_q →(diffusion)→ x̃
    
    Args:
        refiner: Trained SOM-VAE diffusion refiner
        som_vae_model: Pretrained frozen SOM-VAE model
        x_0: Original input [B, C, H, W]
        num_inference_steps: Number of diffusion steps (20-50 recommended)
    
    Returns:
        tuple: (refined_reconstruction, som_vae_reconstruction, cluster_indices)
    """
    # Get SOM-VAE reconstruction and cluster indices
    x_hat_e, x_hat_q, z_e, z_q, indices, logits = som_vae_model(x_0)
    
    # Apply diffusion refinement
    refined_recon, _ = refine_with_diffusion(refiner, som_vae_model, x_0, num_inference_steps)
    
    return refined_recon, x_hat_q, indices