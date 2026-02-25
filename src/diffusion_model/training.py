import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from src.diffusion_model.model import create_somvae_diffusion_refiner


def train_somvae_diffusion_refiner(som_vae_model, train_dataset, val_dataset,
                                   epochs=10, batch_size=32, lr=2e-4,
                                   device='cuda', timesteps=1000, num_inference_steps=50):
    """Train the diffusion refiner on top of a pretrained SOM-VAE using DiffuseVAE Formulation-1
    
    Args:
        som_vae_model: Pretrained frozen SOM-VAE model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        timesteps: Number of diffusion timesteps
        num_inference_steps: Number of inference steps for validation
    """
    
    
    # Create the diffusion refiner (now with conditional architecture)
    refiner = create_somvae_diffusion_refiner(som_vae_model, device, timesteps)
    
    # Create optimizer for diffusion model only
    optimizer = torch.optim.Adam(refiner.model.parameters(), lr=lr)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # Training loop
        refiner.model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for x_0, _, _ in pbar:
            x_0 = x_0.to(device)
            
            loss_val = refiner.train_step(som_vae_model, x_0, optimizer)
            train_loss += loss_val
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

        avg_train_loss = train_loss / num_batches
        
        # Validation loop — ИСПРАВЛЕНО: используем ту же логику, что и в train_step!
        refiner.model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for x_0_val, _, _ in val_loader:
                x_0_val = x_0_val.to(device)
                
                # ← КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: зашумляем ОРИГИНАЛ, а не реконструкцию!
                with torch.no_grad():
                    _, x_hat_q, _, _, _, _ = som_vae_model(x_0_val)
                    x_hat_q = x_hat_q.detach()

                t = torch.randint(0, refiner.scheduler.timesteps, (x_0_val.shape[0],), device=device)
                eps = torch.randn_like(x_0_val)  # ← шум для оригинала!
                x_t = refiner.scheduler.add_noise(x_0_val, eps, t)  # ← зашумляем оригинал!
                
                # ← ПЕРЕДАЁМ УСЛОВИЕ (реконструкцию) в модель!
                eps_pred = refiner.model(x_t, t, condition=x_hat_q)
                
                loss = F.mse_loss(eps_pred, eps)
                
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        
        tqdm.write(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    return refiner