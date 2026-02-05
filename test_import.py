#!/usr/bin/env python3
"""
Тест импорта модулей проекта
"""

import sys
import os

# Добавляем src в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

print(f"Current directory: {current_dir}")
print(f"Added to path: {src_path}")
print(f"Python path: {sys.path[:3]}...")

try:
    # Тест импорта diffusion_vae
    from diffusion_vae import create_diffusion_vae, ConditionalDiffusionVAE
    print("✅ Successfully imported diffusion_vae!")

    # Тест создания scheduler
    from diffusion_vae import DiffusionScheduler
    scheduler = DiffusionScheduler(timesteps=100)
    print(f"✅ Scheduler created with {scheduler.timesteps} timesteps")

    # Тест создания U-Net
    from diffusion_vae import LightConditionalUNet
    unet = LightConditionalUNet()
    print("✅ U-Net created successfully!")

    # Тест импорта som_vae
    from som_vae import EuroSAT_GlobalSOM_Deep, GlobalSpatialSOMLayer
    print("✅ Successfully imported som_vae!")

    # Тест создания модели SOM-VAE
    som_vae_model = EuroSAT_GlobalSOM_Deep(grid_size=(8, 8), latent_dim=(128, 8, 8))
    print("✅ SOM-VAE model created successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
