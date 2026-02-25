"""
SOM-VAE Diffusion Refiner Package

This package implements a diffusion model refiner for SOM-VAE (Self-Organizing Map Variational Autoencoder)
based on the DiffuseVAE concept. The refiner improves reconstruction quality by learning to denoise
the quantized reconstruction from SOM-VAE.

Architecture:
- x → z_q → x̂_q →(diffusion)→ x̃
- Where x̂_q is the quantized reconstruction from decoder(z_q)
- The diffusion model learns to refine x̂_q to produce better reconstructions
"""

from .model import (
    SOMVAEDiffusionRefiner,
    create_somvae_diffusion_refiner,
    SOMVAERefinerUNet,
    DiffusionScheduler
)

from .training import train_somvae_diffusion_refiner

from .inference import (
    sample_ddim,
    refine_with_diffusion,
    reconstruct_with_diffusion_refiner
)

__all__ = [
    'SOMVAEDiffusionRefiner',
    'create_somvae_diffusion_refiner',
    'SOMVAERefinerUNet',
    'DiffusionScheduler',
    'train_somvae_diffusion_refiner',
    'sample_ddim',
    'refine_with_diffusion',
    'reconstruct_with_diffusion_refiner'
]