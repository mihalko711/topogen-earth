# SOM-VAE Diffusion Refiner

This module implements a diffusion model refiner for SOM-VAE (Self-Organizing Map Variational Autoencoder) as described in the DiffuseVAE concept. The refiner improves the reconstruction quality of SOM-VAE by learning to denoise the quantized reconstruction.

## Architecture Overview

The diffusion refiner follows this pipeline:
```
x → z_q → x̂_q →(diffusion 20–50 steps)→ x̃
```

Where:
- `x`: Original input image
- `z_q`: Quantized latent representation from SOM-VAE
- `x̂_q`: Reconstruction from quantized latent (`decoder(z_q)`)
- `x̃`: Final refined reconstruction after diffusion

## Key Features

### 1. Separate Training
- The pretrained SOM-VAE model is frozen during diffusion training
- Only the diffusion model parameters are updated

### 2. Input to Diffusion
- The diffusion model receives the noisy quantized reconstruction `x̂_q^t` as input
- This is different from standard diffusion which operates on the original `x`
- The conditioning on `z_q` is not needed since semantic information is already encoded in `x̂_q`

### 3. Loss Function
- Standard diffusion loss: `𝔼[‖ε − ε_θ(x̂_q^t, t)‖²]`
- Noise `ε` is added to `x̂_q` (quantized reconstruction), not to `x` (original)

### 4. Inference
- Uses DDIM (Denoising Diffusion Implicit Models) sampler for faster inference
- Typically 20-50 steps instead of 1000 for DDPM
- Maintains cluster interpretability while improving reconstruction quality

## Components

### Model (`model.py`)
- `DiffusionScheduler`: Handles noise scheduling
- `SOMVAERefinerUNet`: U-Net architecture for refining quantized reconstructions
- `SOMVAEDiffusionRefiner`: Main class wrapping the diffusion model

### Training (`training.py`)
- `train_somvae_diffusion_refiner`: Main training function

### Inference (`inference.py`)
- `reconstruct_with_diffusion_refiner`: Complete reconstruction pipeline
- `refine_with_diffusion`: Core refinement function
- `sample_ddim`: DDIM sampling for fast inference

## Usage

```python
from src.diffusion_model.model import create_somvae_diffusion_refiner
from src.diffusion_model.training import train_somvae_diffusion_refiner
from src.diffusion_model.inference import reconstruct_with_diffusion_refiner

# Load pretrained SOM-VAE
som_vae_model = load_pretrained_som_vae()

# Train the refiner
refiner = train_somvae_diffusion_refiner(
    som_vae_model=som_vae_model,
    train_dataset=train_data,
    val_dataset=val_data,
    epochs=10,
    batch_size=32,
    device='cuda'
)

# Use the refiner for reconstruction
refined_recon, som_vae_recon, cluster_indices = reconstruct_with_diffusion_refiner(
    refiner=refiner,
    som_vae_model=som_vae_model,
    x_0=input_image,
    num_inference_steps=50
)
```

## Benefits

1. **Improved Reconstruction Quality**: The diffusion refiner "cleans up" artifacts introduced by quantization in SOM-VAE
2. **Preserved Interpretability**: Cluster assignments remain interpretable since we're refining the quantized representation
3. **Faster Inference**: DDIM allows for 20-50 step inference vs 1000 steps for DDPM
4. **Modular Design**: Can be applied on top of any pretrained SOM-VAE without retraining