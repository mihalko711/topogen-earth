"""
Example script demonstrating the usage of the SOM-VAE Diffusion Refiner
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os

# Import the diffusion refiner modules
from src.diffusion_model.model import create_somvae_diffusion_refiner, SOMVAEDiffusionRefiner
from src.diffusion_model.training import train_somvae_diffusion_refiner
from src.diffusion_model.inference import reconstruct_with_diffusion_refiner


def load_pretrained_som_vae(checkpoint_path=None):
    """Load a pretrained SOM-VAE model
    
    Args:
        checkpoint_path: Path to the pretrained SOM-VAE checkpoint
    """
    from src.som_vae.model import EuroSAT_GlobalSOM_Deep
    
    # Create a SOM-VAE model with the same architecture as used in the project
    som_vae_model = EuroSAT_GlobalSOM_Deep(
        in_channels=3,
        grid_size=(16, 16),
        latent_dim=(32, 4, 4),
        num_classes=10
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load the pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        som_vae_model.load_state_dict(checkpoint)
        print(f"Loaded pretrained SOM-VAE from {checkpoint_path}")
    else:
        print("Using randomly initialized SOM-VAE model (in practice, load a pretrained one)")
    
    return som_vae_model


def create_sample_dataset(size=100, img_size=64):
    """Create a sample dataset for demonstration purposes"""
    # Create dummy data for demonstration
    images = torch.randn(size, 3, img_size, img_size)
    labels = torch.randint(0, 10, (size,))
    cluster_assignments = torch.randint(0, 256, (size,))  # 16x16 grid = 256 clusters
    
    dataset = TensorDataset(images, labels, cluster_assignments)
    return dataset


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create a pretrained SOM-VAE model
    # Note: In practice, you would load a pretrained model
    som_vae_model = load_pretrained_som_vae()
    som_vae_model = som_vae_model.to(device)
    
    # Create sample datasets (in practice, use your actual dataset)
    train_dataset = create_sample_dataset(size=500)
    val_dataset = create_sample_dataset(size=100)
    
    print("Starting training of the diffusion refiner...")
    
    # Train the diffusion refiner
    refiner = train_somvae_diffusion_refiner(
        som_vae_model=som_vae_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,  # Reduced for demo
        batch_size=16,  # Reduced for demo
        lr=2e-4,
        device=device,
        timesteps=1000,
        num_inference_steps=50
    )
    
    print("Training completed!")
    
    # Perform inference with the trained refiner
    print("Performing inference with the diffusion refiner...")
    
    # Get a sample batch
    sample_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    sample_x, sample_y, sample_clusters = next(iter(sample_loader))
    sample_x = sample_x.to(device)
    
    # Reconstruct using the complete pipeline
    refined_recon, som_vae_recon, cluster_indices = reconstruct_with_diffusion_refiner(
        refiner=refiner,
        som_vae_model=som_vae_model,
        x_0=sample_x,
        num_inference_steps=20  # Faster inference for demo
    )
    
    print(f"Original shape: {sample_x.shape}")
    print(f"SOM-VAE reconstruction shape: {som_vae_recon.shape}")
    print(f"Refined reconstruction shape: {refined_recon.shape}")
    print(f"Cluster indices shape: {cluster_indices.shape}")
    
    print("Diffusion refiner example completed successfully!")


if __name__ == "__main__":
    main()