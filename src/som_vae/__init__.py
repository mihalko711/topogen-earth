"""
SOM-VAE implementation for satellite imagery classification
Based on the EuroSAT dataset
"""

# Import all necessary components from the new modular structure
from .model import GlobalSpatialSOMLayer, EuroSAT_GlobalSOM_Deep
from .training import (
    som_vae_loss,
    train_som_vae,
    train_som_vae_pretrained,
    restart_som_with_data,
    save_model,
    load_model
)
from .visualization import (
    visualize_som_sample,
    plot_som_map,
    plot_som_reconstruction_map,
    analyze_latent_similarity,
    analyze_random,
    visualize_enhancement
)

# Also expose the dataset
from ..datasets.dataset import EuroSATDataset, get_train_transform, get_val_transform