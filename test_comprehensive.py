#!/usr/bin/env python3
"""
Comprehensive test to verify the complete refactored structure
"""

def test_complete_structure():
    """Test the complete refactored structure"""
    print("Testing complete refactored structure...")
    
    # Test 1: Individual module imports
    print("\n1. Testing individual module imports...")
    try:
        from src.datasets.dataset import EuroSATDataset, get_train_transform, get_val_transform
        print("   ✓ Datasets module imported successfully")
    except Exception as e:
        print(f"   ✗ Datasets module import failed: {e}")
        return False
    
    try:
        from src.som_vae.model import GlobalSpatialSOMLayer, EuroSAT_GlobalSOM_Deep
        print("   ✓ SOM-VAE model module imported successfully")
    except Exception as e:
        print(f"   ✗ SOM-VAE model module import failed: {e}")
        return False
    
    try:
        from src.som_vae.training import som_vae_loss, train_som_vae, restart_som_with_data
        print("   ✓ SOM-VAE training module imported successfully")
    except Exception as e:
        print(f"   ✗ SOM-VAE training module import failed: {e}")
        return False
    
    try:
        from src.som_vae.visualization import visualize_som_sample, plot_som_map, plot_som_reconstruction_map
        print("   ✓ SOM-VAE visualization module imported successfully")
    except Exception as e:
        print(f"   ✗ SOM-VAE visualization module import failed: {e}")
        return False
    
    # Test 2: Package import
    print("\n2. Testing package import...")
    try:
        from src.som_vae import (
            GlobalSpatialSOMLayer,
            EuroSAT_GlobalSOM_Deep,
            som_vae_loss,
            train_som_vae,
            restart_som_with_data,
            visualize_som_sample,
            plot_som_map,
            plot_som_reconstruction_map,
            EuroSATDataset,
            get_train_transform,
            get_val_transform
        )
        print("   ✓ Complete package imported successfully")
    except Exception as e:
        print(f"   ✗ Complete package import failed: {e}")
        return False
    
    # Test 3: Functional test
    print("\n3. Testing functionality...")
    try:
        import torch
        
        # Create a model
        model = EuroSAT_GlobalSOM_Deep(
            in_channels=3,
            grid_size=(4, 4),
            latent_dim=(64, 4, 4),
            num_classes=10
        )
        
        # Test forward pass
        x = torch.randn(1, 3, 64, 64)
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(x)
        
        # Test loss computation
        labels = torch.randint(0, 10, (1,))
        total_loss, rec_loss, comm_loss, som_loss, cls_loss = som_vae_loss(
            x, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som
        )
        
        print("   ✓ Functional test passed")
    except Exception as e:
        print(f"   ✗ Functional test failed: {e}")
        return False
    
    # Test 4: Check that original file was replaced appropriately
    print("\n4. Checking original file handling...")
    try:
        # The old som_vae.py file should still exist but be updated
        import os
        if os.path.exists("src/som_vae.py"):
            print("   ✓ Original som_vae.py file still exists")
        else:
            print("   ! Original som_vae.py file was removed (this may be intended)")
    except Exception as e:
        print(f"   ? Issue checking original file: {e}")
    
    print("\n5. Verifying new structure...")
    import os
    structure_checks = [
        ("src/datasets/dataset.py", os.path.exists("src/datasets/dataset.py")),
        ("src/som_vae/model.py", os.path.exists("src/som_vae/model.py")),
        ("src/som_vae/training.py", os.path.exists("src/som_vae/training.py")),
        ("src/som_vae/visualization.py", os.path.exists("src/som_vae/visualization.py")),
        ("src/som_vae/__init__.py", os.path.exists("src/som_vae/__init__.py")),
        ("src/datasets/__init__.py", os.path.exists("src/datasets/__init__.py")),
    ]
    
    all_good = True
    for name, exists in structure_checks:
        if exists:
            print(f"   ✓ {name} exists")
        else:
            print(f"   ✗ {name} missing")
            all_good = False
    
    return all_good


def main():
    print("Running comprehensive test of refactored structure...\n")
    
    success = test_complete_structure()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All comprehensive tests passed! The refactoring is complete and working correctly.")
        print("\nThe new structure is:")
        print("- src/")
        print("  ├── __init__.py")
        print("  ├── datasets/")
        print("  │   ├── __init__.py")
        print("  │   └── dataset.py (contains EuroSATDataset and transforms)")
        print("  └── som_vae/")
        print("      ├── __init__.py")
        print("      ├── model.py (contains GlobalSpatialSOMLayer and EuroSAT_GlobalSOM_Deep)")
        print("      ├── training.py (contains loss functions and training functions)")
        print("      └── visualization.py (contains visualization functions)")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return success


if __name__ == "__main__":
    main()