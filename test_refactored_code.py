#!/usr/bin/env python3
"""
Simple test script to verify that the refactored SOM-VAE code works correctly
"""

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
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
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test that we can create the model"""
    print("\nTesting model creation...")
    
    try:
        from src.som_vae import EuroSAT_GlobalSOM_Deep
        
        # Create a small model for testing
        model = EuroSAT_GlobalSOM_Deep(
            in_channels=3,
            grid_size=(4, 4),  # Small grid for testing
            latent_dim=(64, 4, 4),  # Encoder produces 4x4 spatial dims, adjust channels as needed
            num_classes=10
        )
        
        print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_model_forward_pass():
    """Test that the model can perform a forward pass"""
    print("\nTesting model forward pass...")
    
    try:
        import torch
        from src.som_vae import EuroSAT_GlobalSOM_Deep
        
        # Create a small model for testing
        model = EuroSAT_GlobalSOM_Deep(
            in_channels=3,
            grid_size=(4, 4),
            latent_dim=(64, 4, 4),
            num_classes=10
        )
        
        # Create a dummy input
        x = torch.randn(2, 3, 64, 64)  # batch of 2, 3 channels, 64x64 images
        
        # Perform forward pass
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  x_hat_e shape: {x_hat_e.shape}")
        print(f"  x_hat_q shape: {x_hat_q.shape}")
        print(f"  z_e shape: {z_e.shape}")
        print(f"  z_q shape: {z_q.shape}")
        print(f"  indices shape: {indices.shape}")
        print(f"  logits shape: {logits.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_loss_function():
    """Test that the loss function works"""
    print("\nTesting loss function...")
    
    try:
        import torch
        from src.som_vae import EuroSAT_GlobalSOM_Deep, som_vae_loss
        
        # Create a small model for testing
        model = EuroSAT_GlobalSOM_Deep(
            in_channels=3,
            grid_size=(4, 4),
            latent_dim=(64, 4, 4),
            num_classes=10
        )
        
        # Create dummy inputs
        x = torch.randn(2, 3, 64, 64)
        labels = torch.randint(0, 10, (2,))
        
        # Forward pass
        x_hat_e, x_hat_q, z_e, z_q, indices, logits = model(x)
        
        # Calculate loss
        total_loss, rec_loss, comm_loss, som_loss, cls_loss = som_vae_loss(
            x, x_hat_e, x_hat_q, z_e, z_q, indices, logits, labels, model.som
        )
        
        print(f"✓ Loss calculation successful")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Reconstruction loss: {rec_loss.item():.4f}")
        print(f"  Commitment loss: {comm_loss.item():.4f}")
        print(f"  SOM loss: {som_loss.item():.4f}")
        print(f"  Classification loss: {cls_loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Loss calculation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running tests for refactored SOM-VAE code...\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_model_forward_pass,
        test_loss_function
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The refactoring was successful.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all(results)


if __name__ == "__main__":
    main()