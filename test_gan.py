#!/usr/bin/env python3
"""
Test script for GAN functionality without full training.
Useful for testing the implementation and generating initial samples.
"""

import torch
from generator import create_generator
from discriminator import create_discriminator
from visualization import plot_generated_images, generate_digit_interpolation


def test_models():
    """Test that models can be created and run forward pass"""
    print("Testing GAN models...")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create models
    latent_dim = 100
    generator = create_generator(latent_dim=latent_dim, device=device)
    discriminator = create_discriminator(device=device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Test forward pass
    batch_size = 16
    noise = torch.randn(batch_size, latent_dim, device=device)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # Generate fake images
        fake_images = generator(noise)
        print(f"Generated images shape: {fake_images.shape}")
        print(f"Generated images range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")

        # Test discriminator
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # For discriminator test, we need real images too
        # Let's just test with fake images
        disc_output = discriminator(fake_images)
        print(f"Discriminator output shape: {disc_output.shape}")
        print(f"Discriminator output range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")

    print("✓ Models created and forward pass successful!")

    return generator, latent_dim, device


def generate_test_samples(generator, latent_dim, device):
    """Generate and save test samples"""
    print("Generating test samples...")

    plot_generated_images(
        generator,
        latent_dim,
        device=device,
        num_images=16,
        save_path='test_samples.png'
    )

    generate_digit_interpolation(
        generator,
        latent_dim,
        device=device,
        steps=8,
        save_path='test_interpolation'
    )

    print("✓ Test samples generated!")


def main():
    """Main test function"""
    print("=" * 50)
    print("GAN TEST SCRIPT")
    print("=" * 50)

    try:
        # Test models
        generator, latent_dim, device = test_models()

        # Generate test samples
        generate_test_samples(generator, latent_dim, device)

        print("\n" + "=" * 50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("Check test_samples.png for generated images")
        print("Check test_interpolation/ for interpolation frames")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
