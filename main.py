#!/usr/bin/env python3
"""
Main script for training a GAN on MNIST handwritten digits dataset.

This script implements a Deep Convolutional GAN (DCGAN) that generates realistic
handwritten digits similar to the MNIST dataset.
"""

import torch
import argparse
from data_loader import get_mnist_dataloader
from generator import create_generator
from discriminator import create_discriminator
from train_utils import GANTrainer
from visualization import (
    plot_generated_images,
    compare_real_fake,
    generate_digit_interpolation,
    plot_training_progress,
    save_model_summary
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GAN on MNIST dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--sample_interval', type=int, default=100, help='Interval for generating samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    return parser.parse_args()


def main():
    """Main training function"""
    print("=" * 60)
    print("MNIST GAN TRAINING SCRIPT")
    print("=" * 60)

    # Check PyTorch and CUDA setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 60)

    # Parse arguments
    args = parse_arguments()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create data loader
    print("Loading MNIST dataset...")
    dataloader = get_mnist_dataloader(batch_size=args.batch_size)

    # Create models
    print("Creating GAN models...")
    generator = create_generator(latent_dim=args.latent_dim, device=args.device)
    discriminator = create_discriminator(device=args.device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Create trainer with stability improvements
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=args.device,
        lr=args.lr,
        beta1=args.beta1,
        label_smoothing=0.1,  # Smooth labels from 1.0/0.0 to 0.9/0.1
        noise_std=0.1        # Add noise to discriminator inputs
    )

    # Save model summary
    save_model_summary(generator, discriminator)

    # Train the GAN
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.epochs,
        latent_dim=args.latent_dim,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )

    # Generate final results
    print("\nGenerating final results...")

    # Generate sample images
    plot_generated_images(
        generator,
        args.latent_dim,
        device=args.device,
        num_images=16,
        save_path='generated_images/final_samples.png'
    )

    # Get a batch of real images for comparison
    real_images, _ = next(iter(dataloader))

    # Compare real vs fake images
    compare_real_fake(
        real_images,
        generator,
        args.latent_dim,
        device=args.device,
        num_images=8,
        save_path='generated_images/real_vs_fake_comparison.png'
    )

    # Generate interpolation between digits
    generate_digit_interpolation(
        generator,
        args.latent_dim,
        device=args.device,
        steps=10,
        save_path='generated_images/interpolation'
    )

    # Create training progress collage
    plot_training_progress(save_path='generated_images/training_progress.png')

    print("\nTraining completed successfully!")
    print("Check the 'generated_images' directory for results.")
    print("Check the 'checkpoints' directory for saved models.")


if __name__ == "__main__":
    main()