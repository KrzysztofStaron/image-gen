import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np


def get_mnist_dataloader(batch_size=128, image_size=28):
    """
    Load MNIST dataset and return DataLoader

    Args:
        batch_size (int): Batch size for training
        image_size (int): Size to resize images to

    Returns:
        DataLoader: MNIST training data loader
    """
    # Define transforms for MNIST images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
    ])

    # Download and load MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return train_loader


def get_noise(batch_size, latent_dim, device):
    """
    Generate random noise for generator input

    Args:
        batch_size (int): Number of noise samples to generate
        latent_dim (int): Dimension of latent space
        device: PyTorch device

    Returns:
        torch.Tensor: Random noise tensor
    """
    return torch.randn(batch_size, latent_dim, device=device)


def denormalize_images(images):
    """
    Convert images from [-1, 1] range back to [0, 1] range for visualization

    Args:
        images (torch.Tensor): Images in range [-1, 1]

    Returns:
        torch.Tensor: Images in range [0, 1]
    """
    return (images + 1) / 2
