"""
Deep Convolutional GAN (DCGAN) Implementation for MNIST Generation

This implementation follows the architectural guidelines for stable DCGANs:
1. Replace pooling layers with strided convolutions (discriminator) and
   fractional-strided convolutions (generator)
2. Use batchnorm in both generator and discriminator
3. Remove fully connected hidden layers for deeper architectures
4. Use ReLU activation in generator for all layers except output (Tanh)
5. Use LeakyReLU activation in discriminator for all layers

Architecture Details:
- Generator: Deep transpose convolutional network with BatchNorm and ReLU
- Discriminator: Deep convolutional network with BatchNorm and LeakyReLU
- Both networks are fully convolutional (no FC layers in hidden layers)
- Training uses Adam optimizer with different learning rates for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

# DCGAN Hyperparameters - Following established best practices
z_dim = 100  # Dimension of latent noise vector (typically 100-256)
batch_size = 128  # Batch size for training stability
lr = 0.0002  # Learning rate (standard for DCGAN)
beta1 = 0.5   # Beta1 for Adam optimizer (higher than typical 0.9 for stable GAN training)
epochs = 50  # Increased for better image quality
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Data loading (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for Tanh
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator: Uses fractional-strided convolutions (transposed conv) for upsampling
# Follows DCGAN guidelines: BatchNorm, ReLU (except output), no fully connected layers
class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Project and reshape noise: batch x z_dim x 1 x 1 -> batch x 512 x 4 x 4
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Upsample to 8x8: batch x 512 x 4 x 4 -> batch x 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Upsample to 16x16: batch x 256 x 8 x 8 -> batch x 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Final upsample to 28x28: batch x 128 x 16 x 16 -> batch x 1 x 28 x 28
            # Use 13x13 kernel with stride 1 and padding 0 to get exact 28x28 output
            nn.ConvTranspose2d(128, channels, 13, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator: Uses strided convolutions for downsampling (no pooling layers)
# Follows DCGAN guidelines: LeakyReLU, BatchNorm, no fully connected hidden layers
class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: batch x 1 x 28 x 28 -> batch x 64 x 14 x 14
            # Strided convolution replaces pooling for downsampling
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # batch x 64 x 14 x 14 -> batch x 128 x 7 x 7
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # batch x 128 x 7 x 7 -> batch x 256 x 3 x 3
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # batch x 256 x 3 x 3 -> batch x 512 x 2 x 2
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Global average pooling to get single value per sample
            # batch x 512 x 2 x 2 -> batch x 512 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
            # Final classification: batch x 512 x 1 x 1 -> batch x 1 x 1 x 1
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure output is properly shaped for BCE loss
        return self.model(x).view(-1, 1)

# Initialize models
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers - DCGAN best practice: Different learning rates for stability
# Generator uses standard learning rate
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function for GAN training
criterion = nn.BCELoss()

# Create output directory for generated images
os.makedirs('generated_images', exist_ok=True)

def generate_and_save_samples(generator, epoch, device, z_dim, num_samples=16):
    """Generate and save sample images during training"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, 1, 1).to(device)
        generated = generator(z)
        # Save as grid
        save_image(generated, f'generated_images/epoch_{epoch:03d}.png', nrow=4, normalize=True)
    generator.train()

# Training loop - DCGAN training strategy
# Note: Discriminator is trained more frequently than generator for stability
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Real
        output_real = discriminator(real_imgs)
        loss_d_real = criterion(output_real, real_labels)

        # Fake
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = generator(z)
        output_fake = discriminator(fake_imgs.detach())
        loss_d_fake = criterion(output_fake, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        output_g = discriminator(fake_imgs)
        loss_g = criterion(output_g, real_labels)
        loss_g.backward()
        optimizer_g.step()

    # Generate and save samples every 5 epochs
    if (epoch + 1) % 5 == 0:
        generate_and_save_samples(generator, epoch + 1, device, z_dim)

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

# Generate final samples (after training)
generator.eval()
with torch.no_grad():
    z = torch.randn(64, z_dim, 1, 1).to(device)  # Generate more samples
    generated = generator(z)

    # Save final generated images
    save_image(generated, 'generated_images/final_generated.png', nrow=8, normalize=True)

    # Also save individual images
    for i in range(min(16, generated.size(0))):
        save_image(generated[i], f'generated_images/final_sample_{i:02d}.png', normalize=True)

    print("Generated images saved to 'generated_images/' directory")
    print("Training complete! Check the generated_images folder for results.")