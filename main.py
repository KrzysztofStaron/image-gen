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

z_dim = 100
batch_size = 128
lr = 0.0002
beta1 = 0.5
epochs = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=1, num_classes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        # Calculate input dimension after concatenating noise and labels
        input_dim = z_dim + num_classes

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channels, 13, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros(labels.size(0), self.num_classes).to(z.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        # Flatten z to 2D: (batch_size, z_dim)
        z_flat = z.view(z.size(0), -1)

        # Concatenate noise and labels: (batch_size, z_dim + num_classes)
        input_vector = torch.cat([z_flat, one_hot_labels], dim=1)

        # Reshape for ConvTranspose2d: (batch_size, channels, height, width)
        input_vector = input_vector.view(input_vector.size(0), input_vector.size(1), 1, 1)

        return self.model(input_vector)

class Discriminator(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final layers after concatenating with labels
        self.final_layers = nn.Sequential(
            nn.Conv2d(512 + num_classes, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Process image through convolutional layers
        features = self.conv_layers(x)

        # Convert labels to one-hot encoding and expand to match spatial dimensions
        one_hot_labels = torch.zeros(labels.size(0), self.num_classes).to(x.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        # Expand labels to match feature map size (512 channels will be added)
        label_features = one_hot_labels.view(labels.size(0), self.num_classes, 1, 1)
        label_features = label_features.expand(-1, -1, features.size(2), features.size(3))

        # Concatenate features with label information
        combined_features = torch.cat([features, label_features], dim=1)

        return self.final_layers(combined_features).view(-1, 1)

num_classes = 10
generator = Generator(z_dim, num_classes=num_classes).to(device)
discriminator = Discriminator(num_classes=num_classes).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr*0.11, betas=(beta1, 0.999))

criterion = nn.BCELoss()

os.makedirs('generated_images', exist_ok=True)

def load_weights_if_exist(generator, discriminator, optimizer_g, optimizer_d, weights_path='saved_weights/dcgan_final_weights.pth'):
    """Load model weights if they exist for resuming training or inference."""
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)

        try:
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(".4f")
            return checkpoint['epoch']
        except RuntimeError as e:
            print(f"Warning: Could not load weights due to architecture mismatch: {e}")
            print("This might be because you're switching between conditional and non-conditional GAN.")
            print("Starting training from scratch with new architecture.")
            return 0
    else:
        print("No existing weights found, starting training from scratch")
        return 0

def load_generator_weights(generator, weights_path='saved_weights/generator_weights.pth'):
    """Load only generator weights for inference."""
    if os.path.exists(weights_path):
        print(f"Loading generator weights from {weights_path}")
        try:
            generator.load_state_dict(torch.load(weights_path, map_location=device))
            return True
        except RuntimeError as e:
            print(f"Warning: Could not load generator weights due to architecture mismatch: {e}")
            print("This might be because you're switching between conditional and non-conditional GAN.")
            return False
    else:
        print("No generator weights found")
        return False

def generate_and_save_samples(generator, epoch, device, z_dim, num_samples=16, num_classes=10):
    generator.eval()
    with torch.no_grad():
        # Generate random labels for diverse samples
        labels = torch.randint(0, num_classes, (num_samples,)).to(device)
        z = torch.randn(num_samples, z_dim, 1, 1).to(device)
        generated = generator(z, labels)
        save_image(generated, f'generated_images/epoch_{epoch:03d}.png', nrow=4, normalize=True)
    generator.train()

def generate_specific_digit(generator, digit, device, z_dim, num_samples=16):
    """
    Generate specific digit using the conditional GAN.

    Args:
        generator: The generator model
        digit: The digit to generate (0-9)
        device: Device to run on
        z_dim: Dimension of noise vector
        num_samples: Number of samples to generate
    """
    generator.eval()
    with torch.no_grad():
        # Create labels for the specific digit
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        z = torch.randn(num_samples, z_dim, 1, 1).to(device)
        generated = generator(z, labels)

        # Save the generated images
        os.makedirs('generated_images', exist_ok=True)
        save_image(generated, f'generated_images/digit_{digit}_samples.png', nrow=4, normalize=True)

        # Save individual samples
        for i in range(min(10, num_samples)):
            save_image(generated[i], f'generated_images/digit_{digit}_sample_{i:02d}.png', normalize=True)

        print(f"Generated {num_samples} samples of digit {digit}")
        print(f"Images saved to 'generated_images/digit_{digit}_samples.png'")

    generator.train()

def generate_all_digits(generator, device, z_dim, samples_per_digit=10):
    """
    Generate samples of all digits (0-9) using the conditional GAN.

    Args:
        generator: The generator model
        device: Device to run on
        z_dim: Dimension of noise vector
        samples_per_digit: Number of samples per digit
    """
    generator.eval()
    with torch.no_grad():
        all_images = []
        all_labels = []

        for digit in range(10):
            labels = torch.full((samples_per_digit,), digit, dtype=torch.long).to(device)
            z = torch.randn(samples_per_digit, z_dim, 1, 1).to(device)
            generated = generator(z, labels)
            all_images.append(generated)
            all_labels.extend([digit] * samples_per_digit)

        # Concatenate all images
        all_images = torch.cat(all_images, dim=0)

        # Save a grid of all digits
        save_image(all_images, 'generated_images/all_digits.png', nrow=10, normalize=True)
        print("Generated samples of all digits saved to 'generated_images/all_digits.png'")

    generator.train()

# Try to load existing weights to resume training
start_epoch = load_weights_if_exist(generator, discriminator, optimizer_g, optimizer_d)

for epoch in range(start_epoch, epochs):
    for i, (real_imgs, labels) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        batch_size = real_imgs.size(0)

        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        output_real = discriminator(real_imgs, labels)
        loss_d_real = criterion(output_real, real_labels)

        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = generator(z, labels)
        output_fake = discriminator(fake_imgs.detach(), labels)
        loss_d_fake = criterion(output_fake, fake_labels)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        output_g = discriminator(fake_imgs, labels)
        loss_g = criterion(output_g, real_labels)
        loss_g.backward()
        optimizer_g.step()

    if (epoch + 1) % 5 == 0:
        generate_and_save_samples(generator, epoch + 1, device, z_dim, num_classes=num_classes)

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

# Save model weights after training
os.makedirs('saved_weights', exist_ok=True)
torch.save({
    'epoch': epochs,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_g_state_dict': optimizer_g.state_dict(),
    'optimizer_d_state_dict': optimizer_d.state_dict(),
    'loss_g': loss_g.item(),
    'loss_d': loss_d.item(),
}, 'saved_weights/dcgan_final_weights.pth')

# Save separate weight files for easier loading
torch.save(generator.state_dict(), 'saved_weights/generator_weights.pth')
torch.save(discriminator.state_dict(), 'saved_weights/discriminator_weights.pth')

print("Model weights saved to 'saved_weights/' directory")

generator.eval()
with torch.no_grad():
    # Generate random samples for final evaluation
    labels = torch.randint(0, num_classes, (64,)).to(device)
    z = torch.randn(64, z_dim, 1, 1).to(device)
    generated = generator(z, labels)

    save_image(generated, 'generated_images/final_generated.png', nrow=8, normalize=True)

    for i in range(min(16, generated.size(0))):
        save_image(generated[i], f'generated_images/final_sample_{i:02d}.png', normalize=True)

    # Generate specific digits as examples
    print("\nGenerating specific digit examples...")
    for digit in [0, 1, 2, 5, 8]:
        generate_specific_digit(generator, digit, device, z_dim, num_samples=8)

    # Generate all digits
    generate_all_digits(generator, device, z_dim, samples_per_digit=8)

    print("\nGenerated images saved to 'generated_images/' directory")
    print("Training complete! Check the generated_images folder for results.")
    print("You can now generate specific digits using the generate_specific_digit() function!")
    print("\nExample usage after training:")
    print("# Generate 16 samples of digit '5'")
    print("generate_specific_digit(generator, 5, device, z_dim, num_samples=16)")
    print("\n# Generate samples of all digits")
    print("generate_all_digits(generator, device, z_dim, samples_per_digit=10)")