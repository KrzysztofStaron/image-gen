import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for GAN that generates 28x28 MNIST-like images
    """
    def __init__(self, latent_dim=100, channels=1):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels

        # Define the generator architecture
        self.model = nn.Sequential(
            # Input: (latent_dim) -> Output: (512, 3, 3)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Output: (512, 3, 3) -> (256, 7, 7)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Output: (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Output: (128, 14, 14) -> (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Output: (64, 28, 28) -> (channels, 28, 28)
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        """
        Forward pass through generator

        Args:
            z (torch.Tensor): Random noise tensor of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, 28, 28)
        """
        # Reshape input to (batch_size, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


def create_generator(latent_dim=100, channels=1, device='cuda'):
    """
    Create and initialize generator model

    Args:
        latent_dim (int): Dimension of latent space
        channels (int): Number of output channels (1 for grayscale)
        device (str): Device to place model on

    Returns:
        Generator: Initialized generator model
    """
    model = Generator(latent_dim, channels)
    return model.to(device)
