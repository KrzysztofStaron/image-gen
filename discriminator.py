import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator network for GAN that classifies 28x28 images as real or fake
    """
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        self.channels = channels

        # Define the discriminator architecture
        self.model = nn.Sequential(
            # Input: (channels, 28, 28) -> (64, 14, 14)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 7, 7) -> (256, 3, 3)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 3, 3) -> (512, 3, 3)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # (512, 3, 3) -> (1, 3, 3)
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False),

            # Global average pooling to get single output value
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Forward pass through discriminator

        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, 28, 28)

        Returns:
            torch.Tensor: Probability that images are real (shape: (batch_size, 1))
        """
        return self.model(x)


def create_discriminator(channels=1, device='cuda'):
    """
    Create and initialize discriminator model

    Args:
        channels (int): Number of input channels (1 for grayscale)
        device (str): Device to place model on

    Returns:
        Discriminator: Initialized discriminator model
    """
    model = Discriminator(channels)
    return model.to(device)
