import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
z_dim = 100
batch_size = 128
lr = 0.0002
beta1 = 0.5
epochs = 25  # Small number for minimal training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for Tanh
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: batch x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 7x7x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 14x14x64
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 28x28x1
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: batch x 1 x 28 x 28
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14x64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 7x7x128
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
            # 1x1x1
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

# Initialize models
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss
criterion = nn.BCELoss()

# Training loop
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

    print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

# Generate samples (after training)
generator.eval()
with torch.no_grad():
    z = torch.randn(16, z_dim, 1, 1).to(device)
    generated = generator(z).cpu().numpy()

# Display some generated images (for visualization, in a real env you'd use plt.show())
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated[i][0], cmap='gray')
    ax.axis('off')
# plt.show()  # Uncomment to display in your environment