import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from data_loader import get_noise, denormalize_images


class GANTrainer:
    """
    Trainer class for GAN training with stability improvements
    """
    def __init__(self, generator, discriminator, device='cuda', lr=0.0002, beta1=0.5, beta2=0.999,
                 label_smoothing=0.1, noise_std=0.1):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers - separate learning rates for stability
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(beta1, beta2))  # Slower D learning

        # Labels for real and fake images with smoothing
        self.real_label = 1.0 - label_smoothing  # Smoothed to 0.9
        self.fake_label = 0.0 + label_smoothing  # Smoothed to 0.1

        # Noise injection for discriminator
        self.noise_std = noise_std

        # Add spectral normalization to discriminator
        self._add_spectral_norm()

        # Create directories for saving models and images
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('generated_images', exist_ok=True)

    def _add_spectral_norm(self):
        """Add spectral normalization to discriminator for stability"""
        for module in self.discriminator.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)

    def train_step(self, real_images, latent_dim):
        """
        Perform one training step with stability improvements

        Args:
            real_images (torch.Tensor): Batch of real images
            latent_dim (int): Dimension of latent space

        Returns:
            tuple: (d_loss, g_loss) - discriminator and generator losses
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Add noise to real images for discriminator training
        if self.noise_std > 0:
            noise = torch.randn_like(real_images) * self.noise_std
            real_images_noisy = real_images + noise
        else:
            real_images_noisy = real_images

        # Create labels with some randomization for stability
        real_labels = torch.full((batch_size,), self.real_label, device=self.device)
        fake_labels = torch.full((batch_size,), self.fake_label, device=self.device)

        # Add small random noise to labels
        real_labels += torch.randn_like(real_labels) * 0.05
        fake_labels += torch.randn_like(fake_labels) * 0.05
        real_labels.clamp_(0.7, 1.0)  # Keep labels in reasonable range
        fake_labels.clamp_(0.0, 0.3)

        # ================================
        # Train Discriminator
        # ================================

        self.discriminator.zero_grad()

        # Train with real images
        output_real = self.discriminator(real_images_noisy).view(-1)
        loss_D_real = self.criterion(output_real, real_labels)

        # Train with fake images
        noise = get_noise(batch_size, latent_dim, self.device)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach()).view(-1)
        loss_D_fake = self.criterion(output_fake, fake_labels)

        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        self.optimizer_D.step()

        # ================================
        # Train Generator (with potential multiple steps)
        # ================================

        # Sometimes train generator multiple times per discriminator step
        num_gen_steps = 2 if torch.rand(1).item() < 0.3 else 1  # 30% chance of 2 steps

        total_g_loss = 0
        for _ in range(num_gen_steps):
            self.generator.zero_grad()

            # Generate new fake images
            noise = get_noise(batch_size, latent_dim, self.device)
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images).view(-1)

            # Generator loss with non-saturating loss
            loss_G = self.criterion(output_fake, real_labels)
            total_g_loss += loss_G.item()

            loss_G.backward()
            self.optimizer_G.step()

        avg_g_loss = total_g_loss / num_gen_steps

        return loss_D.item(), avg_g_loss

    def train(self, dataloader, num_epochs, latent_dim, save_interval=10, sample_interval=100):
        """
        Train the GAN

        Args:
            dataloader (DataLoader): Training data loader
            num_epochs (int): Number of training epochs
            latent_dim (int): Dimension of latent space
            save_interval (int): Interval for saving model checkpoints
            sample_interval (int): Interval for generating sample images
        """
        print("Starting GAN training...")
        print(f"Device: {self.device}")
        print(f"Latent dimension: {latent_dim}")
        print(f"Training for {num_epochs} epochs")
        print("-" * 50)

        # Training statistics
        D_losses = []
        G_losses = []

        # Training loop
        for epoch in range(num_epochs):
            epoch_D_losses = []
            epoch_G_losses = []

            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

            for i, (real_images, _) in enumerate(progress_bar):
                d_loss, g_loss = self.train_step(real_images, latent_dim)

                epoch_D_losses.append(d_loss)
                epoch_G_losses.append(g_loss)

                # Update progress bar
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}'
                })

                # Generate sample images
                if (i + 1) % sample_interval == 0:
                    self.generate_samples(epoch, i + 1, latent_dim)

            # Calculate average losses for epoch
            avg_D_loss = sum(epoch_D_losses) / len(epoch_D_losses)
            avg_G_loss = sum(epoch_G_losses) / len(epoch_G_losses)

            D_losses.append(avg_D_loss)
            G_losses.append(avg_G_loss)

            print(f'Epoch {epoch+1}/{num_epochs} - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')

            # Save model checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)

        print("Training completed!")
        self.plot_losses(D_losses, G_losses)
        self.save_final_models()

    def generate_samples(self, epoch, batch_idx, latent_dim, num_samples=16):
        """
        Generate and save sample images

        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch index
            latent_dim (int): Dimension of latent space
            num_samples (int): Number of samples to generate
        """
        self.generator.eval()

        with torch.no_grad():
            noise = get_noise(num_samples, latent_dim, self.device)
            fake_images = self.generator(noise)

            # Denormalize images
            fake_images = denormalize_images(fake_images).cpu()

            # Create grid of images
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            axes = axes.flatten()

            for i in range(num_samples):
                axes[i].imshow(fake_images[i].squeeze(), cmap='gray')
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(f'generated_images/samples_epoch_{epoch}_batch_{batch_idx}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

        self.generator.train()

    def plot_losses(self, D_losses, G_losses):
        """Plot and save training losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
        plt.plot(G_losses, label='Generator Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('generated_images/training_losses.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')

    def save_final_models(self):
        """Save final trained models"""
        torch.save(self.generator.state_dict(), 'checkpoints/generator_final.pth')
        torch.save(self.discriminator.state_dict(), 'checkpoints/discriminator_final.pth')
        print("Final models saved!")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        return checkpoint['epoch']
