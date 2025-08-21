#!/usr/bin/env python3
"""
Stable GAN training with monitoring and early stopping
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
    parser = argparse.ArgumentParser(description='Train stable GAN on MNIST dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')  # Smaller batch
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # Lower LR
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='Label smoothing factor')
    parser.add_argument('--noise_std', type=float, default=0.15, help='Noise standard deviation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    return parser.parse_args()


class StableGANTrainer(GANTrainer):
    """Enhanced GAN trainer with stability monitoring"""

    def __init__(self, generator, discriminator, device='cuda', lr=0.0002, beta1=0.5, beta2=0.999,
                 label_smoothing=0.1, noise_std=0.1, patience=10):
        super().__init__(generator, discriminator, device, lr, beta1, beta2, label_smoothing, noise_std)
        self.patience = patience
        self.best_g_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []

    def train_step_with_monitoring(self, real_images, latent_dim):
        """Training step with stability monitoring"""
        d_loss, g_loss = self.train_step(real_images, latent_dim)

        # Monitor for instability
        self.loss_history.append((d_loss, g_loss))

        # Check for discriminator overpowering (D_loss too low, G_loss too high)
        if len(self.loss_history) > 5:
            recent_d = sum(d for d, g in self.loss_history[-5:]) / 5
            recent_g = sum(g for d, g in self.loss_history[-5:]) / 5

            if recent_d < 0.1 and recent_g > 5.0:  # Instability detected
                print(f"⚠️  Instability detected! D_loss: {recent_d:.4f}, G_loss: {recent_g:.4f}")
                print("🔧 Applying stability measures...")

                # Temporarily reduce discriminator learning rate
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] *= 0.5

                # Train generator more
                self._extra_generator_steps(real_images, latent_dim, num_steps=3)

                # Restore discriminator learning rate
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] *= 2.0

        return d_loss, g_loss

    def _extra_generator_steps(self, real_images, latent_dim, num_steps=2):
        """Train generator extra steps to catch up"""
        batch_size = real_images.size(0)
        real_labels = torch.full((batch_size,), self.real_label, device=self.device)

        for _ in range(num_steps):
            self.generator.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=self.device)
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images).view(-1)
            loss_G = self.criterion(output_fake, real_labels)
            loss_G.backward()
            self.optimizer_G.step()

    def check_early_stopping(self, g_loss):
        """Check if training should stop early"""
        if g_loss < self.best_g_loss:
            self.best_g_loss = g_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print(f"🛑 Early stopping triggered after {self.patience} epochs without improvement")
            return True
        return False


def main():
    """Main stable training function"""
    print("=" * 70)
    print("STABLE GAN TRAINING WITH MONITORING")
    print("=" * 70)

    # Check PyTorch and CUDA setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 70)

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

    # Create stable trainer
    trainer = StableGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=args.device,
        lr=args.lr,
        beta1=args.beta1,
        label_smoothing=args.label_smoothing,
        noise_std=args.noise_std,
        patience=args.patience
    )

    # Save model summary
    save_model_summary(generator, discriminator)

    # Train with stability monitoring
    print(f"Starting stable training for {args.epochs} epochs...")
    print("🔍 Monitoring for instability and applying corrections automatically")

    D_losses = []
    G_losses = []

    for epoch in range(args.epochs):
        epoch_D_losses = []
        epoch_G_losses = []

        print(f"\nEpoch {epoch+1}/{args.epochs}")

        for i, (real_images, _) in enumerate(dataloader):
            d_loss, g_loss = trainer.train_step_with_monitoring(real_images, args.latent_dim)

            epoch_D_losses.append(d_loss)
            epoch_G_losses.append(g_loss)

            # Print progress every 50 batches
            if i % 50 == 0:
                print(f"  Batch {i:3d} - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")

        # Calculate average losses for epoch
        avg_D_loss = sum(epoch_D_losses) / len(epoch_D_losses)
        avg_G_loss = sum(epoch_G_losses) / len(epoch_G_losses)

        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)

        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   D_loss: {avg_D_loss:.4f}")
        print(f"   G_loss: {avg_G_loss:.4f}")

        # Check for early stopping
        if trainer.check_early_stopping(avg_G_loss):
            break

        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch + 1)

    print("\n🎉 Training completed!")
    trainer.plot_losses(D_losses, G_losses)
    trainer.save_final_models()

    # Generate final results
    print("\n🖼️  Generating final results...")

    # Generate sample images
    plot_generated_images(
        generator,
        args.latent_dim,
        device=args.device,
        num_images=16,
        save_path='generated_images/stable_final_samples.png'
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
        save_path='generated_images/stable_real_vs_fake.png'
    )

    print("\n✅ Check the 'generated_images' folder for results!")
    print("📁 Check the 'checkpoints' folder for saved models!")


if __name__ == "__main__":
    main()
