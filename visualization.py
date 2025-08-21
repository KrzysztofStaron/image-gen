import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_noise, denormalize_images
import os


def plot_generated_images(generator, latent_dim, device='cuda', num_images=16, figsize=(8, 8), save_path=None):
    """
    Generate and plot a grid of images from the generator

    Args:
        generator: Trained generator model
        latent_dim (int): Dimension of latent space
        device (str): Device to run on
        num_images (int): Number of images to generate
        figsize (tuple): Figure size for the plot
        save_path (str): Path to save the plot (optional)
    """
    generator.eval()

    with torch.no_grad():
        # Generate random noise
        noise = get_noise(num_images, latent_dim, device)
        fake_images = generator(noise)

        # Denormalize images
        fake_images = denormalize_images(fake_images).cpu()

        # Create grid
        grid_size = int(np.sqrt(num_images))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)

        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(num_images):
            axes[i].imshow(fake_images[i].squeeze(), cmap='gray')
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Generated images saved to {save_path}")
        else:
            plt.show()

        plt.close()

    generator.train()


def compare_real_fake(real_images, generator, latent_dim, device='cuda', num_images=8, save_path=None):
    """
    Create a comparison plot of real vs fake images

    Args:
        real_images (torch.Tensor): Batch of real images
        generator: Trained generator model
        latent_dim (int): Dimension of latent space
        device (str): Device to run on
        num_images (int): Number of image pairs to compare
        save_path (str): Path to save the plot (optional)
    """
    generator.eval()

    with torch.no_grad():
        # Get real images
        real_batch = real_images[:num_images].cpu()
        real_batch = denormalize_images(real_batch)

        # Generate fake images
        noise = get_noise(num_images, latent_dim, device)
        fake_batch = generator(noise)
        fake_batch = denormalize_images(fake_batch).cpu()

        # Create comparison plot
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))

        for i in range(num_images):
            # Real image
            axes[0, i].imshow(real_batch[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Real', fontsize=12)

            # Fake image
            axes[1, i].imshow(fake_batch[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Generated', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Real vs fake comparison saved to {save_path}")
        else:
            plt.show()

        plt.close()

    generator.train()


def generate_digit_interpolation(generator, latent_dim, device='cuda', steps=10, save_path=None):
    """
    Generate smooth interpolation between two random latent vectors

    Args:
        generator: Trained generator model
        latent_dim (int): Dimension of latent space
        device (str): Device to run on
        steps (int): Number of interpolation steps
        save_path (str): Path to save the animation frames (optional)
    """
    generator.eval()

    with torch.no_grad():
        # Generate two random latent vectors
        z1 = get_noise(1, latent_dim, device)
        z2 = get_noise(1, latent_dim, device)

        # Create interpolation
        alphas = torch.linspace(0, 1, steps).to(device)
        interpolated_images = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp)
            img = denormalize_images(img).cpu().squeeze()
            interpolated_images.append(img)

        # Create animation plot
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame):
            ax.clear()
            ax.imshow(interpolated_images[frame], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Interpolation Step {frame + 1}/{steps}')

        # Save individual frames
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for i, img in enumerate(interpolated_images):
                plt.figure(figsize=(4, 4))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{save_path}/interpolation_{i:03d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            print(f"Interpolation frames saved to {save_path}")

        # Show final plot
        plt.figure(figsize=(15, 3))
        for i, img in enumerate(interpolated_images):
            plt.subplot(1, steps, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'{i + 1}')

        if save_path:
            plt.savefig(f'{save_path}/interpolation_sequence.png', dpi=150, bbox_inches='tight')
            print(f"Interpolation sequence saved to {save_path}/interpolation_sequence.png")
        else:
            plt.show()

        plt.close()

    generator.train()


def plot_training_progress(image_dir='generated_images', save_path=None):
    """
    Create a collage of generated images from different training stages

    Args:
        image_dir (str): Directory containing generated sample images
        save_path (str): Path to save the collage (optional)
    """
    import glob
    from PIL import Image

    # Get all sample image files
    sample_files = glob.glob(f'{image_dir}/samples_*.png')
    sample_files.sort()

    if not sample_files:
        print("No sample images found!")
        return

    # Load and resize images
    images = []
    for file in sample_files[:6]:  # Show first 6 samples
        img = Image.open(file)
        img = img.resize((200, 200))
        images.append(img)

    # Create collage
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (img, file) in enumerate(zip(images, sample_files)):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        # Extract epoch and batch info from filename
        filename = os.path.basename(file)
        axes[i].set_title(f'{filename}', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress collage saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_model_summary(generator, discriminator, save_path='model_summary.txt'):
    """
    Save a summary of the model architectures

    Args:
        generator: Generator model
        discriminator: Discriminator model
        save_path (str): Path to save the summary
    """
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GAN MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("GENERATOR ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        f.write(str(generator) + "\n\n")

        f.write("DISCRIMINATOR ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        f.write(str(discriminator) + "\n\n")

        # Count parameters
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())

        f.write("PARAMETER COUNTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Generator parameters: {gen_params:,}\n")
        f.write(f"Discriminator parameters: {disc_params:,}\n")
        f.write(f"Total parameters: {gen_params + disc_params:,}\n")

    print(f"Model summary saved to {save_path}")
