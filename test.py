"""
Test script for conditional DCGAN digit generation
Loads pre-trained weights and generates images based on user input labels
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

# Model parameters (must match training)
z_dim = 100
num_classes = 10
channels = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

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

def load_generator_weights(generator, weights_path='saved_weights/generator_weights.pth'):
    """Load generator weights for inference."""
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

def generate_single_digit(generator, digit, device, z_dim, save_path="generated_test.png"):
    """
    Generate a single image of a specific digit.

    Args:
        generator: The generator model
        digit: The digit to generate (0-9)
        device: Device to run on
        z_dim: Dimension of noise vector
        save_path: Path to save the generated image
    """
    generator.eval()
    with torch.no_grad():
        # Create label for the specific digit
        label = torch.tensor([digit], dtype=torch.long).to(device)
        # Generate noise
        z = torch.randn(1, z_dim, 1, 1).to(device)
        # Generate image
        generated = generator(z, label)

        # Save the generated image
        save_image(generated[0], save_path, normalize=True)
        print(f"Generated digit {digit}, saved to {save_path}")

        # Also display the image
        img = generated[0].cpu().squeeze().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray')
        plt.title(f'Generated Digit: {digit}')
        plt.axis('off')
        plt.show()

    generator.train()

def generate_multiple_samples(generator, digit, device, z_dim, num_samples=8, save_path="generated_samples.png"):
    """
    Generate multiple samples of a specific digit.

    Args:
        generator: The generator model
        digit: The digit to generate (0-9)
        device: Device to run on
        z_dim: Dimension of noise vector
        num_samples: Number of samples to generate
        save_path: Path to save the generated image grid
    """
    generator.eval()
    with torch.no_grad():
        # Create labels for the specific digit
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        # Generate noise
        z = torch.randn(num_samples, z_dim, 1, 1).to(device)
        # Generate images
        generated = generator(z, labels)

        # Save the generated images as a grid
        save_image(generated, save_path, nrow=int(np.sqrt(num_samples)), normalize=True)
        print(f"Generated {num_samples} samples of digit {digit}, saved to {save_path}")

        # Display the grid
        img_grid = generated.cpu()
        grid_img = torch.cat([img_grid[i] for i in range(min(16, num_samples))], dim=2)
        if num_samples > 16:
            grid_img = torch.cat([grid_img, torch.cat([img_grid[i] for i in range(16, min(32, num_samples))], dim=2)], dim=1)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.squeeze().numpy(), cmap='gray')
        plt.title(f'Generated Samples of Digit: {digit}')
        plt.axis('off')
        plt.show()

    generator.train()

def main():
    # Initialize generator
    generator = Generator(z_dim, num_classes=num_classes).to(device)

    # Load weights
    if not load_generator_weights(generator):
        print("Failed to load weights. Please ensure the weights file exists.")
        return

    print("Generator loaded successfully!")
    print("\nCommands:")
    print("- Enter a digit (0-9) to generate a single image")
    print("- Enter 'samples X' to generate X samples of a digit (e.g., 'samples 5' for digit 5)")
    print("- Enter 'quit' or 'exit' to quit")

    while True:
        try:
            user_input = input("\nEnter a digit (0-9) or command: ").strip().lower()

            if user_input in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            elif user_input.startswith('samples '):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit():
                    digit = int(parts[1])
                    if 0 <= digit <= 9:
                        num_samples = 8  # Default number of samples
                        save_path = f"generated_digit_{digit}_samples.png"
                        generate_multiple_samples(generator, digit, device, z_dim, num_samples, save_path)
                    else:
                        print("Please enter a digit between 0 and 9")
                else:
                    print("Invalid format. Use 'samples X' where X is a digit 0-9")

            elif user_input.isdigit():
                digit = int(user_input)
                if 0 <= digit <= 9:
                    save_path = f"generated_digit_{digit}.png"
                    generate_single_digit(generator, digit, device, z_dim, save_path)
                else:
                    print("Please enter a digit between 0 and 9")

            else:
                print("Invalid input. Please enter a digit (0-9), 'samples X', or 'quit'")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
