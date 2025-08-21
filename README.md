# MNIST GAN Implementation

A complete implementation of a Generative Adversarial Network (GAN) trained on the MNIST handwritten digits dataset. This project generates realistic synthetic handwritten digits using PyTorch.

## Features

- **DCGAN Architecture**: Deep Convolutional GAN with transposed convolutions
- **MNIST Dataset**: Training on 60,000 handwritten digit images (0-9)
- **Comprehensive Training**: Full training pipeline with progress tracking
- **Visualization Tools**: Generate samples, compare real vs fake images, and create interpolations
- **Model Persistence**: Save and load model checkpoints
- **Configurable Parameters**: Easy to adjust training parameters via command line

## Project Structure

```
├── main.py                 # Main training script with argument parsing
├── data_loader.py          # MNIST dataset loading and preprocessing
├── generator.py            # Generator network architecture
├── discriminator.py        # Discriminator network architecture
├── train_utils.py          # Training utilities and GAN trainer class
├── visualization.py        # Visualization and evaluation utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── checkpoints/           # Saved model weights (created during training)
└── generated_images/      # Generated samples and plots (created during training)
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- tqdm
- pillow

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have a CUDA-capable GPU for faster training (optional but recommended)

## Usage

### Basic Training

Run the GAN training with default parameters:

```bash
python main.py
```

This will train for 50 epochs with a batch size of 128 and latent dimension of 100.

### Custom Training Parameters

```bash
python main.py \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --lr 0.0001 \
    --save_interval 5 \
    --sample_interval 50
```

### Available Arguments

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 128)
- `--latent_dim`: Dimension of latent space (default: 100)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Beta1 parameter for Adam optimizer (default: 0.5)
- `--save_interval`: Epoch interval for saving checkpoints (default: 10)
- `--sample_interval`: Batch interval for generating sample images (default: 100)
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')

## Architecture Details

### Generator

- Takes random noise of dimension `latent_dim` (default: 100)
- Uses transposed convolutional layers to upsample to 28x28 images
- Output range: [-1, 1] (tanh activation)
- Architecture: 512 → 256 → 128 → 64 → 1 channels

### Discriminator

- Takes 28x28 grayscale images as input
- Uses convolutional layers with LeakyReLU activations
- Output: probability of image being real (0-1)
- Architecture: 1 → 64 → 128 → 256 → 512 → 1

### Training

- Uses Binary Cross Entropy loss
- Adam optimizer with learning rate 0.0002 and beta1=0.5
- Alternating training: discriminator → generator
- Real labels smoothed to 0.9 for better stability

## Output Files

### Generated Images (`generated_images/`)

- `samples_epoch_X_batch_Y.png`: Sample images generated during training
- `training_losses.png`: Plot of discriminator and generator losses
- `final_samples.png`: Final generated samples (16 images in a grid)
- `real_vs_fake_comparison.png`: Side-by-side comparison of real and fake images
- `interpolation_sequence.png`: Smooth interpolation between two random digits
- `training_progress.png`: Collage showing training progression

### Model Checkpoints (`checkpoints/`)

- `checkpoint_epoch_X.pth`: Full model state at epoch X
- `generator_final.pth`: Final generator weights
- `discriminator_final.pth`: Final discriminator weights
- `model_summary.txt`: Architecture summary and parameter counts

## Results

After training, you should expect to see:

- Realistic handwritten digits that resemble MNIST samples
- Smooth interpolation between different digit styles
- Progressive improvement in image quality throughout training
- Training convergence with discriminator and generator losses stabilizing

## Tips for Better Results

1. **Training Time**: GANs can take time to converge. Start with 50-100 epochs
2. **Batch Size**: Larger batch sizes (128+) can lead to more stable training
3. **Latent Dimension**: 100-128 is typically sufficient for MNIST
4. **Learning Rate**: 0.0002 is standard for GANs, but you can experiment with 0.0001-0.0003
5. **Sample Inspection**: Regularly check generated samples to monitor training progress

## Troubleshooting

- **CUDA Issues**: If you get CUDA errors, try running with `--device cpu`
- **Memory Issues**: Reduce batch size if you run out of GPU memory
- **Training Instability**: GANs can be unstable. Try reducing learning rate or increasing batch size
- **Poor Results**: If generated images are poor, try training for more epochs or adjusting the latent dimension

## References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - Ian Goodfellow et al.
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) - Alec Radford et al.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Yann LeCun et al.
