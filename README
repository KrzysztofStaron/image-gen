# Handwritten Digit Generator (DCGAN)

A DCGAN implementation that generates realistic handwritten digits using the MNIST dataset.

## The Bug That Broke Everything

I made a simple but devastating mistake: my MNIST images were 28×28, but my generator was producing 32×32 images. This tiny 4-pixel difference meant the discriminator could **perfectly** distinguish real from fake every time, completely preventing the generator from learning.

### Before Fix (Broken)

![Before](old.png)
_Generator produced meaningless noise_

### After Fix (Working)

![After](final_generated.png)
_Generator produces recognizable digits_

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run training
python main.py
```

## What It Does

- Downloads MNIST dataset automatically
- Trains generator and discriminator networks
- Saves progress images every 5 epochs
- Generates final samples after training

## Key Lesson

**Image size matching is absolutely critical** - even a 4-pixel mismatch can break GAN training completely.
