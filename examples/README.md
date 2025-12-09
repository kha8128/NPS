# NPS-LogP Examples

This directory contains Jupyter notebooks demonstrating how to use the LogP foundation model.

## Notebooks

### [01_quick_start.ipynb](01_quick_start.ipynb)
**Getting started with NPS-LogP**

Basic usage including:
- Loading/creating a model
- Denoising crystal structures
- Classifying phases
- Computing order parameters

### [02_training_minimal.ipynb](02_training_minimal.ipynb)
**Training your own model**

Complete training workflow:
- Preparing training data
- Configuring model and data loaders
- Running training with PyTorch Lightning
- Evaluating and saving the trained model

### [03_finetuning_from_mace_mp.ipynb](03_finetuning_from_mace_mp.ipynb)
**Fine-tuning from pretrained MACE weights**

Transfer learning workflow:
- Loading pretrained MACE checkpoints
- Matching model architecture to pretrained weights
- Fine-tuning with frozen or trainable backbone
- Faster convergence with pretrained initialization

## Requirements

To run these notebooks, you need:

```bash
# Install NPS
pip install -e /path/to/NPS

# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

## Data Files

Some notebooks generate example data files:
- `training_data/` - Reference structures for training
- `*.extxyz` - Output structures with analysis

## Tips

1. **GPU recommended**: Training and inference are much faster on GPU
2. **Memory**: Large structures may require chunking (see `denoise_large_structure`)
3. **Pre-trained models**: For best results, fine-tune from MACE-MP weights

## Additional Resources

- [Main README](../README.md) - Installation and overview
- [LogP Module Docs](../NPS/logp/README.md) - Detailed API documentation
- [Paper]() - Method description and benchmarks
