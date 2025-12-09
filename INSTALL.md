# Installation Guide

## Quick Install

```bash
# Clone the repository
git clone https://github.com/Critical-Materials-Institute/NPS.git
cd NPS

# Install in editable mode
pip install -e .
```

## Detailed Installation

### Prerequisites

- Python 3.9 - 3.12
- CUDA 11.8+ (for GPU support, optional but recommended)

### Option 1: pip (Recommended)

```bash
# 1. Create a virtual environment
python -m venv nps-env
source nps-env/bin/activate  # Linux/Mac
# or: nps-env\Scripts\activate  # Windows

# 2. Install PyTorch first (with appropriate CUDA version)
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch_geometric

# 4. Clone and install NPS
git clone https://github.com/Critical-Materials-Institute/NPS.git
cd NPS
pip install -e .
```

### Option 2: Conda

```bash
# 1. Create conda environment from file
conda env create -f environment.yml

# 2. Activate
conda activate nps-logp

# 3. Install NPS in editable mode
pip install -e .
```

### Option 3: Manual Installation

If you have issues with the above, install dependencies manually:

```bash
# Core dependencies
pip install numpy scipy joblib tqdm

# PyTorch (adjust for your CUDA version)
pip install torch

# PyTorch Geometric
pip install torch_geometric

# E3NN and MACE
pip install e3nn mace-torch

# Lightning for training
pip install lightning

# ASE for atomic structure handling
pip install ase

# Visualization (optional)
pip install matplotlib seaborn
```

## Verify Installation

```python
# Test basic import
from NPS.logp import denoise_structure, classify_structure
from NPS.logp.models import LitLogPModel, LogPModelWrapper

print("Installation successful!")
```

## Common Issues

### 1. CUDA/PyTorch Mismatch

If you get CUDA errors, ensure your PyTorch version matches your CUDA version:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### 2. PyTorch Geometric Installation Fails

Try installing with pre-built wheels:

```bash
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Adjust the URL for your PyTorch and CUDA versions.

### 3. MACE Installation Issues

MACE requires a C++ compiler. On Ubuntu:

```bash
sudo apt-get install build-essential
```

On Mac:

```bash
xcode-select --install
```

### 4. Import Errors

Make sure you're in the right environment and NPS is installed:

```bash
pip list | grep nps
pip install -e /path/to/NPS
```

## GPU Memory Requirements

| Model Size | Approximate GPU Memory |
|------------|------------------------|
| Default (128 hidden) | ~4 GB |
| Large (256 hidden) | ~8 GB |
| Training (batch=8) | ~12 GB |

For large structures (>10,000 atoms), use the chunking feature:

```python
from NPS.logp.inference import denoise_large_structure
result = denoise_large_structure(atoms, model, chunk_size=20.0)
```

## HPC Installation (SLURM/PBS)

Example module setup for typical HPC systems:

```bash
module load python/3.10
module load cuda/12.1
module load gcc/11.2

# Create venv in your scratch space
python -m venv $SCRATCH/nps-env
source $SCRATCH/nps-env/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric lightning mace-torch ase e3nn

cd /path/to/NPS
pip install -e .
```

## Development Installation

For contributing to NPS:

```bash
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black NPS/
isort NPS/
```
