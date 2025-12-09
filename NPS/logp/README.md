# NPS-LogP: Log-Probability Foundation Model for Crystal Structures

A unified framework for crystal structure **denoising**, **phase classification**, and **order parameter estimation** based on log-probability modeling.

## Overview

This module implements the method described in:

> Kwon et al., "A log-probability foundation model for crystal structure denoising, phase classification, and order parameters"

The key idea is to train a model that predicts per-atom, per-class log-probabilities $\log P_{ac}$ for each atom $a$ and crystal class $c$. From these:

- **Denoising**: Gradient ascent in log-probability space removes thermal noise
- **Classification**: `argmax` over classes gives phase labels
- **Order Parameters**: The $\log P$ values themselves serve as continuous OPs

## Installation

```bash
# Install NPS (if not already)
pip install -e /path/to/NPS

# Required dependencies
pip install torch torch_geometric lightning mace-torch ase
```

## Quick Start

### Inference with Pre-trained Model

```python
from NPS.logp import denoise_structure, classify_structure
from NPS.logp.models import LitLogPModel
import ase.io

# Load model
lit_model = LitLogPModel.load_from_checkpoint("model.ckpt")
model = lit_model.get_inference_model()

# Load structure
atoms = ase.io.read("noisy_structure.extxyz")

# Denoise
denoised = denoise_structure(atoms, model, steps=8)

# Classify
result = classify_structure(
    atoms, model, 
    structure_types=["bcc", "fcc", "hcp"],
    steps=8
)
print(f"Predicted phase: {result['majority_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Training a New Model

```bash
python -m NPS.logp.scripts.train \
    --train_data "data/structures/*.extxyz" \
    --structure_types "bcc,fcc,hcp,omega" \
    --batch_size 8 \
    --lr 1e-4 \
    --max_steps 100000 \
    --output_dir ./training
```

For distributed training:

```bash
torchrun --nproc_per_node=4 -m NPS.logp.scripts.train \
    --train_data "data/structures/*.extxyz" \
    --structure_types "bcc,fcc,hcp" \
    --gpus_per_node 4
```

### Command-Line Inference

```bash
# Classify structures
python -m NPS.logp.scripts.infer \
    --checkpoint model.ckpt \
    --input "structures/*.extxyz" \
    --structure_types "bcc,fcc,hcp" \
    --mode classify

# Denoise a trajectory
python -m NPS.logp.scripts.infer \
    --checkpoint model.ckpt \
    --input trajectory.extxyz \
    --mode denoise \
    --output denoised.extxyz

# Compute order parameters
python -m NPS.logp.scripts.infer \
    --checkpoint model.ckpt \
    --input structure.extxyz \
    --structure_types "bcc,fcc,hcp" \
    --mode order_params
```

## Module Structure

```
NPS/logp/
├── __init__.py              # Main exports
├── README.md                # This file
│
├── models/
│   ├── mace_denoiser.py     # MACE-based denoiser architecture
│   ├── wrappers.py          # Model wrapper with logP computation
│   └── lightning_module.py  # PyTorch Lightning training module
│
├── data/
│   ├── datasets.py          # PyG datasets for crystal structures
│   ├── datamodules.py       # Lightning DataModules
│   └── utils.py             # Chunking, neighbor lists, etc.
│
├── inference/
│   ├── denoise.py           # Denoising functions
│   └── classify.py          # Classification functions
│
├── utils/
│   ├── graph_utils.py       # Graph construction utilities
│   └── visualization.py     # Plotting utilities
│
└── scripts/
    ├── train.py             # Training CLI
    └── infer.py             # Inference CLI
```

## API Reference

### Inference Functions

```python
# Denoise a single structure
from NPS.logp import denoise_structure
denoised = denoise_structure(atoms, model, steps=8)

# Denoise a trajectory
from NPS.logp import denoise_trajectory
denoised_traj = denoise_trajectory(trajectory, model, steps=8)

# Classify structure
from NPS.logp import classify_structure
result = classify_structure(atoms, model, structure_types, steps=8)

# Zero-shot classification (no denoising)
from NPS.logp import classify_zeroshot
predictions, confidence = classify_zeroshot(atoms, model, structure_types)
```

### Model Creation

```python
from NPS.logp.models import LogPModelWrapper, LitLogPModel

# For inference
model = LogPModelWrapper(
    num_species=89,
    nclass=4,
    cutoff=6.0,
    hidden_irreps="128x0e+128x1o+128x2e",
)

# For training
lit_model = LitLogPModel(
    nclass=4,
    cutoff=6.0,
    sigma_max=0.15,
    wt_classification=1.0,
    learn_rate=1e-4,
)
```

### Data Loading

```python
from NPS.logp.data import StrainedPeriodicStructureDataModule

datamodule = StrainedPeriodicStructureDataModule(
    file_list=["struct1.extxyz", "struct2.extxyz"],
    cutoff=6.0,
    structure_types=["bcc", "fcc", "hcp"],
    batch_size=8,
)
```

## Training Data Format

Training data should be provided as structure files (extxyz, cif, POSCAR, etc.) with one file per prototype. The filename stem is used as the structure type label:

```
data/
├── bcc.extxyz       # BCC prototype
├── fcc.extxyz       # FCC prototype
├── hcp.extxyz       # HCP prototype
└── omega.extxyz     # Omega prototype
```

## Citation

If you use this code, please cite:

```bibtex
@article{kwon2025logp,
  title={A log-probability foundation model for crystal structure denoising, 
         phase classification, and order parameters},
  author={Kwon, Hyuna and Sadigh, Babak and Hamel, Sebastien and 
          Lordi, Vincenzo and Klepeis, John and Zhou, Fei},
  journal={},
  year={2025}
}
```

## License

This code is released under the same license as the NPS repository. See the main LICENSE file for details.
