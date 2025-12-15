# NPS: Neural Phase Structures

A machine learning framework for crystal structure analysis, including denoising, phase classification, and order parameter estimation.

## Features

### LogP Foundation Model (`NPS.logp`)

A unified framework for crystal structure analysis based on log-probability modeling:

- **Denoising**: Remove thermal noise from MD snapshots via gradient ascent in log-probability space
- **Phase Classification**: Assign per-atom phase labels across hundreds of AFLOW prototypes
- **Order Parameters**: Continuous, defect-sensitive OPs derived from per-phase logits $l_{ac}$

Key advantages over traditional methods (PTM, CNA):
- Universal: Works with any AFLOW prototype, not just FCC/BCC/HCP
- Robust: Accurate classification even at melting temperatures
- Probabilistic: Per-atom confidence scores expose ambiguity
- Interpretable: Logits measure squared distance to ideal prototypes

## Installation

```bash
# Clone repository
git clone https://github.com/kha8128/NPS.git
cd NPS

# Install (requires Python 3.9+)
pip install -e .

# Download pre-trained checkpoint (~244 MB)
python scripts/download_checkpoint.py
```

See [INSTALL.md](INSTALL.md) for detailed instructions including GPU setup.

## Pre-trained Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| `logp_strained_scaled_elemental` | 244 MB | 40 AFLOW prototypes with strain augmentation (**recommended**) |
| `logp_scaled_elemental` | 244 MB | 40 AFLOW prototypes without strain augmentation |
| `logp_scaled_elemental_binary` | 244 MB | 40 elemental + 363 binary prototypes |

Checkpoints are hosted on [GitHub Releases](https://github.com/kha8128/NPS/releases) due to size.

```bash
# Download recommended checkpoint
python scripts/download_checkpoint.py --name logp_strained_scaled_elemental

# Download all checkpoints
python scripts/download_checkpoint.py --all

# List available checkpoints
python scripts/download_checkpoint.py --list
```

## Quick Start

### Classify a Crystal Structure

```python
from NPS.logp import classify_structure
from NPS.logp.models import LitLogPModel
from NPS.logp.constants import STRUCTURE_TYPES
from ase.build import bulk

# Load pre-trained model
model = LitLogPModel.load_from_checkpoint(
    "checkpoints/logp_strained_scaled_elemental.ckpt",
    map_location='cpu'
).get_inference_model()

# Create or load structure
atoms = bulk('Fe', 'bcc', a=2.87, cubic=True) * (3,3,3)

# Classify with denoising
result = classify_structure(
    atoms, model,
    structure_types=STRUCTURE_TYPES,
    steps=8,
    device='cpu'
)

print(f"Phase: {result['majority_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Denoise a Thermal MD Snapshot

```python
from NPS.logp import denoise_structure
import numpy as np

# Add thermal noise to simulate MD snapshot
noisy_atoms = atoms.copy()
noisy_atoms.positions += np.random.normal(0, 0.1, atoms.positions.shape)

# Remove thermal noise (8 iterations)
denoised = denoise_structure(noisy_atoms, model, steps=8, device='cpu')

# Check improvement
print(f"Noise RMSD: {np.sqrt(np.mean((noisy_atoms.positions - atoms.positions)**2)):.3f} $\AA$")
print(f"After denoise: {np.sqrt(np.mean((denoised.positions - atoms.positions)**2)):.3f} $\AA$")
```

### Compute Order Parameters

```python
from NPS.logp.inference import compute_order_parameters

ops = compute_order_parameters(
    atoms, model,
    structure_types=STRUCTURE_TYPES,
    device='cpu'
)

# ops["A_cI2_229"] contains per-atom BCC logits (order parameter)
# ops["max_logp"] contains max logit for each atom (confidence)
```

### Command-Line Interface

```bash
# Train a model (using sample data)
python -m NPS.logp.scripts.train \
    --train_data "examples/sample_data/*.extxyz" \
    --structure_types "A_cI2_229,A_cF4_225,A_hP2_194" \
    --batch_size 8

# Run inference
python -m NPS.logp.scripts.infer \
    --checkpoint model.ckpt \
    --input structure.extxyz \
    --structure_types "A_cI2_229,A_cF4_225,A_hP2_194" \
    --mode classify
```

## Documentation

- [Main NPS Repository](../../README.md) - Detailed API documentation
- [Installation Guide](../../INSTALL.md) - Setup instructions for various environments
- [Examples](../../examples/) - Jupyter notebooks and tutorials

## Citation

If you use this code, please cite:

```bibtex
@article{kwon2025logp,
  title={A probabilistic foundation model for crystal structure 
         denoising, phase classification, and order parameters},
  author={Kwon, Hyuna and Sadigh, Babak and Hamel, Sebastien and 
          Lordi, Vincenzo and Klepeis, John and Zhou, Fei},
  journal={arXiv preprint arXiv:2512.11077},
  year={2025},
  url={https://arxiv.org/abs/2512.11077}
}
```

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
