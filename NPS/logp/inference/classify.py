"""
Classification functions for crystal structures.

These functions use the log-probability model to assign
phase labels and compute per-atom, per-class probabilities.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms

from ..data.utils import get_neighborhood_graph
from .denoise import denoise_snapshot


@torch.no_grad()
def get_phase_probabilities(
    atoms: Atoms,
    model: torch.nn.Module,
    cutoff: float = 6.0,
    device: str = None,
) -> np.ndarray:
    """
    Compute per-atom, per-class log-probabilities without denoising.
    
    Args:
        atoms: ASE Atoms object
        model: Trained LogP model
        cutoff: Radial cutoff for graph construction
        device: Computation device
        
    Returns:
        logP: Log-probabilities array (N, nclass)
    """
    species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.positions.copy(), dtype=torch.float32)
    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float32)
    
    # Auto-detect device if not specified
    # Note: MPS (Apple Silicon) doesn't support float64, which MACE requires
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"  # MPS not supported due to float64 in MACE
    
    model = model.to(device)
    model.eval()
    
    # Build graph
    edge_index, edge_vec, edge_len, shifts = get_neighborhood_graph(
        positions=pos.numpy(),
        cutoff=cutoff,
        pbc=atoms.pbc,
        cell=cell.numpy(),
    )
    
    # Create batch
    batch = Data(
        z=species,
        x=species,
        species=species,
        pos=pos,
        positions=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_attr=edge_len,
        shifts=shifts,
        cell=cell,
        ptr=torch.tensor([0, len(atoms)]),
        batch=torch.zeros(len(species), dtype=torch.long),
    ).to(device)
    
    # Forward pass
    logp, _ = model(batch)
    
    return logp.detach().cpu().numpy()


@torch.no_grad()
def classify_zeroshot(
    atoms: Atoms,
    model: torch.nn.Module,
    structure_types: List[str],
    cutoff: float = 6.0,
    device: str = None,
) -> Tuple[np.ndarray, float]:
    """
    Zero-shot classification without denoising.
    
    Args:
        atoms: ASE Atoms object
        model: Trained LogP model
        structure_types: List of structure type names
        cutoff: Radial cutoff
        device: Computation device
        
    Returns:
        predictions: Per-atom predicted class indices (N,)
        confidence: Mean max log-probability (confidence score)
    """
    logp = get_phase_probabilities(atoms, model, cutoff, device)
    
    predictions = np.argmax(logp, axis=1)
    confidence = np.mean(np.max(logp, axis=1))
    
    return predictions, confidence


@torch.no_grad()
def classify_structure(
    atoms: Atoms,
    model: torch.nn.Module,
    structure_types: List[str],
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 0.15,
    device: str = None,
    return_logp: bool = False,
) -> Dict:
    """
    Classify crystal structure with optional denoising.
    
    Performs iterative denoising and returns classification results
    based on the final log-probabilities.
    
    Args:
        atoms: ASE Atoms object
        model: Trained LogP model
        structure_types: List of structure type names
        steps: Number of denoising iterations
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        device: Computation device
        return_logp: If True, include full logP trajectory
        
    Returns:
        Dictionary containing:
            - predictions: Per-atom class indices (N,)
            - labels: Per-atom class names (N,)
            - accuracy: Fraction assigned to majority class
            - majority_class: Most common predicted class
            - confidence: Mean max log-probability
            - logp: Final log-probabilities (if return_logp=True)
            - logp_trajectory: All logP values (if return_logp=True)
    """
    # Run denoising
    pos_traj, logp_traj = denoise_snapshot(
        atoms=atoms,
        model=model,
        steps=steps,
        cutoff=cutoff,
        sigma_scale=sigma_scale,
        device=device,
    )
    
    # Get final predictions
    final_logp = logp_traj[-1]
    predictions = np.argmax(final_logp, axis=1)
    
    # Compute statistics
    unique, counts = np.unique(predictions, return_counts=True)
    majority_idx = unique[np.argmax(counts)]
    majority_class = structure_types[majority_idx]
    accuracy = np.max(counts) / len(predictions)
    confidence = np.mean(np.max(final_logp, axis=1))
    
    # Map to labels
    labels = np.array([structure_types[p] for p in predictions])
    
    result = {
        "predictions": predictions,
        "labels": labels,
        "accuracy": accuracy,
        "majority_class": majority_class,
        "confidence": confidence,
    }
    
    if return_logp:
        result["logp"] = final_logp
        result["logp_trajectory"] = logp_traj
        
    return result


def classify_trajectory(
    trajectory: List[Atoms],
    model: torch.nn.Module,
    structure_types: List[str],
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 0.15,
    device: str = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Classify a trajectory of structures.
    
    Args:
        trajectory: List of ASE Atoms objects
        model: Trained LogP model
        structure_types: List of structure type names
        steps: Number of denoising iterations per frame
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        device: Computation device
        verbose: Print progress
        
    Returns:
        List of classification result dictionaries
    """
    results = []
    
    for i, atoms in enumerate(trajectory):
        if verbose and i % 10 == 0:
            print(f"Classifying frame {i}/{len(trajectory)}")
            
        result = classify_structure(
            atoms=atoms,
            model=model,
            structure_types=structure_types,
            steps=steps,
            cutoff=cutoff,
            sigma_scale=sigma_scale,
            device=device,
        )
        results.append(result)
        
    return results


def compute_order_parameters(
    atoms: Atoms,
    model: torch.nn.Module,
    structure_types: List[str],
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 0.15,
    device: str = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-atom order parameters for each phase.
    
    The log-probability for each phase serves as a continuous
    order parameter measuring similarity to that phase.
    
    Args:
        atoms: ASE Atoms object
        model: Trained LogP model
        structure_types: List of structure type names
        steps: Number of denoising iterations
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        device: Computation device
        
    Returns:
        Dictionary mapping phase names to per-atom log-P values
    """
    _, logp_traj = denoise_snapshot(
        atoms=atoms,
        model=model,
        steps=steps,
        cutoff=cutoff,
        sigma_scale=sigma_scale,
        device=device,
    )
    
    final_logp = logp_traj[-1]
    
    order_parameters = {}
    for i, name in enumerate(structure_types):
        order_parameters[name] = final_logp[:, i]
        
    # Also include max and confidence
    order_parameters["max_logp"] = np.max(final_logp, axis=1)
    order_parameters["predicted_phase"] = np.array([
        structure_types[p] for p in np.argmax(final_logp, axis=1)
    ])
    
    return order_parameters
