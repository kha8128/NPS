"""
Denoising functions for crystal structures.

These functions take noisy atomic configurations and iteratively
refine them using the learned log-probability gradient (score).
"""

from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
import ase
from ase import Atoms

from ..data.utils import get_neighborhood_graph


@torch.no_grad()
def denoise_snapshot(
    atoms: Atoms,
    model: torch.nn.Module,
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 1.0,
    device: str = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Denoise a single atomic configuration.
    
    Performs iterative gradient ascent in log-probability space
    to remove thermal noise from atom positions.
    
    Args:
        atoms: ASE Atoms object (will not be modified)
        model: Trained LogP model wrapper
        steps: Number of denoising iterations
        cutoff: Radial cutoff for graph construction
        sigma_scale: Scale factor for step size (related to training sigma)
        device: Device to run inference on (None for auto-detect)
        
    Returns:
        pos_traj: List of position arrays, one per denoising step
        logp_traj: List of log-probability arrays (N, nclass) per step
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Get atomic data
    species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.positions.copy(), dtype=torch.float32)
    cell = torch.tensor(atoms.cell.array.copy(), dtype=torch.float32)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    pos_traj = [atoms.positions.copy()]
    logp_traj = []
    
    for step in range(steps):
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
            pos=pos.clone(),
            positions=pos.clone(),
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_attr=edge_len,
            shifts=shifts,
            cell=cell,
            ptr=torch.tensor([0, len(atoms)]),
            batch=torch.zeros(len(species), dtype=torch.long),
        ).to(device)
        
        # Forward pass
        logp, score = model(batch)
        
        # Store log-probabilities
        logp_traj.append(logp.detach().cpu().numpy())
        
        # Update positions: gradient ascent in log-probability space
        # score = d(logP)/dr, displacement points toward higher logP
        # Scale by sigma^2 to match training noise level
        displacement = score * sigma_scale**2
        
        pos = pos.cpu() + displacement.cpu()  # + not -
        pos_traj.append(pos.numpy().copy())
        
    return pos_traj, logp_traj


def denoise_structure(
    atoms: Atoms,
    model: torch.nn.Module,
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 1.0,
    device: str = None,
    return_trajectory: bool = False,
) -> Union[Atoms, Tuple[Atoms, List[Atoms]]]:
    """
    Denoise a structure and return the cleaned result.
    
    Args:
        atoms: Input ASE Atoms object
        model: Trained LogP model
        steps: Number of denoising iterations
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        device: Computation device
        return_trajectory: If True, also return intermediate structures
        
    Returns:
        Denoised Atoms object (and optionally full trajectory)
    """
    pos_traj, logp_traj = denoise_snapshot(
        atoms=atoms,
        model=model,
        steps=steps,
        cutoff=cutoff,
        sigma_scale=sigma_scale,
        device=device,
    )
    
    # Create denoised structure
    denoised = atoms.copy()
    denoised.positions = pos_traj[-1]
    
    if return_trajectory:
        trajectory = []
        for pos in pos_traj:
            frame = atoms.copy()
            frame.positions = pos
            trajectory.append(frame)
        return denoised, trajectory
    
    return denoised


def denoise_trajectory(
    trajectory: List[Atoms],
    model: torch.nn.Module,
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 1.0,
    device: str = None,
    verbose: bool = True,
) -> List[Atoms]:
    """
    Denoise a trajectory of structures.
    
    Args:
        trajectory: List of ASE Atoms objects (e.g., from MD simulation)
        model: Trained LogP model
        steps: Number of denoising iterations per frame
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        device: Computation device
        verbose: Print progress
        
    Returns:
        List of denoised Atoms objects
    """
    denoised_traj = []
    
    for i, atoms in enumerate(trajectory):
        if verbose and i % 10 == 0:
            print(f"Denoising frame {i}/{len(trajectory)}")
            
        denoised = denoise_structure(
            atoms=atoms,
            model=model,
            steps=steps,
            cutoff=cutoff,
            sigma_scale=sigma_scale,
            device=device,
        )
        denoised_traj.append(denoised)
        
    return denoised_traj


@torch.no_grad()
def denoise_large_structure(
    atoms: Atoms,
    model: torch.nn.Module,
    steps: int = 8,
    cutoff: float = 6.0,
    sigma_scale: float = 1.0,
    chunk_size: float = 20.0,
    device: str = None,
) -> Atoms:
    """
    Denoise a large structure by processing chunks.
    
    For structures too large to fit in GPU memory, this function
    splits them into overlapping chunks, denoises each chunk,
    and stitches the results together.
    
    Args:
        atoms: Large ASE Atoms object
        model: Trained LogP model
        steps: Number of denoising iterations
        cutoff: Radial cutoff
        sigma_scale: Step size scale factor
        chunk_size: Size of each chunk's core region
        device: Computation device
        
    Returns:
        Denoised Atoms object
    """
    from ..data.utils import chunk_atoms_with_overlap, stitch_chunk_predictions
    
    # Split into chunks
    chunks = chunk_atoms_with_overlap(atoms, chunk_size=chunk_size, cutoff=cutoff)
    print(f"Split into {len(chunks)} chunks")
    
    # Denoise each chunk
    all_positions = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        pos_traj, _ = denoise_snapshot(
            atoms=chunk,
            model=model,
            steps=steps,
            cutoff=cutoff,
            sigma_scale=sigma_scale,
            device=device,
        )
        
        # Get final positions
        all_positions.append(pos_traj[-1])
    
    # Stitch together (only use core atom positions)
    final_positions = stitch_chunk_predictions(
        chunks=chunks,
        predictions=all_positions,
        n_atoms=len(atoms),
        mode="core_only",
    )
    
    # Create result
    result = atoms.copy()
    result.positions = final_positions
    
    return result
