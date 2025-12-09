"""
Utility functions for data processing.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

try:
    from mace.data.neighborhood import get_neighborhood
except ImportError:
    get_neighborhood = None


def get_neighborhood_graph(
    positions: np.ndarray,
    cutoff: float,
    pbc: tuple,
    cell: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute neighbor graph for periodic structure.
    
    Args:
        positions: Atom positions (N, 3)
        cutoff: Radial cutoff distance
        pbc: Periodic boundary conditions (3,)
        cell: Unit cell matrix (3, 3)
        
    Returns:
        edge_index: Graph connectivity (2, E)
        edge_vec: Edge vectors (E, 3)
        edge_len: Edge lengths (E, 1)
        shifts: Periodic shift vectors (E, 3)
    """
    if get_neighborhood is None:
        raise ImportError("MACE is required for neighborhood computation")
    
    positions = np.asarray(positions, dtype=np.float32)
    cell = np.asarray(cell, dtype=np.float32)
    
    result = get_neighborhood(
        positions=positions,
        cutoff=float(cutoff),
        pbc=pbc,
        cell=cell
    )
    
    if len(result) == 3:
        edge_index_np, shifts_np, unit_shifts = result
    elif len(result) == 4:
        edge_index_np, shifts_np, unit_shifts, _ = result
    else:
        raise ValueError(f"Unexpected neighborhood result: {len(result)} values")
    
    # Convert to torch
    pos = torch.tensor(positions, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    shifts = torch.tensor(shifts_np, dtype=torch.float32)
    
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
    
    return edge_index, edge_vec, edge_len, shifts


def chunk_atoms_with_overlap(
    atoms,
    chunk_size: float = 20.0,
    cutoff: float = 5.0,
) -> List:
    """
    Split large structure into overlapping chunks for memory-efficient processing.
    
    Each chunk includes a buffer region (ghost atoms) around the core region
    to ensure accurate neighbor computations at boundaries.
    
    Args:
        atoms: ASE Atoms object
        chunk_size: Size of core region in each dimension
        cutoff: Buffer/ghost region size (should match model cutoff)
        
    Returns:
        List of ASE Atoms objects (chunks), each with:
            - arrays['is_core']: Boolean mask for core atoms
            - arrays['global_id']: Original atom indices
    """
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    
    lx, ly, lz = cell.lengths()
    nx = max(1, int(np.ceil(lx / chunk_size)))
    ny = max(1, int(np.ceil(ly / chunk_size)))
    nz = max(1, int(np.ceil(lz / chunk_size)))
    
    chunks = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Core region bounds
                x_core_min = ix * chunk_size
                y_core_min = iy * chunk_size
                z_core_min = iz * chunk_size
                x_core_max = min((ix + 1) * chunk_size, lx)
                y_core_max = min((iy + 1) * chunk_size, ly)
                z_core_max = min((iz + 1) * chunk_size, lz)
                
                # Extended region with buffer
                x_min = x_core_min - cutoff
                y_min = y_core_min - cutoff
                z_min = z_core_min - cutoff
                x_max = x_core_max + cutoff
                y_max = y_core_max + cutoff
                z_max = z_core_max + cutoff
                
                # Find atoms in extended region
                mask = (
                    (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) &
                    (positions[:, 1] >= y_min) & (positions[:, 1] < y_max) &
                    (positions[:, 2] >= z_min) & (positions[:, 2] < z_max)
                )
                
                # Find core atoms
                core_mask = (
                    (positions[:, 0] >= x_core_min) & (positions[:, 0] < x_core_max) &
                    (positions[:, 1] >= y_core_min) & (positions[:, 1] < y_core_max) &
                    (positions[:, 2] >= z_core_min) & (positions[:, 2] < z_core_max)
                )
                
                if not mask.any():
                    continue
                    
                selected_indices = np.where(mask)[0]
                core_indices = np.where(core_mask)[0]
                
                chunk = atoms[selected_indices]
                chunk.set_cell(cell)
                chunk.set_pbc(True)
                
                # Mark core vs ghost atoms
                is_core = np.isin(selected_indices, core_indices)
                chunk.arrays['is_core'] = is_core
                chunk.arrays['global_id'] = selected_indices
                
                chunks.append(chunk)
                
    return chunks


def chunk_atoms_by_atom(
    atoms,
    radius: float = 6.0,
    n_jobs: int = 8,
) -> List:
    """
    Create per-atom chunks (like LAMMPS neighbor lists).
    
    Each chunk is centered on one atom and includes all neighbors
    within the given radius. Useful for very large systems.
    
    Args:
        atoms: ASE Atoms object
        radius: Neighbor radius
        n_jobs: Number of parallel workers
        
    Returns:
        List of ASE Atoms objects, one per atom in original structure
    """
    positions = atoms.get_positions()
    tree = cKDTree(positions)
    
    def process_one(i):
        center = positions[i]
        neighbor_indices = tree.query_ball_point(center, radius)
        
        if len(neighbor_indices) == 0:
            return None
            
        chunk = atoms[neighbor_indices]
        chunk.arrays['is_core'] = np.zeros(len(neighbor_indices), dtype=bool)
        chunk.arrays['global_id'] = np.array(neighbor_indices)
        
        # Find center atom in chunk
        core_local_idx = np.where(np.array(neighbor_indices) == i)[0]
        if len(core_local_idx) == 0:
            return None
            
        chunk.arrays['is_core'][core_local_idx[0]] = True
        return chunk
    
    chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_one)(i) for i in range(len(positions))
    )
    
    return [c for c in chunks if c is not None]


def stitch_chunk_predictions(
    chunks: List,
    predictions: List[np.ndarray],
    n_atoms: int,
    mode: str = "mean",
) -> np.ndarray:
    """
    Combine predictions from overlapping chunks.
    
    Args:
        chunks: List of ASE Atoms with 'global_id' and 'is_core' arrays
        predictions: List of prediction arrays, one per chunk
        n_atoms: Total number of atoms in original structure
        mode: How to combine overlapping predictions ("mean", "core_only")
        
    Returns:
        Combined predictions (n_atoms, ...)
    """
    if mode == "core_only":
        # Only use predictions from core regions (no overlap)
        pred_shape = predictions[0].shape[1:] if predictions[0].ndim > 1 else ()
        result = np.zeros((n_atoms,) + pred_shape)
        
        for chunk, pred in zip(chunks, predictions):
            core_mask = chunk.arrays['is_core']
            global_ids = chunk.arrays['global_id'][core_mask]
            result[global_ids] = pred[core_mask]
            
        return result
        
    elif mode == "mean":
        # Average predictions in overlapping regions
        pred_shape = predictions[0].shape[1:] if predictions[0].ndim > 1 else ()
        result = np.zeros((n_atoms,) + pred_shape)
        counts = np.zeros(n_atoms)
        
        for chunk, pred in zip(chunks, predictions):
            global_ids = chunk.arrays['global_id']
            result[global_ids] += pred
            counts[global_ids] += 1
            
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        result = result / counts.reshape(-1, *([1] * len(pred_shape)))
        
        return result
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
