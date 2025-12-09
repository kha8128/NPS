"""
Graph construction utilities for atomic structures.
"""

from typing import Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from ase import Atoms

try:
    from mace.data.neighborhood import get_neighborhood
except ImportError:
    get_neighborhood = None


def get_graph_from_atoms(
    atoms: Atoms,
    cutoff: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct graph representation from ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object with positions, cell, and pbc
        cutoff: Radial cutoff distance for edges
        
    Returns:
        edge_index: Graph connectivity (2, E)
        edge_vec: Edge vectors including periodic shifts (E, 3)
        edge_len: Edge lengths (E, 1)
        shifts: Periodic shift vectors (E, 3)
    """
    if get_neighborhood is None:
        raise ImportError(
            "MACE is required for graph construction. "
            "Install with: pip install mace-torch"
        )
    
    positions = np.asarray(atoms.positions, dtype=np.float32)
    cell = np.asarray(atoms.cell.array, dtype=np.float32)
    
    result = get_neighborhood(
        positions=positions,
        cutoff=float(cutoff),
        pbc=atoms.pbc,
        cell=cell
    )
    
    if len(result) == 3:
        edge_index_np, shifts_np, _ = result
    elif len(result) == 4:
        edge_index_np, shifts_np, _, _ = result
    else:
        raise ValueError(f"Unexpected neighborhood result length: {len(result)}")
    
    # Convert to torch tensors
    pos = torch.tensor(positions, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    shifts = torch.tensor(shifts_np, dtype=torch.float32)
    
    # Compute edge vectors and lengths
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
    
    return edge_index, edge_vec, edge_len, shifts


def atoms_to_data(
    atoms: Atoms,
    cutoff: float = 6.0,
    structure_type: Optional[int] = None,
) -> Data:
    """
    Convert ASE Atoms to PyTorch Geometric Data object.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Radial cutoff for graph construction
        structure_type: Optional structure type label
        
    Returns:
        PyG Data object ready for model input
    """
    edge_index, edge_vec, edge_len, shifts = get_graph_from_atoms(atoms, cutoff)
    
    species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
    
    data = Data(
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
    )
    
    if structure_type is not None:
        data.structure_type = torch.full_like(species, structure_type)
        
    return data


def filter_edges_by_cutoff(
    data: Data,
    cutoff: float,
) -> Data:
    """
    Remove edges longer than the specified cutoff.
    
    Useful when data was constructed with a larger cutoff
    and needs to be filtered for a model with smaller cutoff.
    
    Args:
        data: PyG Data object
        cutoff: Maximum edge length to keep
        
    Returns:
        Filtered Data object
    """
    edge_len = torch.linalg.norm(data.edge_vec, dim=1)
    mask = edge_len < cutoff
    
    data = data.clone()
    data.edge_index = data.edge_index[:, mask]
    data.edge_attr = data.edge_attr[mask]
    data.edge_vec = data.edge_vec[mask]
    data.shifts = data.shifts[mask]
    
    return data


def update_graph_after_displacement(
    data: Data,
    displacement: torch.Tensor,
) -> Data:
    """
    Update graph representation after atoms have been displaced.
    
    Args:
        data: Original PyG Data object
        displacement: Position changes (N, 3)
        
    Returns:
        Updated Data object with new positions and edge vectors
    """
    data = data.clone()
    
    # Update positions
    data.pos = data.pos + displacement
    data.positions = data.pos
    
    # Update edge vectors
    i, j = data.edge_index
    data.edge_vec = data.edge_vec + displacement[j] - displacement[i]
    data.edge_attr = torch.linalg.norm(data.edge_vec, dim=1, keepdim=True)
    
    return data
