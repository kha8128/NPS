"""
PyTorch Geometric datasets for periodic crystal structures.
"""

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from mace.data.neighborhood import get_neighborhood


class PeriodicStructureDataset(Dataset):
    """
    Dataset for periodic crystal structures with graph-based representation.
    
    Each structure is converted to a PyG Data object with:
    - species: Atomic numbers
    - pos/positions: Atom positions
    - edge_index: Graph connectivity
    - edge_vec: Edge vectors (including periodic shifts)
    - edge_attr: Edge lengths
    - shifts: Periodic shift vectors
    - cell: Unit cell
    - structure_type: (optional) Class label for each atom
    
    Args:
        atoms_list: List of ASE Atoms objects
        cutoff: Radial cutoff for neighbor search
        duplicate: Number of times to duplicate each structure (for data augmentation)
        structure_type_list: Optional list of structure type labels (integers)
    """
    
    def __init__(
        self,
        atoms_list: List,
        cutoff: float = 6.0,
        duplicate: int = 1,
        structure_type_list: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.dataset = []
        
        if structure_type_list is not None:
            assert len(structure_type_list) >= len(atoms_list), \
                "structure_type_list must have at least as many entries as atoms_list"
        
        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 0:
                continue
                
            # Wrap atoms into unit cell
            atoms.wrap()
            
            # Get positions and cell
            pos_np = atoms.positions.astype(np.float32)
            cell_np = atoms.cell.array.astype(np.float32)
            
            # Compute neighbor list using MACE's implementation
            edge_index_np, shifts_np, unit_shifts_np, cell_np = get_neighborhood(
                positions=pos_np,
                cutoff=cutoff,
                pbc=atoms.pbc,
                cell=cell_np
            )
            
            # Convert to tensors
            pos = torch.as_tensor(pos_np, dtype=torch.float32)
            cell = torch.as_tensor(cell_np, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index_np, dtype=torch.long).contiguous()
            shifts = torch.as_tensor(shifts_np, dtype=torch.float32)
            
            # Compute edge vectors and lengths
            edge_vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts
            edge_attr = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
            
            # Atomic numbers as species
            species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            
            # Create PyG Data object
            data = Data(
                z=species,
                x=species,
                species=species,
                pos=pos,
                positions=pos,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_attr=edge_attr,
                shifts=shifts,
                cell=cell,
            )
            
            # Add structure type if provided
            if structure_type_list is not None:
                data.structure_type = torch.full_like(species, structure_type_list[idx])
                
            self.dataset.append(data)
        
        # Duplicate dataset
        if duplicate > 1:
            self.dataset = [d.clone() for d in self.dataset for _ in range(duplicate)]
            
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Data:
        return self.dataset[idx].clone()


class ChemistryAgnosticDataset(Dataset):
    """
    Dataset that ignores chemical species (treats all atoms as same element).
    
    Useful for:
    - High-entropy alloys
    - Geometry-only classification
    - Transfer learning across chemistries
    
    Args:
        atoms_list: List of ASE Atoms objects
        cutoff: Radial cutoff for neighbor search
        duplicate: Number of times to duplicate each structure
        structure_type_list: Optional list of structure type labels
        dummy_species: Atomic number to assign to all atoms (default: 1 for H)
    """
    
    def __init__(
        self,
        atoms_list: List,
        cutoff: float = 6.0,
        duplicate: int = 1,
        structure_type_list: Optional[List[int]] = None,
        dummy_species: int = 1,
    ):
        super().__init__()
        
        self.dataset = []
        
        if structure_type_list is not None:
            assert len(structure_type_list) >= len(atoms_list)
            
        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 0:
                continue
                
            atoms.wrap()
            
            pos_np = atoms.positions.astype(np.float32)
            cell_np = atoms.cell.array.astype(np.float32)
            
            edge_index_np, shifts_np, unit_shifts_np, cell_np = get_neighborhood(
                positions=pos_np,
                cutoff=cutoff,
                pbc=atoms.pbc,
                cell=cell_np
            )
            
            pos = torch.as_tensor(pos_np, dtype=torch.float32)
            cell = torch.as_tensor(cell_np, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index_np, dtype=torch.long).contiguous()
            shifts = torch.as_tensor(shifts_np, dtype=torch.float32)
            
            edge_vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts
            edge_attr = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
            
            # All atoms get same species
            species = torch.full((len(atoms),), dummy_species, dtype=torch.long)
            
            data = Data(
                z=species,
                x=species,
                species=species,
                pos=pos,
                positions=pos,
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_attr=edge_attr,
                shifts=shifts,
                cell=cell,
            )
            
            if structure_type_list is not None:
                data.structure_type = torch.full_like(species, structure_type_list[idx])
                
            self.dataset.append(data)
            
        if duplicate > 1:
            self.dataset = [d.clone() for d in self.dataset for _ in range(duplicate)]
            
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Data:
        return self.dataset[idx].clone()
