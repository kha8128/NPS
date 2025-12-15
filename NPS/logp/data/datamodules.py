"""
PyTorch Lightning DataModules for crystal structure datasets.
"""

import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import ase.io
import torch
import lightning as L
from torch_geometric.loader import DataLoader

from .datasets import PeriodicStructureDataset, ChemistryAgnosticDataset


class PeriodicStructureDataModule(L.LightningDataModule):
    """
    DataModule for loading periodic crystal structures.
    
    Args:
        file_list: List of structure file paths (extxyz, cif, etc.)
        cutoff: Radial cutoff for neighbor search
        duplicate: Number of times to duplicate each structure
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        structure_types: List of structure type names (for classification)
        train_size: Fraction of data for training (rest is validation)
        dataset_cls: Dataset class to use
    """
    
    def __init__(
        self,
        file_list: List[str],
        cutoff: float = 6.0,
        duplicate: int = 32,
        batch_size: int = 8,
        num_workers: int = 4,
        structure_types: Optional[List[str]] = None,
        train_size: float = 0.9,
        dataset_cls: type = PeriodicStructureDataset,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_cls', 'cutoff'])
        
        self.file_list = file_list
        self.cutoff = cutoff
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.structure_types = structure_types
        self.train_size = train_size
        self.dataset_cls = dataset_cls
        
    def prepare_data(self):
        """Download or prepare data (called on single GPU)."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets (called on every GPU)."""
        atoms_list = []
        structure_type_list = []
        
        for f in self.file_list:
            try:
                atoms = ase.io.read(f)
                atoms_list.append(atoms)
                
                # Get structure type from filename stem
                if self.structure_types:
                    stem = Path(f).stem
                    if stem in self.structure_types:
                        stype = self.structure_types.index(stem)
                    else:
                        stype = 0  # Default
                    structure_type_list.append(stype)
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        # Create dataset
        self.dataset = self.dataset_cls(
            atoms_list=atoms_list,
            cutoff=self.cutoff,
            duplicate=self.duplicate,
            structure_type_list=structure_type_list if self.structure_types else None,
        )
        
        # Train/val split
        train_size = int(len(self.dataset) * self.train_size)
        val_size = len(self.dataset) - train_size
        
        self.train_set, self.valid_set = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


class StrainedPeriodicStructureDataModule(L.LightningDataModule):
    """
    DataModule with random strain augmentation for crystal structures.
    
    Applies random isotropic scaling and symmetric strain to each structure
    during setup, mimicking thermal expansion and mechanical deformation.
    
    Args:
        file_list: List of structure file paths
        cutoff: Radial cutoff for neighbor search
        duplicate: Number of times to duplicate each structure
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        structure_types: List of structure type names
        train_size: Fraction of data for training
        scale_range: Range for isotropic scaling (min, max)
        strain_delta: Maximum strain perturbation magnitude
        strain_symmetric: Whether to use symmetric strain (avoids rotation)
        dataset_cls: Dataset class to use
    """
    
    def __init__(
        self,
        file_list: List[str],
        cutoff: float = 6.0,
        duplicate: int = 32,
        batch_size: int = 8,
        num_workers: int = 4,
        structure_types: Optional[List[str]] = None,
        train_size: float = 0.9,
        scale_range: tuple = (0.9, 1.1),
        strain_delta: float = 0.05,
        strain_symmetric: bool = True,
        dataset_cls: type = PeriodicStructureDataset,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset_cls', 'cutoff'])
        
        self.file_list = file_list
        self.cutoff = cutoff
        self.duplicate = duplicate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.structure_types = structure_types
        self.train_size = train_size
        self.scale_range = scale_range
        self.strain_delta = strain_delta
        self.strain_symmetric = strain_symmetric
        self.dataset_cls = dataset_cls
        
    def _sample_strain_matrix(self) -> np.ndarray:
        """Sample random strain matrix."""
        E = np.random.uniform(-self.strain_delta, self.strain_delta, size=(3, 3))
        
        if self.strain_symmetric:
            E = 0.5 * (E + E.T)
            
        return np.eye(3) + E
    
    def _augment_atoms(self, atoms):
        """Apply random scaling and strain to atoms object."""
        # Isotropic scaling
        scale_factor = random.uniform(*self.scale_range)
        
        # Anisotropic strain
        S = self._sample_strain_matrix()
        
        # Combined transformation
        T = S * scale_factor
        
        # Apply to positions and cell
        pos_new = atoms.positions @ T.T
        cell_new = atoms.cell.array @ T.T
        
        atoms.set_positions(pos_new)
        atoms.set_cell(cell_new, scale_atoms=False)
        atoms.wrap()
        
        return atoms
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        atoms_list = []
        structure_type_list = []
        
        for f in self.file_list:
            try:
                atoms = ase.io.read(f)
                atoms = self._augment_atoms(atoms)
                atoms_list.append(atoms)
                
                if self.structure_types:
                    stem = Path(f).stem
                    if stem in self.structure_types:
                        stype = self.structure_types.index(stem)
                    else:
                        stype = 0
                    structure_type_list.append(stype)
                    
            except Exception as e:
                print(f"Error reading/augmenting {f}: {e}")
                
        self.dataset = self.dataset_cls(
            atoms_list=atoms_list,
            cutoff=self.cutoff,
            duplicate=self.duplicate,
            structure_type_list=structure_type_list if self.structure_types else None,
        )
        
        train_size = int(len(self.dataset) * self.train_size)
        val_size = len(self.dataset) - train_size
        
        self.train_set, self.valid_set = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
