"""
Compatibility shim for old checkpoints.
"""
from NPS.logp.models.lightning_module import *
from NPS.logp.models.lightning_module import LitLogPModel
from NPS.logp.data.datasets import PeriodicStructureDataset

__all__ = ['LitLogPModel', 'PeriodicStructureDataset']
