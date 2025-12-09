"""
Model wrappers for the LogP foundation model.

These wrappers handle:
- One-hot encoding of species
- Autograd-based score computation from log-probabilities
- Loading pretrained weights
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from e3nn import o3

from mace import modules

from .mace_denoiser import MACEDenoiser, MACEDenoiserLight


class LogPModelWrapper(nn.Module):
    """
    Wrapper for MACE denoiser that computes log-probabilities and scores.
    
    This wrapper:
    - Converts atomic numbers to one-hot node attributes
    - Computes per-atom, per-class log-probabilities
    - Derives the score (gradient of total log-probability) via autograd
    
    Args:
        num_species: Number of chemical species (default 89 for elements 1-89)
        nclass: Number of structure classes to predict
        cutoff: Radial cutoff distance
        hidden_irreps: Hidden layer irreducible representations
        num_interactions: Number of message passing layers
        num_bessel: Number of Bessel basis functions
        correlation: Body order for equivariant products
        radial_mlp: Radial MLP layer sizes
        avg_num_neighbors: Average number of neighbors
        pretrained_state_dict: Optional pretrained weights to load
        freeze_backbone: If True, freeze all layers except readout heads
        use_light_model: If True, use MACEDenoiserLight with direct score
    """
    
    def __init__(
        self,
        num_species: int = 89,
        nclass: int = 1,
        cutoff: float = 6.0,
        hidden_irreps: str = "128x0e+128x1o+128x2e",
        num_interactions: int = 3,
        num_bessel: int = 10,
        correlation: int = 3,
        radial_mlp: Optional[list] = None,
        avg_num_neighbors: float = 30.0,
        pretrained_state_dict: Optional[Dict] = None,
        freeze_backbone: bool = False,
        use_light_model: bool = False,
    ):
        super().__init__()
        
        self.num_species = num_species
        self.nclass = nclass
        self.use_light_model = use_light_model
        
        if radial_mlp is None:
            radial_mlp = [64, 64, 64]
            
        atomic_numbers = list(range(num_species))
        
        # Select model class
        model_cls = MACEDenoiserLight if use_light_model else MACEDenoiser
        
        self._model = model_cls(
            r_max=cutoff,
            num_bessel=num_bessel,
            num_polynomial_cutoff=5,
            max_ell=3,
            interaction_cls=modules.interaction_classes['RealAgnosticResidualInteractionBlock'],
            interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            num_interactions=num_interactions,
            num_elements=len(atomic_numbers),
            hidden_irreps=o3.Irreps(hidden_irreps),
            atomic_energies=torch.zeros((1, len(atomic_numbers)), dtype=torch.float32),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            pair_repulsion=False,
            distance_transform='None',
            correlation=correlation,
            gate=modules.gate_dict['silu'],
            MLP_irreps=o3.Irreps('16x0e'),
            atomic_inter_scale=1.0,
            atomic_inter_shift=0,
            radial_MLP=radial_mlp,
            radial_type='bessel',
            nclass=nclass,
        )
        
        # Load pretrained weights if provided
        if pretrained_state_dict is not None:
            self._load_pretrained(pretrained_state_dict)
            
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _load_pretrained(self, state_dict: Dict) -> None:
        """Load pretrained weights, filtering by shape compatibility."""
        model_dict = self._model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(f"Loading {len(filtered)}/{len(state_dict)} pretrained parameters")
        self._model.load_state_dict(filtered, strict=False)
        
    def _freeze_backbone(self) -> None:
        """Freeze all parameters except readout/decoder layers."""
        print("Freezing backbone (encoder + interactions)")
        for name, param in self._model.named_parameters():
            if 'readout' not in name.lower() and 'decoder' not in name.lower():
                param.requires_grad = False
                
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        print(f"Trainable parameters: {trainable}/{total} ({100*trainable/total:.1f}%)")
        
    def forward(
        self, 
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing log-probabilities and scores.
        
        Args:
            batch: PyG Data object with:
                - species: Atomic numbers (N,)
                - pos: Positions (N, 3)
                - edge_index: Graph connectivity (2, E)
                - shifts: Periodic shift vectors (E, 3)
                
        Returns:
            logP: Per-atom, per-class log-probabilities (N, nclass)
            score: Gradient of total log-probability w.r.t. positions (N, 3)
        """
        # Create one-hot node attributes
        batch.node_attrs = torch.nn.functional.one_hot(
            batch.species, self.num_species
        ).to(batch.pos.dtype)
        
        # Sync positions
        batch.positions = batch.pos
        
        if self.use_light_model:
            # Direct score prediction (no autograd needed)
            out = self._model(batch, compute_force=False)
            logP = out["node_energy_channels"]
            score = out["node_score"]
        else:
            # Compute score via autograd
            batch.pos = batch.pos.clone().detach().requires_grad_(True)
            batch.positions = batch.pos
            
            with torch.enable_grad():
                out = self._model(batch, compute_force=False)
                logP = out["node_energy_channels"]
                
                # Per-atom objective: log-sum-exp over classes
                if self.nclass > 1:
                    per_atom = torch.logsumexp(logP, dim=-1)
                else:
                    per_atom = logP.squeeze(-1)
                    
                # Mean over atoms
                if self.nclass > 1:
                    tot = torch.logsumexp(logP, dim=-1).sum()
                else:
                    tot = logP.sum()
                
                # Compute score as gradient
                score = torch.autograd.grad(
                    [tot], [batch.pos], 
                    create_graph=self.training
                )[0]
                
        return logP, score


def load_pretrained_wrapper(
    checkpoint_path: str,
    device: str = "cpu",
    **override_kwargs,
) -> LogPModelWrapper:
    """
    Load a pretrained LogPModelWrapper from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.ckpt or .pt)
        device: Device to load model to
        **override_kwargs: Override model configuration
        
    Returns:
        Loaded LogPModelWrapper
    """
    import os
    
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint
        from .lightning_module import LitLogPModel
        lit_model = LitLogPModel.load_from_checkpoint(
            checkpoint_path, 
            map_location=device
        )
        return lit_model.model
    else:
        # Direct state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Try to infer config from state dict
        # This is a fallback - ideally config should be saved with checkpoint
        model = LogPModelWrapper(**override_kwargs)
        model._load_pretrained(state_dict)
        return model
