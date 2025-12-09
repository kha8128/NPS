"""
MACE-based denoiser models for log-probability prediction.

This module extends the MACE architecture to predict per-atom, per-class
log-probabilities for crystal structure classification and denoising.
"""

from typing import Dict, Optional

import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace import modules
from mace.modules.blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
)


@compile_mode("script")
class MACEDenoiser(modules.ScaleShiftMACE):
    """
    MACE model extended with multi-class log-probability output.
    
    This model adds a secondary decoder head (decoders2) that outputs
    per-atom log-probabilities for each structure class. The gradients
    of these log-probabilities define the denoising score field.
    
    Args:
        nclass: Number of structure classes to predict
        **kwargs: Arguments passed to ScaleShiftMACE
    """
    
    def __init__(self, nclass: int = 1, **kwargs):
        super().__init__(**kwargs)
        
        hidden_irreps = kwargs["hidden_irreps"]
        num_interactions = kwargs["num_interactions"]
        heads = self.heads
        MLP_irreps = kwargs["MLP_irreps"]
        gate = kwargs["gate"]
        
        # Track feature dimensions for splitting
        self.n_feat_out_list = [o3.Irreps(hidden_irreps).dim]
        
        # Secondary decoders for log-probability output
        self.decoders2 = nn.ModuleList()
        self.decoders2.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{nclass}x0e"))
        )
        
        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                # Last layer: scalar-only for output
                hidden_irreps_out = str(hidden_irreps[0])
            else:
                hidden_irreps_out = hidden_irreps
                
            if i == num_interactions - 2:
                self.decoders2.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{nclass}x0e"),
                        len(heads),
                    )
                )
            else:
                self.decoders2.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{nclass}x0e"))
                )
            self.n_feat_out_list.append(o3.Irreps(hidden_irreps_out).dim)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass returning both standard MACE outputs and log-probabilities.
        
        Returns:
            Dictionary containing:
                - All standard MACE outputs (energy, forces, etc.)
                - node_energy_channels: Per-atom, per-class log-probabilities (N, nclass)
        """
        output = super().forward(data, **kwargs)
        
        # Compute per-class log-probabilities from each interaction layer
        node_energy_channel_list = []
        for node_feats, readout in zip(
            torch.split(output["node_feats"], self.n_feat_out_list, -1),
            self.decoders2
        ):
            node_energy_channel_list.append(readout(node_feats))
            
        # Sum contributions from all layers
        output["node_energy_channels"] = torch.sum(
            torch.stack(node_energy_channel_list, dim=0), dim=0
        )
        
        return output


@compile_mode("script")
class MACEDenoiserLight(modules.ScaleShiftMACE):
    """
    Lightweight MACE denoiser with direct score prediction.
    
    This variant adds both:
    - Scalar decoders for log-probability (same as MACEDenoiser)
    - Vector decoders for direct score prediction (avoids autograd)
    
    The direct score prediction is more memory-efficient for large systems.
    
    Args:
        nclass: Number of structure classes to predict
        **kwargs: Arguments passed to ScaleShiftMACE
    """
    
    def __init__(self, nclass: int = 1, **kwargs):
        super().__init__(**kwargs)
        
        hidden_irreps = kwargs["hidden_irreps"]
        num_interactions = kwargs["num_interactions"]
        heads = self.heads
        MLP_irreps = kwargs["MLP_irreps"]
        gate = kwargs["gate"]
        
        self.n_feat_out_list = [o3.Irreps(hidden_irreps).dim]
        
        # Scalar decoders for log-probability
        self.decoders2 = nn.ModuleList()
        self.decoders2.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{nclass}x0e"))
        )
        
        # Vector decoders for direct score prediction
        # Output is 1o (odd parity vector) to match hidden irreps
        self.decoders_score = nn.ModuleList()
        self.decoders_score.append(
            LinearReadoutBlock(hidden_irreps, o3.Irreps("1x1o"))
        )
        
        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(hidden_irreps[0])
            else:
                hidden_irreps_out = hidden_irreps
                
            # Scalar head
            if i == num_interactions - 2:
                self.decoders2.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{nclass}x0e"),
                        len(heads),
                    )
                )
            else:
                self.decoders2.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps(f"{nclass}x0e"))
                )
                
            # Score head
            if i == num_interactions - 2:
                self.decoders_score.append(
                    LinearReadoutBlock(hidden_irreps_out, o3.Irreps("1x1o"))
                )
            else:
                self.decoders_score.append(
                    LinearReadoutBlock(hidden_irreps, o3.Irreps("1x1o"))
                )
                
            self.n_feat_out_list.append(o3.Irreps(hidden_irreps_out).dim)
            
        assert len(self.decoders2) == len(self.decoders_score) == len(self.n_feat_out_list)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass returning log-probabilities and direct score predictions.
        
        Returns:
            Dictionary containing:
                - All standard MACE outputs
                - node_energy_channels: Per-atom, per-class log-probabilities (N, nclass)
                - node_score: Direct score/denoising direction (N, 3)
        """
        output = super().forward(data, **kwargs)
        
        node_energy_channel_list = []
        node_score_list = []
        
        node_feats_splits = torch.split(output["node_feats"], self.n_feat_out_list, -1)
        
        for node_feats, readout_scalar, readout_vector in zip(
            node_feats_splits, self.decoders2, self.decoders_score
        ):
            node_energy_channel_list.append(readout_scalar(node_feats))
            node_score_list.append(readout_vector(node_feats))
            
        output["node_energy_channels"] = torch.sum(
            torch.stack(node_energy_channel_list, dim=0), dim=0
        )
        output["node_score"] = torch.sum(
            torch.stack(node_score_list, dim=0), dim=0
        )
        
        return output


def create_mace_denoiser(
    nclass: int = 1,
    num_species: int = 89,
    cutoff: float = 6.0,
    num_bessel: int = 10,
    num_polynomial_cutoff: int = 5,
    max_ell: int = 3,
    num_interactions: int = 3,
    hidden_irreps: str = "128x0e+128x1o+128x2e",
    correlation: int = 3,
    mlp_irreps: str = "16x0e",
    radial_mlp: list = None,
    avg_num_neighbors: float = 30.0,
    atomic_numbers: list = None,
    light: bool = False,
) -> nn.Module:
    """
    Factory function to create a MACE denoiser model.
    
    Args:
        nclass: Number of structure classes
        num_species: Number of chemical species
        cutoff: Radial cutoff distance
        num_bessel: Number of Bessel basis functions
        num_polynomial_cutoff: Order of polynomial cutoff
        max_ell: Maximum spherical harmonic order
        num_interactions: Number of message passing layers
        hidden_irreps: Hidden layer irreducible representations
        correlation: Body order for equivariant products
        mlp_irreps: MLP irreducible representations
        radial_mlp: Radial MLP layer sizes
        avg_num_neighbors: Average number of neighbors (for normalization)
        atomic_numbers: List of atomic numbers to consider
        light: If True, use MACEDenoiserLight with direct score prediction
        
    Returns:
        MACE denoiser model
    """
    if radial_mlp is None:
        radial_mlp = [64, 64, 64]
    if atomic_numbers is None:
        atomic_numbers = list(range(num_species))
        
    model_cls = MACEDenoiserLight if light else MACEDenoiser
    
    model = model_cls(
        r_max=cutoff,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
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
        MLP_irreps=o3.Irreps(mlp_irreps),
        atomic_inter_scale=1.0,
        atomic_inter_shift=0,
        radial_MLP=radial_mlp,
        radial_type='bessel',
        nclass=nclass,
    )
    
    return model
