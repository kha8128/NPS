"""
PyTorch Lightning module for training the LogP model.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import lightning as L

from .wrappers import LogPModelWrapper


class LitLogPModel(L.LightningModule):
    """
    Lightning module for training the log-probability denoiser model.
    
    This module handles:
    - Adding noise to structures (rattling)
    - Computing denoising score-matching loss
    - Computing classification loss (if nclass > 1)
    - EMA model averaging
    
    Args:
        num_species: Number of chemical species
        nclass: Number of structure classes
        cutoff: Radial cutoff for graph construction
        hidden_irreps: Hidden layer irreducible representations
        num_interactions: Number of message passing layers
        sigma_max: Maximum noise level for training
        sigma_logp_scale: Scale factor for log-probability gradient
        wt_classification: Weight for classification loss
        learn_rate: Learning rate
        lr_schedule: Learning rate schedule ("cosine" or "constant")
        ema_decay: Exponential moving average decay rate
        pretrained_state_dict: Optional pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        use_light_model: Use direct score prediction (memory efficient)
    """
    
    def __init__(
        self,
        num_species: int = 89,
        nclass: int = 1,
        cutoff: float = 6.0,
        hidden_irreps: str = "128x0e+128x1o+128x2e",
        num_interactions: int = 3,
        sigma_max: float = 0.15,
        sigma_logp_scale: float = 1.0,
        wt_classification: float = 1.0,
        learn_rate: float = 1e-4,
        lr_schedule: str = "cosine",
        ema_decay: float = 0.9999,
        pretrained_state_dict: Optional[Dict] = None,
        freeze_backbone: bool = False,
        use_light_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_state_dict'])
        
        # Model
        self.model = LogPModelWrapper(
            num_species=num_species,
            nclass=nclass,
            cutoff=cutoff,
            hidden_irreps=hidden_irreps,
            num_interactions=num_interactions,
            pretrained_state_dict=pretrained_state_dict,
            freeze_backbone=freeze_backbone,
            use_light_model=use_light_model,
        )
        
        # EMA model
        ema_avg = lambda avg_params, params, num_avg: (
            ema_decay * avg_params + (1 - ema_decay) * params
        )
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg
        )
        
        # Store params
        self.sigma_max = sigma_max
        self.cutoff = cutoff
        self.learn_rate = learn_rate
        self.lr_schedule = lr_schedule
        self.sigma_logp_scale = sigma_logp_scale
        self.nclass = nclass
        self.wt_classification = wt_classification
        
    def _rattle_particles(
        self, 
        batch, 
        sigma_max: float, 
        sigma_min: float = 0.001
    ):
        """Add random Gaussian noise to particle positions."""
        # Different noise level per graph in batch
        sigma = torch.empty(
            batch.num_graphs, 
            device=batch.pos.device
        ).uniform_(sigma_min, sigma_max)
        
        # Expand sigma to per-atom
        sigma_per_atom = sigma[batch.batch, None]  # (N, 1)
        
        # Store sigma for loss computation
        batch.sigma = sigma_per_atom
        
        # Add noise: disp = sigma * epsilon, where epsilon ~ N(0,1)
        batch.disp = sigma_per_atom * torch.randn_like(batch.pos)
        batch.pos = batch.pos + batch.disp
        batch.positions = batch.pos
        
        # Update edge vectors
        i, j = batch.edge_index
        batch.edge_vec = batch.edge_vec + batch.disp[j] - batch.disp[i]
        batch.edge_attr = torch.linalg.norm(batch.edge_vec, dim=1, keepdim=True)
        
        return batch
    
    def _downselect_edges(self, batch, cutoff: float):
        """Remove edges longer than cutoff."""
        edge_len = torch.linalg.norm(batch.edge_vec, dim=1)
        mask = edge_len < cutoff
        
        batch.edge_index = batch.edge_index[:, mask]
        batch.edge_attr = batch.edge_attr[mask]
        batch.edge_vec = batch.edge_vec[mask]
        batch.shifts = batch.shifts[mask]
        
        return batch
    
    def _compute_loss(self, batch, return_components: bool = False):
        """
        Compute combined score-matching and classification loss.
        
        Args:
            batch: Input batch
            return_components: If True, return individual loss components
            
        Returns:
            Total loss, or (total, score_matching, classification) if return_components
        """
        # Add noise and filter edges
        batch = self._rattle_particles(batch, sigma_max=self.sigma_max)
        batch = self._downselect_edges(batch, cutoff=self.cutoff)
        
        # Forward pass
        logP, pred_score = self.model(batch)
        
        # Target: noise scaled by -1/sigma^2 for logP model
        target = batch.disp * (-1.0 / self.sigma_logp_scale**2)
        
        # Score-matching loss
        loss_sm = F.mse_loss(pred_score, target)
        
        # Classification loss (cross-entropy)
        loss_cl = torch.tensor(0.0, device=batch.pos.device)
        if self.nclass > 1 and hasattr(batch, 'structure_type'):
            loss_cl = F.cross_entropy(logP, batch.structure_type)
            
        # Combined loss
        total_loss = loss_sm + self.wt_classification * loss_cl
        
        if return_components:
            return total_loss, loss_sm, loss_cl
        return total_loss
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss, batch_size=batch.num_graphs, 
                 prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_sm, loss_cl = self._compute_loss(batch, return_components=True)
        self.log('valid_loss', loss, batch_size=batch.num_graphs,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict({"sm_loss": loss_sm, "cl_loss": loss_cl}, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learn_rate,
            weight_decay=1e-4
        )
        
        if self.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2000,
                T_mult=2,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # Update EMA model
        self.ema_model.update_parameters(self.model)
        
    def get_inference_model(self):
        """Get the EMA model for inference."""
        return self.ema_model.module
