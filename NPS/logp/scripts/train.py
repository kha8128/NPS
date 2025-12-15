#!/usr/bin/env python
"""
Training script for the LogP foundation model.

Example usage:
    python -m NPS.logp.scripts.train \
        --train_data "examples/sample_data/*.extxyz" \
        --structure_types "A_cI2_229,A_cF4_225,A_hP2_194" \
        --batch_size 8 \
        --max_steps 100000

For distributed training:
    torchrun --nproc_per_node=4 -m NPS.logp.scripts.train \
        --train_data "examples/sample_data/*.extxyz" \
        --structure_types "A_cI2_229,A_cF4_225,A_hP2_194"
"""

import argparse
import glob
import shlex
from pathlib import Path

import torch
import torch.distributed as dist
import lightning as L
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from NPS.logp.models import LitLogPModel
from NPS.logp.data import StrainedPeriodicStructureDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LogP foundation model for crystal structure analysis"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data", 
        type=str, 
        required=True,
        help="Glob pattern for training structure files (e.g., 'data/*.extxyz')"
    )
    parser.add_argument(
        "--structure_types",
        type=str,
        default="",
        help="Comma-separated structure type names or path to text file"
    )
    parser.add_argument(
        "--atom_types",
        type=str,
        default="",
        help="Comma-separated atom types or path to text file"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_irreps",
        type=str,
        default="128x0e+128x1o+128x2e",
        help="Hidden layer irreducible representations"
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of message passing layers"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Radial cutoff distance"
    )
    parser.add_argument(
        "--use_light_model",
        action="store_true",
        help="Use lightweight model with direct score prediction"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=0.15,
        help="Maximum noise level for training"
    )
    parser.add_argument(
        "--sigma_logp_scale",
        type=float,
        default=1.0,
        help="Scale factor for log-probability gradient"
    )
    parser.add_argument(
        "--wt_classification",
        type=float,
        default=1.0,
        help="Weight for classification loss"
    )
    
    # Distributed training
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training"
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
        help="GPUs per node"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to pretrained MACE weights"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze pretrained backbone, only train decoder"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logp_training",
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="logp_model",
        help="Name for this training run"
    )
    
    return parser.parse_args()


def load_list_from_arg(arg_value: str):
    """Load list from comma-separated string or text file."""
    if not arg_value:
        return []
    
    if Path(arg_value).is_file():
        with open(arg_value) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return [x.strip() for x in arg_value.split(",") if x.strip()]


def main():
    args = parse_args()
    
    # Parse structure and atom types
    structure_types = load_list_from_arg(args.structure_types)
    atom_types = load_list_from_arg(args.atom_types)
    
    nclass = max(1, len(structure_types))
    
    # Expand file patterns
    file_list = []
    for pattern in shlex.split(args.train_data):
        file_list.extend(sorted(glob.glob(pattern)))
    
    if not file_list:
        raise ValueError(f"No files found matching: {args.train_data}")
    
    print(f"Found {len(file_list)} training files")
    print(f"Structure types: {structure_types}")
    print(f"Number of classes: {nclass}")
    
    # Initialize distributed training if needed
    if args.num_nodes > 1 or args.gpus_per_node > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    
    # Create data module
    # Use larger cutoff for graph precomputation (edges are filtered during training)
    datamodule = StrainedPeriodicStructureDataModule(
        file_list=file_list,
        cutoff=args.cutoff + 1.0,
        duplicate=32,
        batch_size=args.batch_size,
        num_workers=4,
        structure_types=structure_types if structure_types else None,
    )
    
    # Load pretrained weights if specified
    pretrained_state_dict = None
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        pretrained_state_dict = torch.load(args.pretrained, map_location="cpu")
        if "state_dict" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["state_dict"]
    
    # Create model
    model = LitLogPModel(
        num_species=100,
        nclass=nclass,
        cutoff=args.cutoff,
        hidden_irreps=args.hidden_irreps,
        num_interactions=args.num_interactions,
        sigma_max=args.sigma_max,
        sigma_logp_scale=args.sigma_logp_scale,
        wt_classification=args.wt_classification,
        learn_rate=args.lr,
        pretrained_state_dict=pretrained_state_dict,
        freeze_backbone=args.freeze_backbone,
        use_light_model=args.use_light_model,
    )
    
    # Callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=f"{args.output_dir}/{args.run_name}/checkpoints",
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_top_k=3,
            monitor="valid_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.run_name,
    )
    
    # Trainer
    trainer = L.Trainer(
        strategy='ddp_find_unused_parameters_true' if args.gpus_per_node > 1 else 'auto',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    
    # Train
    trainer.fit(
        model,
        datamodule,
        ckpt_path=args.checkpoint if args.checkpoint else None,
    )


if __name__ == "__main__":
    main()
