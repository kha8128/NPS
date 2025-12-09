#!/usr/bin/env python
"""
Inference script for the LogP foundation model.

Example usage:
    # Classify a single structure
    python -m NPS.logp.scripts.infer \
        --checkpoint model.ckpt \
        --input structure.extxyz \
        --structure_types "bcc,fcc,hcp" \
        --mode classify

    # Denoise a trajectory
    python -m NPS.logp.scripts.infer \
        --checkpoint model.ckpt \
        --input trajectory.extxyz \
        --mode denoise \
        --output denoised.extxyz
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import ase.io

from NPS.logp.models import LitLogPModel
from NPS.logp.inference import (
    denoise_structure,
    denoise_trajectory,
    classify_structure,
    classify_zeroshot,
    compute_order_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with LogP model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input structure file(s) or glob pattern"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path (for denoise mode)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["denoise", "classify", "classify_zeroshot", "order_params"],
        default="classify",
        help="Inference mode"
    )
    parser.add_argument(
        "--structure_types",
        type=str,
        default="",
        help="Comma-separated structure type names"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Radial cutoff"
    )
    parser.add_argument(
        "--sigma_scale",
        type=float,
        default=1.0,
        help="Sigma scale for denoising"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Path to save plot (for classify mode)"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    lit_model = LitLogPModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    lit_model.eval()
    
    # Get the EMA model for inference
    model = lit_model.get_inference_model()
    model = model.to(device)
    
    return model


def main():
    args = parse_args()
    
    # Parse structure types
    structure_types = [x.strip() for x in args.structure_types.split(",") if x.strip()]
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load input structures
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        raise ValueError(f"No files found matching: {args.input}")
    
    print(f"Processing {len(input_files)} files")
    
    if args.mode == "denoise":
        # Denoise structures
        all_denoised = []
        
        for f in input_files:
            print(f"Denoising: {f}")
            atoms = ase.io.read(f)
            
            if isinstance(atoms, list):
                # Trajectory
                denoised = denoise_trajectory(
                    atoms,
                    model=model,
                    steps=args.steps,
                    cutoff=args.cutoff,
                    sigma_scale=args.sigma_scale,
                    device=args.device,
                )
                all_denoised.extend(denoised)
            else:
                # Single structure
                denoised = denoise_structure(
                    atoms,
                    model=model,
                    steps=args.steps,
                    cutoff=args.cutoff,
                    sigma_scale=args.sigma_scale,
                    device=args.device,
                )
                all_denoised.append(denoised)
        
        # Save output
        output_path = args.output or "denoised.extxyz"
        ase.io.write(output_path, all_denoised)
        print(f"Saved denoised structures to {output_path}")
        
    elif args.mode == "classify":
        # Full classification with denoising
        if not structure_types:
            raise ValueError("--structure_types required for classify mode")
        
        for f in input_files:
            print(f"\nClassifying: {f}")
            atoms = ase.io.read(f)
            
            result = classify_structure(
                atoms,
                model=model,
                structure_types=structure_types,
                steps=args.steps,
                cutoff=args.cutoff,
                sigma_scale=args.sigma_scale,
                device=args.device,
                return_logp=True,
            )
            
            print(f"  Majority class: {result['majority_class']}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            # Distribution of predictions
            unique, counts = np.unique(result['predictions'], return_counts=True)
            print("  Class distribution:")
            for idx, count in zip(unique, counts):
                pct = 100 * count / len(result['predictions'])
                print(f"    {structure_types[idx]}: {count} ({pct:.1f}%)")
                
    elif args.mode == "classify_zeroshot":
        # Zero-shot classification (no denoising)
        if not structure_types:
            raise ValueError("--structure_types required for classify_zeroshot mode")
        
        for f in input_files:
            print(f"\nClassifying (zero-shot): {f}")
            atoms = ase.io.read(f)
            
            predictions, confidence = classify_zeroshot(
                atoms,
                model=model,
                structure_types=structure_types,
                cutoff=args.cutoff,
                device=args.device,
            )
            
            # Distribution
            unique, counts = np.unique(predictions, return_counts=True)
            majority_idx = unique[np.argmax(counts)]
            
            print(f"  Majority class: {structure_types[majority_idx]}")
            print(f"  Confidence: {confidence:.4f}")
            print("  Class distribution:")
            for idx, count in zip(unique, counts):
                pct = 100 * count / len(predictions)
                print(f"    {structure_types[idx]}: {count} ({pct:.1f}%)")
                
    elif args.mode == "order_params":
        # Compute order parameters
        if not structure_types:
            raise ValueError("--structure_types required for order_params mode")
        
        for f in input_files:
            print(f"\nComputing order parameters: {f}")
            atoms = ase.io.read(f)
            
            ops = compute_order_parameters(
                atoms,
                model=model,
                structure_types=structure_types,
                steps=args.steps,
                cutoff=args.cutoff,
                sigma_scale=args.sigma_scale,
                device=args.device,
            )
            
            # Print summary
            for name in structure_types:
                values = ops[name]
                print(f"  {name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
            
            # Save to arrays file if output specified
            if args.output:
                np.savez(args.output, **ops)
                print(f"Saved order parameters to {args.output}")


if __name__ == "__main__":
    main()
