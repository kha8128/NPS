#!/usr/bin/env python
"""
Download pre-trained LogP model checkpoints.

Usage:
    python scripts/download_checkpoint.py
    
Or in Python:
    from scripts.download_checkpoint import download_checkpoint
    download_checkpoint("logp_strained_scaled_elemental")
"""

import os
import sys
import urllib.request
from pathlib import Path

# ============================================
# CHECKPOINT REGISTRY
# Update these URLs after uploading to GitHub Releases or Zenodo
# ============================================

CHECKPOINTS = {
    "logp_strained_scaled_elemental": {
        # GitHub Release URL (update OWNER, REPO, TAG, FILENAME)
        "url": "https://github.com/Critical-Materials-Institute/NPS/releases/download/v0.1.0/logp_strained_scaled_elemental.ckpt",
        "size_mb": 233,
        "description": "Elemental crystals (BCC/FCC/HCP) with strain augmentation - best for MD denoising",
        "structure_types": ["bcc", "fcc", "hcp"],
    },
    "logp_scaled_elemental_binary": {
        "url": "https://github.com/Critical-Materials-Institute/NPS/releases/download/v0.1.0/logp_scaled_elemental_binary.ckpt",
        "size_mb": 233,
        "description": "Elemental + binary compounds - best for structure classification",
        "structure_types": ["bcc", "fcc", "hcp"],  # Update with actual types
    },
}

# Alternative: Zenodo URLs (uncomment and update if using Zenodo)
# CHECKPOINTS = {
#     "logp_strained_scaled_elemental": {
#         "url": "https://zenodo.org/record/XXXXXXX/files/logp_strained_scaled_elemental.ckpt",
#         ...
#     },
# }


def get_checkpoint_dir():
    """Get the checkpoints directory, creating if needed."""
    # Try to find the NPS root directory
    current = Path(__file__).resolve().parent.parent
    checkpoint_dir = current / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


def download_checkpoint(name: str, output_dir: str = None, force: bool = False):
    """
    Download a pre-trained checkpoint.
    
    Args:
        name: Checkpoint name (e.g., "logp_strained_scaled_elemental")
        output_dir: Directory to save checkpoint (default: checkpoints/)
        force: If True, re-download even if file exists
        
    Returns:
        Path to downloaded checkpoint
    """
    if name not in CHECKPOINTS:
        available = ", ".join(CHECKPOINTS.keys())
        raise ValueError(f"Unknown checkpoint: {name}. Available: {available}")
    
    info = CHECKPOINTS[name]
    
    if output_dir is None:
        output_dir = get_checkpoint_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{name}.ckpt"
    
    # Check if already downloaded
    if output_path.exists() and not force:
        print(f"✓ Checkpoint already exists: {output_path}")
        return output_path
    
    # Download
    url = info["url"]
    size_mb = info.get("size_mb", "?")
    
    print(f"Downloading {name} ({size_mb} MB)...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")
    
    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
        print()  # New line after progress
        print(f"✓ Downloaded successfully: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nPlease download manually from:")
        print(f"  {url}")
        print(f"And save to: {output_path}")
        raise
    
    return output_path


def list_checkpoints():
    """Print available checkpoints."""
    print("Available checkpoints:\n")
    for name, info in CHECKPOINTS.items():
        print(f"  {name}")
        print(f"    Size: {info.get('size_mb', '?')} MB")
        print(f"    Description: {info.get('description', 'N/A')}")
        print(f"    Structure types: {info.get('structure_types', 'N/A')}")
        print()


def download_all(output_dir: str = None, force: bool = False):
    """Download all available checkpoints."""
    for name in CHECKPOINTS:
        try:
            download_checkpoint(name, output_dir, force)
        except Exception as e:
            print(f"Failed to download {name}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LogP checkpoints")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--name", type=str, help="Checkpoint name to download")
    parser.add_argument("--all", action="store_true", help="Download all checkpoints")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.all:
        download_all(args.output_dir, args.force)
    elif args.name:
        download_checkpoint(args.name, args.output_dir, args.force)
    else:
        # Default: download recommended checkpoint
        print("Downloading recommended checkpoint...\n")
        download_checkpoint("logp_strained_scaled_elemental")
