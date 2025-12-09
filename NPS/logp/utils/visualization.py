"""
Visualization utilities for LogP model outputs.
"""

from typing import List, Optional, Tuple

import numpy as np


def plot_logp_distributions(
    logp_trajectory: List[np.ndarray],
    structure_types: List[str],
    true_class: Optional[int] = None,
    steps_to_show: List[int] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot log-probability distributions across denoising steps.
    
    Args:
        logp_trajectory: List of logP arrays (N, nclass) from denoising
        structure_types: Names of structure classes
        true_class: True class index (for highlighting)
        steps_to_show: Which steps to plot (default: first, middle, last)
        figsize: Figure size
        save_path: Path to save figure (if None, displays)
    """
    import matplotlib.pyplot as plt
    
    n_steps = len(logp_trajectory)
    n_classes = logp_trajectory[0].shape[1]
    
    if steps_to_show is None:
        steps_to_show = [0, n_steps // 2, n_steps - 1]
    
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(steps_to_show)))
    
    for class_idx, ax in enumerate(axes):
        for color_idx, step in enumerate(steps_to_show):
            if step >= len(logp_trajectory):
                continue
                
            logp = logp_trajectory[step][:, class_idx]
            
            # Plot histogram
            hist, bin_edges = np.histogram(logp, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            label = f"Step {step}"
            ax.plot(bin_centers, hist, color=colors[color_idx], label=label, linewidth=2)
            
        ax.set_xlabel(f"log P({structure_types[class_idx]})")
        ax.set_ylabel("Density")
        
        if true_class is not None and class_idx == true_class:
            ax.set_title(f"{structure_types[class_idx]} (TRUE)", fontweight='bold')
        else:
            ax.set_title(structure_types[class_idx])
            
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_denoising_trajectory(
    pos_trajectory: List[np.ndarray],
    reference_positions: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot RMSD evolution during denoising.
    
    Args:
        pos_trajectory: List of position arrays from denoising
        reference_positions: Reference/ideal positions (for RMSD calculation)
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if reference_positions is None:
        reference_positions = pos_trajectory[-1]
    
    # Compute RMSD at each step
    rmsd_values = []
    for pos in pos_trajectory:
        diff = pos - reference_positions
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        rmsd_values.append(rmsd)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = np.arange(len(pos_trajectory))
    ax.plot(steps, rmsd_values, 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Denoising Convergence")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_classification_matrix(
    logp_list: List[List[np.ndarray]],
    structure_types: List[str],
    steps_to_show: List[int] = None,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
):
    """
    Plot classification matrix showing logP distributions.
    
    Rows = input structures, Columns = predicted classes.
    Diagonal should show high logP peaks for correct classification.
    
    Args:
        logp_list: List of logP trajectories, one per input structure
        structure_types: Names of structure classes
        steps_to_show: Which denoising steps to show
        figsize: Figure size (default: scales with number of classes)
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    n_structures = len(logp_list)
    n_classes = len(structure_types)
    
    if figsize is None:
        figsize = (3 * n_classes, 3 * n_structures)
    
    if steps_to_show is None:
        steps_to_show = [0, len(logp_list[0]) - 1]
    
    fig, axes = plt.subplots(n_structures, n_classes, figsize=figsize)
    
    # Compute global logP range
    all_logp = np.concatenate([
        np.concatenate([frame.ravel() for frame in traj])
        for traj in logp_list
    ])
    logp_range = (np.min(all_logp), np.max(all_logp))
    
    colors = ['lightgray', 'gray', 'black']
    
    for row_idx, logp_traj in enumerate(logp_list):
        for col_idx in range(n_classes):
            ax = axes[row_idx, col_idx] if n_structures > 1 else axes[col_idx]
            
            for step_idx, step in enumerate(steps_to_show):
                if step >= len(logp_traj):
                    continue
                    
                logp = logp_traj[step][:, col_idx]
                hist, bin_edges = np.histogram(logp, bins=50, range=logp_range)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Highlight diagonal (correct predictions)
                if row_idx == col_idx:
                    color = ['#FFB3BA', '#FF6B6B', '#DC143C'][step_idx]
                else:
                    color = colors[step_idx]
                
                ax.plot(bin_centers, hist, color=color, linewidth=1.5)
            
            # Labels
            if row_idx == n_structures - 1:
                ax.set_xlabel(f"logP({structure_types[col_idx]})")
            if col_idx == 0:
                ax.set_ylabel(f"True: {structure_types[row_idx]}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
