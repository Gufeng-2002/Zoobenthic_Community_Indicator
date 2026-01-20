"""
Visualization utilities for contamination assessment.

This module provides plotting functions for PCA results, including ridge plots
to visualize PC loadings across pollution variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.patches import Patch
import os

# Set the figure save path
FIGURE_SAVE_PATH = "../results/figures/01_Contamination_assessment/"


def create_ridge_plot(pca_components, figsize=(16, 10), save_path=None):
    """
    Create a ridge plot of PC loadings with hierarchically clustered variable order.
    
    This visualization shows how each pollution variable contributes to each principal
    component, with variables ordered by similarity for easier interpretation.
    
    Parameters:
    -----------
    pca_components : pd.DataFrame
        PCA loadings matrix (variables × components)
    figsize : tuple, default=(16, 10)
        Figure size (width, height)
    save_path : str, optional
        If provided, save the figure to this directory
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    
    Notes:
    ------
    - Variables are reordered using hierarchical clustering for clearer patterns
    - Positive loadings are shown with solid fill
    - Negative loadings are shown with hatching pattern
    - Each PC is displayed as a separate ridge
    
    Example:
    --------
    >>> PC_loadings, PC_scores, fig = pca_with_PC_loadings(transformed_pollution)
    >>> fig, ax = create_ridge_plot(PC_loadings)
    """
    # Step 1: Perform hierarchical clustering on variables
    distance_matrix = pdist(pca_components.values, metric='euclidean')
    linkage_matrix = linkage(distance_matrix, method='ward')
    clustered_order = leaves_list(linkage_matrix)
    clustered_variable_names = [pca_components.index[i] for i in clustered_order]
    
    # Step 2: Set up the ridge plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ridge plot parameters
    n_pcs = pca_components.shape[1]
    ridge_height = 0.8
    ridge_spacing = 1.0
    baseline_offset = 0.1
    
    # Color scheme: blue gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_pcs))
    
    # Reorder components according to clustering
    pca_components_clustered = pca_components.reindex(clustered_variable_names)
    
    # Step 3: Create ridge plot for each PC
    for i, pc in enumerate(pca_components_clustered.columns):
        loadings = pca_components_clustered[pc].values
        abs_loadings = np.abs(loadings)
        normalized_loadings = (abs_loadings / abs_loadings.max()) * ridge_height
        
        y_baseline = i * ridge_spacing
        x_positions = np.arange(len(clustered_variable_names))
        
        # Draw bars for each variable
        for j, (x_pos, loading, norm_loading) in enumerate(zip(x_positions, loadings, normalized_loadings)):
            y_bottom = y_baseline + baseline_offset
            y_top = y_bottom + norm_loading
            
            # Style positive and negative loadings differently
            if loading >= 0:
                ax.fill_between(
                    [x_pos - 0.4, x_pos + 0.4], 
                    [y_bottom, y_bottom], 
                    [y_top, y_top],
                    color=colors[i], alpha=0.8, 
                    edgecolor='white', linewidth=0.5
                )
            else:
                ax.fill_between(
                    [x_pos - 0.4, x_pos + 0.4], 
                    [y_bottom, y_bottom], 
                    [y_top, y_top],
                    color=colors[i], alpha=0.6, 
                    edgecolor='white', linewidth=0.5, 
                    hatch='///'
                )
        
        # Add baseline and PC label
        ax.axhline(y=y_baseline + baseline_offset, color='lightgray', 
                   linestyle='-', linewidth=0.5, alpha=0.7)
        ax.text(-2, y_baseline + baseline_offset + ridge_height/2, pc, 
                fontsize=12, fontweight='bold', ha='right', va='center')
    
    # Step 4: Style the plot
    ax.set_xlim(-3, len(clustered_variable_names))
    ax.set_ylim(-0.2, n_pcs * ridge_spacing + 0.5)
    ax.set_xticks(range(len(clustered_variable_names)))
    ax.set_xticklabels(clustered_variable_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([])
    
    ax.set_title(
        'PC Loadings Ridge Plot\n(Variables Ordered by Hierarchical Clustering)', 
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xlabel('Chemical Variables (Clustered Order)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Remove spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.8, label='Positive loadings'),
        Patch(facecolor=colors[0], alpha=0.6, hatch='///', label='Negative loadings')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "PCA_PC_loadings_ridge_plot.png"), dpi=300, bbox_inches='tight')
    
    return fig, ax
