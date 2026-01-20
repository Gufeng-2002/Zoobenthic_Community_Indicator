"""
Taxa Loadings Visualization Module

This module provides functions for visualizing species contributions (loadings)
to principal components across habitat clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Union


def get_top_taxa_loadings(
    loadings_df: pd.DataFrame,
    top_n: int = 10,
    pc_columns: Optional[List[str]] = None
) -> Dict[str, pd.Series]:
    """
    Get top taxa (species) with highest absolute loadings for each PC.
    
    Parameters:
    -----------
    loadings_df : pd.DataFrame
        Species loadings DataFrame (species x PCs)
    top_n : int
        Number of top taxa to return per PC
    pc_columns : list, optional
        Specific PC columns to analyze. If None, uses all columns.
        
    Returns:
    --------
    dict
        {pc_name: Series of top species with their loadings}
    """
    if pc_columns is None:
        pc_columns = loadings_df.columns.tolist()
    
    top_taxa = {}
    for pc in pc_columns:
        if pc in loadings_df.columns:
            # Sort by absolute value, keep actual values
            sorted_loadings = loadings_df[pc].reindex(
                loadings_df[pc].abs().sort_values(ascending=False).index
            )
            top_taxa[pc] = sorted_loadings.head(top_n)
    
    return top_taxa


def plot_taxa_loadings_single_cluster(
    loadings_df: pd.DataFrame,
    variance_explained: np.ndarray,
    cluster_id: int,
    top_n: int = 10,
    ax: Optional[plt.Axes] = None,
    color: str = '#1f77b4',
    show_error_bars: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a bar plot of taxa loadings for a single cluster.
    
    Parameters:
    -----------
    loadings_df : pd.DataFrame
        Species loadings DataFrame (species x PCs)
    variance_explained : np.ndarray
        Variance explained by each PC (as percentages)
    cluster_id : int
        Cluster identifier for title
    top_n : int
        Number of top taxa to display
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    color : str
        Bar color
    show_error_bars : bool
        Whether to show error bars (standard deviation across PCs)
    figsize : tuple
        Figure size if creating new figure
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Calculate composite loading (weighted by variance explained)
    weights = variance_explained / variance_explained.sum()
    n_pcs = len(variance_explained)
    
    # Compute weighted mean loading for each species
    weighted_loadings = pd.Series(0.0, index=loadings_df.index)
    loading_std = pd.Series(0.0, index=loadings_df.index)
    
    for i, pc in enumerate(loadings_df.columns[:n_pcs]):
        weighted_loadings += loadings_df[pc].abs() * weights[i]
    
    # Calculate standard deviation across PCs for error bars
    if show_error_bars and n_pcs > 1:
        loading_std = loadings_df.iloc[:, :n_pcs].abs().std(axis=1)
    
    # Get top taxa
    top_taxa = weighted_loadings.nlargest(top_n)
    top_taxa_std = loading_std[top_taxa.index]
    
    # Create bar plot
    x = np.arange(len(top_taxa))
    bars = ax.bar(x, top_taxa.values, color=color, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    if show_error_bars:
        ax.errorbar(x, top_taxa.values, yerr=top_taxa_std.values,
                   fmt='none', ecolor='black', capsize=3, capthick=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_taxa.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Weighted Loading (±SD)', fontsize=10, fontweight='bold')
    ax.set_title(f'Cluster {int(cluster_id)}: Top {top_n} Taxa Loadings\n'
                 f'({n_pcs} PCs, {variance_explained.sum():.1f}% variance)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    return fig


def plot_taxa_loadings_stacked(
    pca_results: Dict,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 12),
    colors: Optional[Dict] = None,
    title: Optional[str] = None,
    dpi: int = 150,
    taxa_loadings_pcs: Optional[Dict[int, Tuple[int, ...]]] = None
) -> plt.Figure:
    """
    Create stacked bar plots of taxa loadings across all clusters.
    
    Each cluster gets its own row, showing top taxa contributions to PCs.
    This allows comparison of which species drive compositional variation
    in different habitat types.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas() containing:
        - 'loadings': {cluster_id: DataFrame}
        - 'variance_explained': {cluster_id: array}
        - 'n_components': {cluster_id: int}
    top_n : int
        Number of top taxa to display per cluster
    figsize : tuple
        Figure size (width, height)
    colors : dict, optional
        Cluster colors {cluster_id: color}
    title : str, optional
        Overall figure title
    dpi : int
        Figure resolution
    taxa_loadings_pcs : dict, optional
        {cluster_id: (pc1, pc2, ...)} specifying which PCs to use for calculating
        weighted loadings in each cluster. PC numbers are 1-indexed.
        If None, uses all PCs from the PCA results.
        Example: {0: (1, 2, 3), 1: (1, 2), 2: (1,)} uses PC1-3 for cluster 0,
        PC1-2 for cluster 1, and PC1 only for cluster 2.
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with stacked bar plots
    """
    loadings = pca_results['loadings']
    variance_explained = pca_results['variance_explained']
    n_components = pca_results['n_components']
    
    clusters = sorted(loadings.keys())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Create figure with subplots (one row per cluster)
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize, dpi=dpi)
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        cluster_loadings = loadings[cluster]
        cluster_variance = variance_explained[cluster]
        cluster_n_pcs = n_components[cluster]
        
        # Determine which PCs to use for this cluster
        if taxa_loadings_pcs is not None and cluster in taxa_loadings_pcs:
            # User-specified PCs (1-indexed), convert to 0-indexed
            pc_indices = [pc - 1 for pc in taxa_loadings_pcs[cluster]]
            # Filter to valid indices that exist
            pc_indices = [i for i in pc_indices if i < len(cluster_loadings.columns)]
            if len(pc_indices) == 0:
                # Fallback to all PCs if no valid indices
                pc_indices = list(range(cluster_n_pcs))
        else:
            # Use all PCs
            pc_indices = list(range(cluster_n_pcs))
        
        pcs_used = [i + 1 for i in pc_indices]  # 1-indexed for display
        
        # Get variance for selected PCs and calculate weights
        selected_variance = cluster_variance[pc_indices]
        weights = selected_variance / selected_variance.sum()
        
        # Calculate weighted loadings using only selected PCs
        weighted_loadings = pd.Series(0.0, index=cluster_loadings.index)
        for w_idx, pc_idx in enumerate(pc_indices):
            pc_col = cluster_loadings.columns[pc_idx]
            weighted_loadings += cluster_loadings[pc_col].abs() * weights[w_idx]
        
        # Calculate std across selected PCs for error bars
        if len(pc_indices) > 1:
            selected_pc_cols = [cluster_loadings.columns[i] for i in pc_indices]
            loading_std = cluster_loadings[selected_pc_cols].abs().std(axis=1)
        else:
            loading_std = pd.Series(0.0, index=cluster_loadings.index)
        
        # Get top taxa
        top_taxa = weighted_loadings.nlargest(top_n)
        top_taxa_std = loading_std[top_taxa.index]
        
        # Create bar plot
        x = np.arange(len(top_taxa))
        color = colors.get(cluster, '#1f77b4')
        
        bars = ax.bar(x, top_taxa.values, color=color, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add error bars
        ax.errorbar(x, top_taxa.values, yerr=top_taxa_std.values,
                   fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(top_taxa.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Weighted Loading', fontsize=10)
        
        # Build title showing which PCs were used
        if len(pcs_used) == cluster_n_pcs:
            # Using all PCs
            pc_str = f'{len(pcs_used)} PCs'
            var_sum = cluster_variance.sum()
        else:
            # Using subset of PCs
            var_sum = selected_variance.sum()
            if len(pcs_used) <= 3:
                pc_str = f'PC{"s" if len(pcs_used) > 1 else ""} {", ".join(map(str, pcs_used))}'
            else:
                pc_str = f'{len(pcs_used)} PCs ({pcs_used[0]}-{pcs_used[-1]})'
        
        ax.set_title(f'Cluster {int(cluster)}: {pc_str} '
                     f'({var_sum:.1f}% variance)',
                     fontsize=11, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add variance annotation
        ax.text(0.98, 0.95, f'n={len(pca_results["training_sites"][cluster])} sites',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, style='italic')
    
    # Set common x-label only on bottom subplot
    axes[-1].set_xlabel('Taxa Species', fontsize=11, fontweight='bold')
    
    # Overall title
    if title is None:
        title = 'Taxa Loadings Across Habitat Clusters\n(Weighted by PC variance explained, ±SD across PCs)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def plot_taxa_loadings_consistent(
    pca_results: Dict,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 12),
    colors: Optional[Dict] = None,
    title: Optional[str] = None,
    dpi: int = 150,
    show_upper_error_only: bool = True,
    taxa_loadings_pcs: Optional[Dict[int, Tuple[int, ...]]] = None
) -> plt.Figure:
    """
    Create stacked bar plots of taxa loadings with CONSISTENT taxa order across all clusters.
    
    Each cluster panel shows the same taxa in the same order (based on mean loading 
    across clusters), with only the upper half of error bars displayed.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas() containing:
        - 'loadings': {cluster_id: DataFrame}
        - 'variance_explained': {cluster_id: array}
        - 'n_components': {cluster_id: int}
    top_n : int
        Number of top taxa to display per cluster
    figsize : tuple
        Figure size (width, height)
    colors : dict, optional
        Cluster colors {cluster_id: color}
    title : str, optional
        Overall figure title
    dpi : int
        Figure resolution
    show_upper_error_only : bool
        If True, only show upper half of error bars (default: True)
    taxa_loadings_pcs : dict, optional
        {cluster_id: (pc1, pc2, ...)} specifying which PCs to use for calculating
        weighted loadings in each cluster. PC numbers are 1-indexed.
        If None, uses all PCs from the PCA results.
        Example: {0: (1, 2, 3), 1: (1, 2), 2: (1,)} uses PC1-3 for cluster 0,
        PC1-2 for cluster 1, and PC1 only for cluster 2.
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with stacked bar plots
    """
    loadings = pca_results['loadings']
    variance_explained = pca_results['variance_explained']
    n_components = pca_results['n_components']
    
    clusters = sorted(loadings.keys())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # First, calculate weighted loadings for all clusters
    weighted_loadings_all = {}
    loading_std_all = {}
    pcs_used_info = {}  # Store info about which PCs were used for each cluster
    
    for cluster in clusters:
        cluster_loadings = loadings[cluster]
        cluster_variance = variance_explained[cluster]
        cluster_n_pcs = n_components[cluster]
        
        # Determine which PCs to use for this cluster
        if taxa_loadings_pcs is not None and cluster in taxa_loadings_pcs:
            # User-specified PCs (1-indexed), convert to 0-indexed
            pc_indices = [pc - 1 for pc in taxa_loadings_pcs[cluster]]
            # Filter to valid indices that exist
            pc_indices = [i for i in pc_indices if i < len(cluster_loadings.columns)]
            if len(pc_indices) == 0:
                # Fallback to all PCs if no valid indices
                pc_indices = list(range(cluster_n_pcs))
        else:
            # Use all PCs
            pc_indices = list(range(cluster_n_pcs))
        
        pcs_used_info[cluster] = [i + 1 for i in pc_indices]  # Store 1-indexed for display
        
        # Get variance for selected PCs and calculate weights
        selected_variance = cluster_variance[pc_indices]
        weights = selected_variance / selected_variance.sum()
        
        # Calculate weighted loadings using only selected PCs
        weighted_loadings = pd.Series(0.0, index=cluster_loadings.index)
        for idx, pc_idx in enumerate(pc_indices):
            pc_col = cluster_loadings.columns[pc_idx]
            weighted_loadings += cluster_loadings[pc_col].abs() * weights[idx]
        
        weighted_loadings_all[cluster] = weighted_loadings
        
        # Calculate std across selected PCs for error bars
        if len(pc_indices) > 1:
            selected_pc_cols = [cluster_loadings.columns[i] for i in pc_indices]
            loading_std_all[cluster] = cluster_loadings[selected_pc_cols].abs().std(axis=1)
        else:
            loading_std_all[cluster] = pd.Series(0.0, index=cluster_loadings.index)
    
    # Get union of top taxa across all clusters
    top_taxa_sets = [set(weighted_loadings_all[c].nlargest(top_n).index) for c in clusters]
    all_top_taxa = list(set.union(*top_taxa_sets))
    
    # Sort by MEAN loading across clusters (for consistent order)
    mean_loadings = pd.Series(index=all_top_taxa, dtype=float)
    for taxon in all_top_taxa:
        mean_loadings[taxon] = np.mean([
            weighted_loadings_all[c].get(taxon, 0) for c in clusters
        ])
    
    # Get the final ordered list of top taxa
    ordered_taxa = mean_loadings.nlargest(top_n).index.tolist()
    
    # Create figure with subplots (one row per cluster)
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize, dpi=dpi)
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        cluster_variance = variance_explained[cluster]
        cluster_n_pcs = n_components[cluster]
        
        # Get values for the ordered taxa
        values = [weighted_loadings_all[cluster].get(t, 0) for t in ordered_taxa]
        std_values = [loading_std_all[cluster].get(t, 0) for t in ordered_taxa]
        
        # Create bar plot
        x = np.arange(len(ordered_taxa))
        color = colors.get(cluster, '#1f77b4')
        
        bars = ax.bar(x, values, color=color, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add error bars - upper half only if specified
        if show_upper_error_only:
            # Only upper error bar: yerr format is [lower_errors, upper_errors]
            lower_errors = np.zeros(len(std_values))
            upper_errors = np.array(std_values)
            ax.errorbar(x, values, yerr=[lower_errors, upper_errors],
                       fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.7)
        else:
            ax.errorbar(x, values, yerr=std_values,
                       fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_taxa, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Weighted Loading', fontsize=10)
        
        # Build title showing which PCs were used
        pcs_used = pcs_used_info[cluster]
        cluster_variance = variance_explained[cluster]
        if len(pcs_used) == n_components[cluster]:
            # Using all PCs
            pc_str = f'{len(pcs_used)} PCs'
            var_sum = cluster_variance.sum()
        else:
            # Using subset of PCs
            pc_indices = [pc - 1 for pc in pcs_used]
            var_sum = cluster_variance[pc_indices].sum()
            if len(pcs_used) <= 3:
                pc_str = f'PC{"s" if len(pcs_used) > 1 else ""} {", ".join(map(str, pcs_used))}'
            else:
                pc_str = f'{len(pcs_used)} PCs ({pcs_used[0]}-{pcs_used[-1]})'
        
        ax.set_title(f'Cluster {int(cluster)}: {pc_str} '
                     f'({var_sum:.1f}% variance)',
                     fontsize=11, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add variance annotation
        ax.text(0.98, 0.95, f'n={len(pca_results["training_sites"][cluster])} sites',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, style='italic')
    
    # Set common x-label only on bottom subplot
    axes[-1].set_xlabel('Taxa Species', fontsize=11, fontweight='bold')
    
    # Overall title
    if title is None:
        title = 'Taxa Loadings Across Habitat Clusters\n(Weighted by PC variance explained, +SD across PCs)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def plot_taxa_loadings_comparison(
    pca_results: Dict,
    top_n: int = 8,
    figsize: Tuple[int, int] = (16, 8),
    colors: Optional[Dict] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a grouped bar plot comparing top taxa across clusters.
    
    Shows the same set of taxa across all clusters to highlight
    differences in their relative importance.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas()
    top_n : int
        Number of top taxa to display
    figsize : tuple
        Figure size
    colors : dict, optional
        Cluster colors
    dpi : int
        Figure resolution
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure
    """
    loadings = pca_results['loadings']
    variance_explained = pca_results['variance_explained']
    n_components = pca_results['n_components']
    
    clusters = sorted(loadings.keys())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Calculate weighted loadings for each cluster
    weighted_loadings_all = {}
    for cluster in clusters:
        cluster_loadings = loadings[cluster]
        cluster_variance = variance_explained[cluster]
        weights = cluster_variance / cluster_variance.sum()
        
        weighted = pd.Series(0.0, index=cluster_loadings.index)
        for i, pc in enumerate(cluster_loadings.columns[:n_components[cluster]]):
            weighted += cluster_loadings[pc].abs() * weights[i]
        weighted_loadings_all[cluster] = weighted
    
    # Get union of top taxa across all clusters
    top_taxa_sets = [set(weighted_loadings_all[c].nlargest(top_n).index) for c in clusters]
    common_taxa = list(set.union(*top_taxa_sets))
    
    # Sort by mean loading across clusters
    mean_loadings = pd.Series(index=common_taxa)
    for taxon in common_taxa:
        mean_loadings[taxon] = np.mean([
            weighted_loadings_all[c].get(taxon, 0) for c in clusters
        ])
    common_taxa = mean_loadings.nlargest(min(top_n * 2, len(common_taxa))).index.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    x = np.arange(len(common_taxa))
    width = 0.8 / n_clusters
    
    for i, cluster in enumerate(clusters):
        offset = (i - n_clusters / 2 + 0.5) * width
        values = [weighted_loadings_all[cluster].get(t, 0) for t in common_taxa]
        
        bars = ax.bar(x + offset, values, width,
                      label=f'Cluster {int(cluster)}',
                      color=colors.get(cluster, '#1f77b4'),
                      alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(common_taxa, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Taxa Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted Loading', fontsize=12, fontweight='bold')
    ax.set_title('Taxa Loading Comparison Across Clusters', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig
