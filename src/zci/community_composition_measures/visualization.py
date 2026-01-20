"""
Visualization Module for Community Composition Analysis

This module provides functions for visualizing:
- Training vs projected sites in PC space
- ZCI vs pollution relationships
- Comprehensive multi-panel figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional, Union
import warnings


def plot_ordination_comparison(
    training_coords: Dict[int, pd.DataFrame],
    projected_coords: Dict[int, pd.DataFrame],
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    variance_explained: Optional[Dict[int, np.ndarray]] = None,
    colors: Optional[Dict] = None,
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 150,
    title: Optional[str] = None,
    show_ellipses: bool = True,
    ellipse_std: float = 2.0,
    pc_plane: Tuple[int, int] = (1, 2)
) -> plt.Figure:
    """
    Visualize training vs projected sites in PC space for each cluster.
    
    Parameters:
    -----------
    training_coords : dict
        {cluster_id: DataFrame with PC coordinates for training sites}
    projected_coords : dict
        {cluster_id: DataFrame with PC coordinates for projected sites}
    raw_data : pd.DataFrame
        Raw data with cluster labels and reference indicator
    cluster_column : str
        Column name for cluster labels
    ref_column : str
        Column name for reference indicator
    variance_explained : dict, optional
        {cluster_id: array of variance explained per PC}
    colors : dict, optional
        {cluster_id: color}
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    title : str, optional
        Overall figure title
    show_ellipses : bool
        Whether to show confidence ellipses around groups
    ellipse_std : float
        Number of standard deviations for ellipse
    pc_plane : tuple
        Which PCs to plot as (pc_x, pc_y) where pc_x and pc_y are 1-indexed.
        Default (1, 2) plots PC1 vs PC2. Use (1, 3) for PC1 vs PC3, etc.
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure
    """
    clusters = sorted(training_coords.keys())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Validate PC plane specification
    pc_x, pc_y = pc_plane
    if pc_x <= 0 or pc_y <= 0:
        raise ValueError(f"PC indices must be positive. Got: pc_plane={pc_plane}")
    if pc_x == pc_y:
        raise ValueError(f"PC indices must be different. Got: pc_x={pc_x}, pc_y={pc_y}")
    
    pc_x_name = f'PC{pc_x}'
    pc_y_name = f'PC{pc_y}'
    
    # Create figure
    fig, axes = plt.subplots(1, n_clusters, figsize=figsize, dpi=dpi)
    if n_clusters == 1:
        axes = [axes]
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        train_coords = training_coords[cluster]
        proj_coords = projected_coords.get(cluster, pd.DataFrame())
        
        # Check that required PCs exist
        if pc_x_name not in train_coords.columns or pc_y_name not in train_coords.columns:
            ax.text(0.5, 0.5, f'Required PCs not found\n(need {pc_x_name} and {pc_y_name})',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title(f'Cluster {int(cluster)}\n(ERROR: missing PCs)', fontsize=12, fontweight='bold')
            continue
        
        # Get variance explained for axis labels
        if variance_explained and cluster in variance_explained:
            var_exp = variance_explained[cluster]
            var_x = var_exp[pc_x - 1] if (pc_x - 1) < len(var_exp) else 0
            var_y = var_exp[pc_y - 1] if (pc_y - 1) < len(var_exp) else 0
            xlabel = f'{pc_x_name} ({var_x:.1f}%)'
            ylabel = f'{pc_y_name} ({var_y:.1f}%)'
        else:
            xlabel = pc_x_name
            ylabel = pc_y_name
        
        # Identify reference vs non-reference among training sites
        ref_mask = raw_data[ref_column].isin([True, 1])
        
        # Plot training reference sites
        train_ref_sites = [s for s in train_coords.index if s in raw_data.index and ref_mask[s]]
        train_nonref_sites = [s for s in train_coords.index if s in raw_data.index and not ref_mask[s]]
        
        if train_ref_sites:
            ref_data = train_coords.loc[train_ref_sites]
            ax.scatter(ref_data[pc_x_name], ref_data[pc_y_name],
                      c=colors.get(cluster, '#1f77b4'),
                      marker='o', s=100, alpha=0.8,
                      edgecolors='black', linewidth=1.5,
                      label='Training (Reference)', zorder=3)
            
            # Add ellipse around reference sites
            if show_ellipses and len(ref_data) > 2:
                _add_confidence_ellipse(ax, ref_data[pc_x_name].values, ref_data[pc_y_name].values,
                                        color=colors.get(cluster, '#1f77b4'),
                                        n_std=ellipse_std, alpha=0.2)
        
        if train_nonref_sites:
            nonref_data = train_coords.loc[train_nonref_sites]
            ax.scatter(nonref_data[pc_x_name], nonref_data[pc_y_name],
                      c=colors.get(cluster, '#1f77b4'),
                      marker='s', s=80, alpha=0.5,
                      edgecolors='black', linewidth=1,
                      label='Training (Non-Ref)', zorder=2)
        
        # Plot projected sites
        if not proj_coords.empty:
            ax.scatter(proj_coords[pc_x_name], proj_coords[pc_y_name],
                      c='lightgray', marker='^', s=100, alpha=0.7,
                      edgecolors='darkgray', linewidth=1,
                      label='Projected Sites', zorder=1)
            
            # Add ellipse around projected sites
            if show_ellipses and len(proj_coords) > 2:
                _add_confidence_ellipse(ax, proj_coords[pc_x_name].values, proj_coords[pc_y_name].values,
                                        color='gray', n_std=ellipse_std, alpha=0.15)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        
        n_train = len(train_coords)
        n_proj = len(proj_coords) if not proj_coords.empty else 0
        ax.set_title(f'Cluster {int(cluster)}\n(Train: {n_train}, Proj: {n_proj})',
                    fontsize=12, fontweight='bold')
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Overall title
    if title is None:
        title = 'PCA Ordination: Training vs Projected Sites by Cluster'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def _add_confidence_ellipse(ax, x, y, color, n_std=2.0, alpha=0.2):
    """Add a confidence ellipse to the axes."""
    if len(x) < 3:
        return
    
    try:
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        mean_x, mean_y = np.mean(x), np.mean(y)
        cov = np.cov(x, y)
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Angle of rotation
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # Width and height (2 * n_std * sqrt(eigenvalue))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                         angle=angle, facecolor=color, alpha=alpha,
                         edgecolor=color, linewidth=1.5)
        ax.add_patch(ellipse)
    except Exception:
        pass  # Skip ellipse if calculation fails


def plot_zci_vs_pollution(
    zci_df: pd.DataFrame,
    colors: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 150,
    show_regression: bool = True,
    show_quantile_regression: bool = False
) -> plt.Figure:
    """
    Create scatter plots showing ZCI vs Pollution Score relationship.
    
    Parameters:
    -----------
    zci_df : pd.DataFrame
        DataFrame with ZCI values (columns: Site, Cluster, Site_Type, ZCI, Pollution_Score)
    colors : dict, optional
        {cluster_id: color}
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    show_regression : bool
        Show OLS regression line
    show_quantile_regression : bool
        Show quantile regression lines (requires statsmodels)
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure
    """
    clusters = sorted(zci_df['Cluster'].unique())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Create figure with GridSpec for better layout control
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols + 1  # Extra row for summary
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    # Individual cluster plots
    for idx, cluster in enumerate(clusters):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        cluster_data = zci_df[zci_df['Cluster'] == cluster].copy()
        color = colors.get(cluster, '#1f77b4')
        
        # Plot by site type
        for site_type, marker, alpha, size in [
            ('Reference', 'o', 0.8, 100),
            ('Training', 's', 0.6, 80),
            ('Projected', '^', 0.5, 80)
        ]:
            subset = cluster_data[cluster_data['Site_Type'] == site_type]
            if len(subset) > 0:
                ax.scatter(subset['Pollution_Score'], subset['ZCI'],
                          c=color if site_type == 'Reference' else 'lightgray',
                          marker=marker, s=size, alpha=alpha,
                          edgecolors='black', linewidth=1,
                          label=site_type)
        
        # Add regression line
        if show_regression and len(cluster_data) > 3:
            x = cluster_data['Pollution_Score'].values
            y = cluster_data['ZCI'].values
            valid = ~(np.isnan(x) | np.isnan(y))
            x, y = x[valid], y[valid]
            
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                
                pearson_r, pearson_p = pearsonr(x, y)
                ax.plot(x_line, p(x_line), '--', color=color, linewidth=2,
                       label=f'r={pearson_r:.3f}')
                
                # Add stats text
                ax.text(0.05, 0.95, f'r={pearson_r:.3f}\np={pearson_p:.4f}\nn={len(x)}',
                       transform=ax.transAxes, fontsize=8, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Pollution Score', fontsize=10)
        ax.set_ylabel('ZCI', fontsize=10)
        ax.set_title(f'Cluster {int(cluster)}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)
    
    # Combined plot in bottom row (spanning all columns)
    ax_combined = fig.add_subplot(gs[n_rows - 1, :])
    
    for cluster in clusters:
        cluster_data = zci_df[zci_df['Cluster'] == cluster]
        color = colors.get(cluster, '#1f77b4')
        
        ax_combined.scatter(cluster_data['Pollution_Score'], cluster_data['ZCI'],
                           c=color, marker='o', s=60, alpha=0.6,
                           edgecolors='black', linewidth=0.5,
                           label=f'Cluster {int(cluster)}')
    
    # Overall regression
    x_all = zci_df['Pollution_Score'].values
    y_all = zci_df['ZCI'].values
    valid = ~(np.isnan(x_all) | np.isnan(y_all))
    x_all, y_all = x_all[valid], y_all[valid]
    
    if len(x_all) > 2:
        z_all = np.polyfit(x_all, y_all, 1)
        p_all = np.poly1d(z_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        pearson_r, pearson_p = pearsonr(x_all, y_all)
        
        ax_combined.plot(x_line, p_all(x_line), 'k--', linewidth=2.5,
                        label=f'Overall (r={pearson_r:.3f})')
        
        ax_combined.text(0.95, 0.95, 
                        f'Overall: r={pearson_r:.3f}, p={pearson_p:.4f}, n={len(x_all)}',
                        transform=ax_combined.transAxes, fontsize=10, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax_combined.set_xlabel('Pollution Score', fontsize=11, fontweight='bold')
    ax_combined.set_ylabel('ZCI (Compositional Distance)', fontsize=11, fontweight='bold')
    ax_combined.set_title('Combined: ZCI vs Pollution Across All Clusters', fontsize=12, fontweight='bold')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(loc='upper left', fontsize=9)
    
    fig.suptitle('Zoobenthic Community Indicator (ZCI) vs Pollution Score',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def create_comprehensive_figure(
    pca_results: Dict,
    projected_coords: Dict,
    zci_df: pd.DataFrame,
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    colors: Optional[Dict] = None,
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 150
) -> plt.Figure:
    """
    Create a comprehensive multi-panel figure showing all analysis results.
    
    Layout:
    - Row 1: Taxa loadings for each cluster
    - Row 2: PC ordination plots (training vs projected)
    - Row 3: ZCI vs pollution scatter plots
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas()
    projected_coords : dict
        {cluster_id: DataFrame} from project_sites_to_pc_space()
    zci_df : pd.DataFrame
        ZCI DataFrame from calculate_zci_all_clusters()
    raw_data : pd.DataFrame
        Raw data with metadata
    cluster_column : str
        Column name for cluster labels
    ref_column : str
        Column name for reference indicator
    colors : dict, optional
        {cluster_id: color}
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
        
    Returns:
    --------
    plt.Figure
        The comprehensive figure
    """
    loadings = pca_results['loadings']
    training_coords = pca_results['pca_coordinates']
    variance_explained = pca_results['variance_explained']
    n_components = pca_results['n_components']
    
    clusters = sorted(loadings.keys())
    n_clusters = len(clusters)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    # Create figure with 3 rows
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(3, n_clusters, hspace=0.35, wspace=0.3)
    
    # Row 1: Taxa loadings
    for idx, cluster in enumerate(clusters):
        ax = fig.add_subplot(gs[0, idx])
        
        cluster_loadings = loadings[cluster]
        cluster_variance = variance_explained[cluster]
        n_pcs = n_components[cluster]
        
        # Calculate weighted loadings
        weights = cluster_variance / cluster_variance.sum()
        weighted = pd.Series(0.0, index=cluster_loadings.index)
        for i in range(n_pcs):
            weighted += cluster_loadings[f'PC{i+1}'].abs() * weights[i]
        
        # Get top 8 taxa
        top_taxa = weighted.nlargest(8)
        
        x = np.arange(len(top_taxa))
        ax.bar(x, top_taxa.values, color=colors.get(cluster, '#1f77b4'),
               alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(top_taxa.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Loading', fontsize=9)
        ax.set_title(f'Cluster {int(cluster)}: Taxa Loadings\n({n_pcs} PCs, {cluster_variance.sum():.1f}%)',
                    fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Row 2: PC ordination
    ref_mask = raw_data[ref_column].isin([True, 1])
    
    for idx, cluster in enumerate(clusters):
        ax = fig.add_subplot(gs[1, idx])
        
        train_coords = training_coords[cluster]
        proj_coords = projected_coords.get(cluster, pd.DataFrame())
        var_exp = variance_explained[cluster]
        
        # Plot training sites
        train_ref = [s for s in train_coords.index if s in raw_data.index and ref_mask[s]]
        train_nonref = [s for s in train_coords.index if s in raw_data.index and not ref_mask[s]]
        
        if train_ref:
            ref_data = train_coords.loc[train_ref]
            ax.scatter(ref_data['PC1'], ref_data['PC2'],
                      c=colors.get(cluster, '#1f77b4'),
                      marker='o', s=80, alpha=0.8,
                      edgecolors='black', linewidth=1,
                      label='Reference')
        
        if train_nonref:
            nonref_data = train_coords.loc[train_nonref]
            ax.scatter(nonref_data['PC1'], nonref_data['PC2'],
                      c=colors.get(cluster, '#1f77b4'),
                      marker='s', s=60, alpha=0.4,
                      edgecolors='black', linewidth=0.5,
                      label='Train (Non-Ref)')
        
        if not proj_coords.empty:
            ax.scatter(proj_coords['PC1'], proj_coords['PC2'],
                      c='lightgray', marker='^', s=70, alpha=0.6,
                      edgecolors='darkgray', linewidth=0.5,
                      label='Projected')
        
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontsize=9)
        ax.set_title(f'Cluster {int(cluster)}: Ordination', fontsize=10, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)
    
    # Row 3: ZCI vs Pollution
    for idx, cluster in enumerate(clusters):
        ax = fig.add_subplot(gs[2, idx])
        
        cluster_data = zci_df[zci_df['Cluster'] == cluster]
        color = colors.get(cluster, '#1f77b4')
        
        # Plot all points
        ax.scatter(cluster_data['Pollution_Score'], cluster_data['ZCI'],
                  c=color, marker='o', s=60, alpha=0.6,
                  edgecolors='black', linewidth=0.5)
        
        # Add regression
        x = cluster_data['Pollution_Score'].values
        y = cluster_data['ZCI'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        if len(x) > 3:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            pearson_r, _ = pearsonr(x, y)
            
            ax.plot(x_line, p(x_line), '--', color='black', linewidth=1.5)
            ax.text(0.05, 0.95, f'r={pearson_r:.3f}\nn={len(x)}',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Pollution Score', fontsize=9)
        ax.set_ylabel('ZCI', fontsize=9)
        ax.set_title(f'Cluster {int(cluster)}: ZCI vs Pollution', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Community Composition Analysis: Taxa Loadings, Ordination, and ZCI',
                fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    return fig
