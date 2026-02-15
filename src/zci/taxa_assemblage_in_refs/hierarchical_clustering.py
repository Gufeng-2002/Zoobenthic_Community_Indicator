"""
Hierarchical Clustering Module for Reference Sites Taxa Assemblages

This module performs hierarchical clustering analysis on species composition data 
from reference sites. It includes multiple transformation methods (Hellinger, Chord, 
Octave) and compares different linkage functions (single, complete, average, Ward's).

Key Functions:
-------------
- cluster_species_hierarchical: Main function to perform clustering analysis
- hellinger_transform: Apply Hellinger transformation to species data
- chord_transform: Apply Chord normalization to species data
- octave_transform: Apply Octave transformation to species data

References:
----------
From "Numerical Ecology with R" by Legendre and Legendre (2019), page 112
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

# Distance measure mapping
DISTANCE_METRICS = {
    'euclidean': 'euclidean',
    'cityblock': 'cityblock',
    'manhattan': 'cityblock',
    'city-block': 'cityblock',
    'city-block (manhattan)': 'cityblock'
}

def octave_to_relative_abundance(octave_df):
    """Convert octave transformed values back to relative abundance.
    """
    p = np.power(2, octave_df) - 0.0625  # inverse of log2 transformation and offset
    # handle the zero values (originally zero before octave transform)
    p = p.clip(lower=0)  # set negative values to zero
    row_sum = p.sum(axis=1)
    # normalize to relative abundance
    p = p.div(row_sum, axis=0)
    p = p.fillna(0)  # handle division by zero
    return p


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Hellinger transformation to taxa data.
    
    The Hellinger transformation first calculates relative abundances (same as chord 
    transformation start), then takes the square root of relative abundances. This gives 
    higher weight to abundant taxa and less weight to rare taxa.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Species abundance data with sites as rows and species as columns.
        Can be raw counts or relative abundances (proportions).
        
    Returns:
    -------
    pd.DataFrame
        Hellinger-transformed species data
        
    Notes:
    -----
    Transformation formula:
    1. If raw counts: Calculate relative abundances: p_ij = x_ij / sum(x_i)
       If already proportions: Use p_ij directly
    2. Take square root: h_ij = sqrt(p_ij)
    
    The Euclidean distance on Hellinger-transformed data is called the Hellinger distance.
    """
    # convert octave transformed data back to relative abundance
    df = octave_to_relative_abundance(df)
    # Take square root of relative abundances
    hellinger_df = np.sqrt(df)
    
    return hellinger_df


def chord_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Chord normalization to taxa data.
    
    The chord-normalized relative abundance normalizes each site vector to unit length,
    placing all sites on the unit hypersphere in the m-dimensional taxa space, where 
    Euclidean distances between sites are determined by composition differences 
    (direction) only.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Species abundance data with sites as rows and species as columns.
        Can be raw counts or relative abundances (proportions).
        
    Returns:
    -------
    pd.DataFrame
        Chord-transformed species data
        
    Notes:
    -----
    Transformation formula:
    c_ij = x_ij / sqrt(sum(x_i^2))
    
    where x_ij is either raw counts or relative abundances depending on input.
    
    Each site vector has norm = 1: ||c_i|| = sqrt(sum(c_ij^2)) = 1
    
    The Euclidean distance on chord-transformed data is called the Chord distance.
    
    Note: When input is relative abundance, the chord transformation still normalizes
    to unit length. The difference is that relative abundances sum to 1, while
    chord-transformed data has L2 norm = 1.
    """
    # For chord transformation, we normalize to unit length regardless of input type
    # The is_relative_abundance parameter is provided for consistency and documentation
    # but the transformation formula remains the same
    # convert octave transformed data back to relative abundance
    df = octave_to_relative_abundance(df)
    
    # Calculate the L2 norm of each site's taxa vector (row-wise)
    norm_rows = np.sqrt((df ** 2).sum(axis=1))
    
    # Divide each taxa value by the site's norm
    chord_df = df.div(norm_rows, axis=0)
    
    return chord_df


def octave_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Octave transformation to taxa data.
    
    The Octave transformation first converts abundances to proportions, then applies
    a log2 transformation with a small constant to handle zeros. This transformation
    is useful for reducing the influence of very abundant species.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Species abundance data with sites as rows and species as columns.
        Can be raw counts or relative abundances (proportions).
        
    Returns:
    -------
    pd.DataFrame
        Octave-transformed species data
        
    Notes:
    -----
    Transformation formula:
    1. If raw counts: Calculate proportions: p_ij = x_ij / sum(x_i)
       If already proportions: Use p_ij directly
    2. Apply log transformation: o_ij = log2(100 * (p_ij + 0.01))
    
    The constant 0.01 prevents log(0) and the multiplication by 100 scales the values.
    """
    # the input data is already octave transformed relative abundance
    octave_df = df.copy()
    
    return octave_df


def _plot_fusion_level(Z: np.ndarray, method_name: str, 
                       transformation_method: str = "Hellinger",
                       distance_metric: str = "Euclidean",
                       ax: Optional[plt.Axes] = None) -> None:
    """
    Plot fusion levels as a step plot showing how many clusters remain at each node height.
    
    Fusion-level plots visualize how clusters are formed at each merge step. Each node 
    shows the group number at each merge step (x-axis: fusion height; y-axis: group number).
    The segment/plateau to its right indicates the linkage distance used to merge two 
    clusters at that step.
    
    Reading the plot:
    - Longer plateau → longer distance between clusters → more distinct/stable clusters
    - Use this to identify good cutting levels (number of clusters)
    
    Parameters:
    ----------
    Z : np.ndarray
        Linkage matrix from hierarchical clustering
    method_name : str
        Name of the linkage method (e.g., 'complete', 'ward')
    transformation_method : str, default='Hellinger'
        Name of the transformation method
    distance_metric : str, default='Euclidean'
        Name of the distance metric used
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes
        
    Notes:
    -----
    Based on "Numerical Ecology with R" fusion level plot concept.
    """
    if ax is None:
        ax = plt.gca()
    
    n = Z.shape[0] + 1  # number of original observations
    heights = Z[:, 2]   # fusion heights
    
    # k goes from n down to 1 as we merge clusters
    k_vals = np.arange(n, 0, -1)
    
    # Create extended arrays for step plot
    # Start with n clusters at height 0, then decrease as we merge
    h_extended = np.concatenate([[0], heights])
    k_extended = k_vals
    
    # Plot as step function (post style keeps horizontal line then drops)
    ax.step(h_extended, k_extended, where='post', linewidth=1.5, color='red')
    
    # Add markers at fusion points
    ax.plot(heights, k_vals[1:], 'o', markersize=4, color='red', alpha=0.7)
    
    # Add text labels for selected k values
    label_positions = [n, n//2, n//4, 5, 3, 2]
    for k in label_positions:
        if k <= n and k >= 1:
            idx = n - k
            if idx < len(heights):
                h = heights[idx] if idx > 0 else 0
                ax.text(h, k, str(k), fontsize=8, ha='left', va='center', 
                       color='red', fontweight='normal')
    
    ax.set_xlabel("h (node height)", fontsize=10)
    ax.set_ylabel("k (number of clusters)", fontsize=10)
    ax.set_title(f"Fusion levels - {method_name.capitalize()}\n{distance_metric} on {transformation_method} trans", 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight Ward method with yellow background
    if method_name.lower() == 'ward':
        ax.set_facecolor('#ffffcc')


def _create_dendrogram_comparison_figure(transformed_data: pd.DataFrame,
                                         transformation_name: str,
                                         distance_metric: str = 'euclidean',
                                         site_labels: Optional[pd.Index] = None) -> plt.Figure:
    """
    Create a 4-panel figure showing dendrograms for different linkage methods.
    
    This function creates dendrograms using four linkage methods (single, complete, 
    average, Ward's) to visually compare how different methods cluster the sites.
    
    Parameters:
    ----------
    transformed_data : pd.DataFrame
        Transformed species data (Hellinger, Chord, or Octave)
    transformation_name : str
        Name of the transformation method for labeling
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cityblock')
    site_labels : pd.Index, optional
        Site labels for dendrograms. If None, uses transformed_data.index
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with 4 dendrogram subplots
        
    Notes:
    -----
    Ward's method always uses Euclidean distance regardless of distance_metric setting.
    """
    if site_labels is None:
        site_labels = transformed_data.index
    
    methods = ["single", "complete", "average", "ward"]
    metric_display = "Manhattan" if distance_metric == 'cityblock' else "Euclidean"
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.ravel()
    
    for ax, method in zip(axs, methods):
        # Compute linkage - Ward's method requires Euclidean distance
        if method == "ward":
            # Ward's method always uses Euclidean
            Z = linkage(transformed_data, method=method)
            metric_used = "Euclidean"
        else:
            # Other methods can use precomputed distances
            D_condensed = pdist(transformed_data, metric=distance_metric)
            Z = linkage(D_condensed, method=method)
            metric_used = metric_display
        
        # Create dendrogram
        dendrogram(Z, labels=site_labels, leaf_rotation=90, ax=ax)
        
        # Highlight Ward's method
        if method == "ward":
            ax.set_facecolor('#ffffcc')
            title_suffix = " (★ Preferred)" if distance_metric == 'euclidean' else " (Euclidean only)"
            ax.set_title(f"{method.capitalize()} linkage{title_suffix}\n({metric_used} dist)", 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{method.capitalize()} linkage\n({metric_used} dist)", 
                        fontsize=12, fontweight='bold')
        
        ax.set_xlabel("Reference Site", fontsize=10)
        ax.set_ylabel("Linkage Distance", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Dendrogram Comparison - {transformation_name} Transformation\n"
                f"(Distance: {metric_display} | Ward's always uses Euclidean)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def _create_linkage_comparison_figure(transformed_data: pd.DataFrame,
                                     transformation_name: str,
                                     distance_metric: str = 'euclidean') -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create a 4-panel figure comparing different linkage methods using cophenetic correlation.
    
    This function compares four linkage methods (single, complete, average, Ward's) by
    computing the cophenetic correlation coefficient for each. The cophenetic correlation
    measures how well the dendrogram preserves the original pairwise distances.
    
    Parameters:
    ----------
    transformed_data : pd.DataFrame
        Transformed species data (Hellinger, Chord, or Octave)
    transformation_name : str
        Name of the transformation method for labeling
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cityblock')
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with 4 subplots
    coph_corrs : dict
        Dictionary mapping method names to cophenetic correlation values
        
    Notes:
    -----
    Higher cophenetic correlation indicates better preservation of original distances.
    However, Ward's method is often favored for balanced cluster sizes despite potentially
    lower cophenetic correlation. Ward's method always uses Euclidean distance.
    """
    methods = ["single", "complete", "average", "ward"]
    coph_corrs = {}
    metric_display = "Manhattan" if distance_metric == 'cityblock' else "Euclidean"
    
    # Compute condensed distance matrix with specified metric
    D_condensed = pdist(transformed_data, metric=distance_metric)
    # Always compute Euclidean for Ward's comparison
    D_euclidean = pdist(transformed_data, metric='euclidean') if distance_metric != 'euclidean' else D_condensed
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    
    for ax, method in zip(axs, methods):
        # Use transformed data directly for Ward's method (always Euclidean)
        if method == "ward":
            Z = linkage(transformed_data, method=method)
            # For Ward's, compare against Euclidean distances
            coph_corr, coph_dists = cophenet(Z, D_euclidean)
            D_compare = D_euclidean
            metric_used = "Euclidean"
        else:
            Z = linkage(D_condensed, method=method)
            coph_corr, coph_dists = cophenet(Z, D_condensed)
            D_compare = D_condensed
            metric_used = metric_display
        
        coph_corrs[method] = coph_corr
        
        # Scatter plot of original vs cophenetic distances
        ax.scatter(D_compare, coph_dists, s=10, alpha=0.6, color='steelblue')
        ax.plot([D_compare.min(), D_compare.max()], 
               [D_compare.min(), D_compare.max()], 
               "k--", linewidth=1.5, label='Perfect correlation')
        
        # Highlight Ward's method
        if method == "ward":
            ax.set_facecolor('#ffffcc')
            title_suffix = " (★ Preferred)" if distance_metric == 'euclidean' else " (Euclidean only)"
            ax.set_title(f"{method.capitalize()} linkage{title_suffix}\n"
                        f"Cophenetic corr = {coph_corr:.3f} ({metric_used})", 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{method.capitalize()} linkage\n"
                        f"Cophenetic corr = {coph_corr:.3f} ({metric_used})", 
                        fontsize=12, fontweight='bold')
        
        ax.set_xlabel("Original distance", fontsize=10)
        ax.set_ylabel("Cophenetic distance", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Linkage Method Comparison - {transformation_name} Transformation\n"
                f"(Distance: {metric_display} | Higher cophenetic corr = better preservation)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, coph_corrs


def _create_fusion_comparison_figure(transformed_data: pd.DataFrame,
                                     transformation_name: str,
                                     distance_metric: str = 'euclidean') -> plt.Figure:
    """
    Create a 4-panel figure showing fusion plots for different linkage methods.
    
    Fusion plots help identify good cutting levels by showing how clusters are formed
    at each merge step. Longer plateaus indicate more stable cluster configurations.
    
    Parameters:
    ----------
    transformed_data : pd.DataFrame
        Transformed species data
    transformation_name : str
        Name of the transformation method for labeling
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cityblock')
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with 4 fusion plots
        
    Notes:
    -----
    Ward's method always uses Euclidean distance regardless of distance_metric setting.
    """
    methods = ["single", "complete", "average", "ward"]
    metric_display = "Manhattan" if distance_metric == 'cityblock' else "Euclidean"
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    
    for ax, method in zip(axs, methods):
        # Ward's method always uses Euclidean
        if method == "ward":
            Z = linkage(transformed_data, method=method)
            metric_used = "Euclidean"
        else:
            D_condensed = pdist(transformed_data, metric=distance_metric)
            Z = linkage(D_condensed, method=method)
            metric_used = metric_display
        _plot_fusion_level(Z, method, transformation_name, metric_used, ax=ax)
    
    fig.suptitle(f"Fusion Level Comparison - {transformation_name} Transformation\n"
                f"(Distance: {metric_display} | Longer plateaus = more stable clusters)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def _create_ward_analysis_figure(transformed_data: pd.DataFrame,
                                 transformation_name: str,
                                 distance_measure = 'euclidean',
                                 n_clusters: int = 3,
                                 site_labels: Optional[pd.Index] = None,
                                 label_positions: Optional[list] = None,
                                 distance_metric: str = 'euclidean') -> plt.Figure:
    """
    Create a 2-panel figure with fusion plot and dendrogram for Ward's method.
    
    This focused analysis on Ward's method shows both the fusion levels (to justify
    the choice of k clusters) and the dendrogram with cluster assignments visualized.
    
    Parameters:
    ----------
    transformed_data : pd.DataFrame
        Transformed species data
    transformation_name : str
        Name of the transformation method
    distance_measure: str, default='euclidean'
        Distance measure requested. Note: Ward's method always uses Euclidean distance.
        But 'cityblock' is optional by huristic approximation.
    n_clusters : int, default=3
        Number of clusters to highlight in the dendrogram
    site_labels : pd.Index, optional
        Site labels for dendrogram. If None, uses transformed_data.index
    label_positions : list of float, optional
        Manual x-positions for cluster labels. If None, uses midpoint of each cluster.
        Should have length equal to n_clusters. Example: [10, 25, 45]
    distance_metric : str, default='euclidean'
        Distance metric requested. Note: Ward's method always uses Euclidean distance.
        This parameter is for display purposes and consistency.
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with fusion plot (left) and dendrogram (right)
        
    Notes:
    -----
    Ward's method always uses Euclidean distance regardless of distance_metric setting.
    If a non-Euclidean metric is specified, a note will be displayed in the figure.
    """
    if site_labels is None:
        site_labels = transformed_data.index
    
    # Compute linkage - Ward's method always uses Euclidean
    if distance_measure == "cityblock":
        from scipy.spatial.distance import pdist, squareform
        D_city = pdist(transformed_data, metric='cityblock')
        Z = linkage(D_city, method='ward')
    elif distance_measure == "euclidean":
        Z = linkage(transformed_data, method='ward')
    
    # Determine display text for distance metric
    metric_display = distance_measure  # Ward's always uses Euclidean
    metric_note = "" if distance_measure == 'euclidean' else " (Ward's requires Euclidean)"
    
    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: Fusion plot
    _plot_fusion_level(Z, 'ward', transformation_name, metric_display, ax=ax1)
    
    # Add vertical shaded region at chosen k (narrow band around fusion height)
    n = Z.shape[0] + 1
    k_idx = n - n_clusters
    if k_idx > 0 and k_idx < len(Z):
        fusion_height = Z[k_idx, 2]
        # Calculate a narrow band width (5% of the x-axis range)
        x_range = ax1.get_xlim()[1] - ax1.get_xlim()[0]
        band_width = x_range * 0.05
        # Add narrow vertical shadow at the fusion height for k clusters
        ax1.axvspan(fusion_height - 0.5 * band_width, fusion_height + 0.5 * band_width, alpha=0.2, color='blue', 
                   label=f'k = {n_clusters} clusters')
        ax1.legend(fontsize=10, loc='upper right')
    
    # Right panel: Dendrogram with cluster assignments
    # First get cluster assignments before creating dendrogram
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(transformed_data)
    
    # Create mapping from site to cluster
    site_to_cluster = dict(zip(site_labels, cluster_labels))
    
    # Create dendrogram
    dend_result = dendrogram(Z, labels=site_labels, leaf_rotation=90, ax=ax2)
    
    # Draw horizontal line at cut height
    if k_idx > 0 and k_idx < len(Z):
        ax2.axhline(y=fusion_height, color='blue', linestyle='--', linewidth=2,
                   label=f'Cut for k = {n_clusters}', alpha=0.7)
    
    # Get leaf positions and labels from dendrogram
    leaf_labels = dend_result['ivl']  # Ordered leaf labels
    
    # Find cluster boundaries and label positions
    # Group consecutive leaves by cluster
    cluster_groups = []
    current_cluster = None
    start_pos = 0
    
    for pos, leaf_label in enumerate(leaf_labels):
        cluster_id = site_to_cluster[leaf_label]
        
        # Check if we're starting a new cluster group
        if cluster_id != current_cluster:
            # If not the first cluster, save the previous cluster group
            if current_cluster is not None:
                end_pos = pos - 1
                mid_pos = (start_pos + end_pos) / 2
                cluster_groups.append((current_cluster, start_pos, end_pos, mid_pos))
            
            # Start new cluster group
            current_cluster = cluster_id
            start_pos = pos
    
    # Add the last cluster group
    if current_cluster is not None:
        end_pos = len(leaf_labels) - 1
        mid_pos = (start_pos + end_pos) / 2
        cluster_groups.append((current_cluster, start_pos, end_pos, mid_pos))
    
    # Add cluster group labels near the cut line
    for cluster_id, start_pos, end_pos, mid_pos in cluster_groups:
        # Place text slightly above the cut line
        if k_idx > 0 and k_idx < len(Z):
            label_y = fusion_height * 1.1
        else:
            label_y = ax2.get_ylim()[1] * 0.95
        
        # Use manual position if provided, otherwise use midpoint
        if label_positions is not None and cluster_id < len(label_positions):
            label_x = label_positions[cluster_id]
        else:
            label_x = mid_pos
            
        ax2.text(label_x, label_y, f'Group {cluster_id}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax2.set_title(f"Ward's Dendrogram - {transformation_name} Transformation\n"
                 f"(k = {n_clusters} clusters, {metric_display} distance)", 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel("Reference Site", fontsize=11)
    ax2.set_ylabel("Linkage Distance", fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10, loc='upper right')
    
    fig.suptitle(f"Ward's Method Analysis - {transformation_name} Transformation{metric_note}", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def cluster_species_hierarchical(
    species_data: pd.DataFrame,
    transformation: str = 'hellinger',
    distance_measure: str = 'euclidean',
    n_clusters: int = 3,
    create_dendrogram_comparison: bool = True,
    create_comparison_plots: bool = True,
    create_fusion_plots: bool = True,
    create_ward_analysis: bool = True,
    label_positions: Optional[list] = None
) -> Tuple[pd.Series, Dict[str, any]]:
    """
    Perform hierarchical clustering on species composition data of reference sites.
    
    This function applies one of three transformations (Hellinger, Chord, or Octave) to 
    the species data, then performs hierarchical clustering using Ward's linkage method.
    It can optionally create visualizations comparing different linkage methods and 
    showing fusion levels to help justify the choice of number of clusters.
    
    Parameters:
    ----------
    species_data : pd.DataFrame
        Raw species abundance data with sites as rows and species as columns
    transformation : str, default='hellinger'
        Transformation method to use: 'hellinger', 'chord', or 'octave'
    distance_measure : str, default='euclidean'
        Distance measure to use. Accepted values:
        - 'euclidean': Euclidean distance (default, recommended for Ward's method)
        - 'cityblock', 'manhattan', 'city-block', 'city-block (manhattan)': Manhattan distance
        Note: Ward's linkage method ALWAYS uses Euclidean distance regardless of this setting.
        The Manhattan distance option affects other linkage methods (single, complete, average)
        in the comparison plots.
    n_clusters : int, default=3
        Number of clusters to create
    create_dendrogram_comparison : bool, default=True
        Whether to create 4-panel dendrogram comparison for different linkage methods
    create_comparison_plots : bool, default=True
        Whether to create 4-panel linkage comparison plots (cophenetic correlation)
    create_fusion_plots : bool, default=True
        Whether to create 4-panel fusion level comparison plots
    create_ward_analysis : bool, default=True
        Whether to create 2-panel Ward's method analysis (fusion + dendrogram)
    label_positions : list of float, optional
        Manual x-positions for cluster labels on Ward's dendrogram. If None, uses
        midpoint of each cluster. Should have length equal to n_clusters.
        Example: [10, 25, 45] for 3 clusters
        
    Returns:
    -------
    cluster_labels : pd.Series
        Cluster assignments for each site (0 to n_clusters-1)
    results : dict
        Dictionary containing:
        - 'transformed_data': Transformed species data
        - 'linkage_matrix': Linkage matrix from Ward's method
        - 'transformation_name': Name of transformation used
        - 'distance_metric': Distance metric used (normalized)
        - 'dendrogram_figure': Dendrogram comparison figure (if created)
        - 'cophenetic_correlations': Dict of cophenetic correlations (if comparison created)
        - 'comparison_figure': Linkage comparison figure (if created)
        - 'fusion_figure': Fusion comparison figure (if created)
        - 'ward_figure': Ward's analysis figure (if created)
        
    Raises:
    ------
    ValueError
        If transformation parameter is not one of: 'hellinger', 'chord', 'octave'
        If distance_measure parameter is not recognized
        
    Examples:
    --------
    >>> # Basic clustering with Hellinger transformation and Euclidean distance
    >>> cluster_labels, results = cluster_species_hierarchical(
    ...     taxa_ref, transformation='hellinger', n_clusters=3
    ... )
    >>> 
    >>> # Clustering with Manhattan distance for comparison plots
    >>> cluster_labels, results = cluster_species_hierarchical(
    ...     taxa_ref, transformation='hellinger', distance_measure='manhattan', n_clusters=3
    ... )
    >>> 
    >>> # Access transformed data and figures
    >>> transformed_data = results['transformed_data']
    >>> fig = results['ward_figure']
    >>> fig.savefig('ward_analysis.png', dpi=300, bbox_inches='tight')
    
    Notes:
    -----
    - Based on methods from "Numerical Ecology with R" (Legendre & Legendre, 2019)
    - Ward's method is preferred for balanced cluster sizes suitable for LDA
    - Ward's method REQUIRES Euclidean distance (variance-based algorithm)
    - Hellinger transformation is recommended for species composition data
    - The function addresses the "double-zero problem" through transformations
    - Manhattan distance can be useful for comparison with other linkage methods
    
    References:
    ----------
    Legendre, P., & Legendre, L. (2019). Numerical ecology with R. Elsevier.
    """
    # Validate transformation parameter
    transformation = transformation.lower()
    if transformation not in ['hellinger', 'chord', 'octave']:
        print(f"Caution: transformation should be 'hellinger', 'chord', or 'octave', "
                        f"got '{transformation}'")
    
    # Validate and normalize distance measure parameter
    distance_measure_lower = distance_measure.lower()
    if distance_measure_lower not in DISTANCE_METRICS:
        raise ValueError(f"distance_measure must be one of {list(DISTANCE_METRICS.keys())}, "
                        f"got '{distance_measure}'")
    distance_metric = DISTANCE_METRICS[distance_measure_lower]
    metric_display = "Manhattan" if distance_metric == 'cityblock' else "Euclidean"
    
    # Warn if non-Euclidean distance is used (Ward's always uses Euclidean)
    if distance_metric != 'euclidean':
        warnings.warn(
            f"Distance measure '{distance_measure}' was specified, but Ward's linkage method "
            f"ALWAYS uses Euclidean distance. The {metric_display} distance will only be used "
            f"for other linkage methods (single, complete, average) in comparison plots.",
            UserWarning
        )
    
    # Apply transformation
    print(f"Applying {transformation.capitalize()} transformation to species data...")
    print(f"Distance measure: {metric_display}" + 
          (" (Ward's should always use Euclidean)" if distance_metric != 'euclidean' else ""))
    if transformation == 'hellinger':
        transformed_data = hellinger_transform(species_data)
        transformation_name = "Hellinger"
    elif transformation == 'chord':
        transformed_data = chord_transform(species_data)
        transformation_name = "Chord"
    elif transformation == 'octave':
        transformed_data = octave_transform(species_data)
        transformation_name = "Octave"
    else: 
        transformed_data = species_data.copy()  # Fallback (should not reach here)
        transformation_name = "None"
    
    print(f"✓ Transformation complete. Shape: {transformed_data.shape}")
    
    # Initialize results dictionary
    results = {
        'transformed_data': transformed_data,
        'transformation_name': transformation_name,
        'distance_metric': distance_metric,
        'distance_metric_display': metric_display
    }
    
    # Create dendrogram comparison if requested
    if create_dendrogram_comparison:
        print(f"\nCreating dendrogram comparison for 4 linkage methods...")
        dend_fig = _create_dendrogram_comparison_figure(
            transformed_data, transformation_name, distance_metric, species_data.index
        )
        results['dendrogram_figure'] = dend_fig
    
    # Create comparison plots if requested
    if create_comparison_plots:
        print(f"\nCreating linkage method comparison plots...")
        comp_fig, coph_corrs = _create_linkage_comparison_figure(
            transformed_data, transformation_name, distance_metric
        )
        results['comparison_figure'] = comp_fig
        results['cophenetic_correlations'] = coph_corrs
        
        print(f"Cophenetic correlations ({metric_display} distance" + 
              (" for non-Ward methods" if distance_metric != 'euclidean' else "") + "):")
        for method, corr in coph_corrs.items():
            metric_note = " (Euclidean)" if method == 'ward' and distance_metric != 'euclidean' else ""
            print(f"  {method.capitalize()}{metric_note}: {corr:.4f}")
    
    # Create fusion plots if requested
    if create_fusion_plots:
        print(f"\nCreating fusion level comparison plots...")
        fusion_fig = _create_fusion_comparison_figure(
            transformed_data, transformation_name, distance_metric
        )
        results['fusion_figure'] = fusion_fig
    
    
    # Perform Ward's clustering (always uses Euclidean distance)
    print(f"\nPerforming Ward's hierarchical clustering with k={n_clusters}...")
    print(f"  (Ward's method should always use Euclidean distance),"
          " and city-block metrics can only be huristically approximated)")
    if distance_measure == "cityblock":
        from scipy.spatial.distance import pdist, squareform
        D_city = pdist(transformed_data, metric='cityblock')
        Z = linkage(D_city, method='ward')
    elif distance_measure == "euclidean":
        Z = linkage(transformed_data, method='ward') # waiting to modify to euclidean
    results['linkage_matrix'] = Z
    
    # Get cluster assignments using sklearn's AgglomerativeClustering
    # Note: Ward's linkage only supports Euclidean distance in sklearn
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    cluster_labels = clustering.fit_predict(transformed_data)
    
    # Convert to Series with site index
    cluster_labels = pd.Series(cluster_labels, index=species_data.index, 
                               name='Cluster_Group')
    
    print(f"✓ Clustering complete.")
    print(f"\nCluster distribution:")
    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        print(f"  Group {i}: {count} sites")
    
    # Create Ward's analysis figure if requested
    if create_ward_analysis:
        print(f"\nCreating Ward's method analysis figure...")
        ward_fig = _create_ward_analysis_figure(
            transformed_data, transformation_name, distance_measure, n_clusters, species_data.index,
            label_positions=label_positions, distance_metric=distance_metric
        )
        results['ward_figure'] = ward_fig
    
    print(f"\n{'='*60}")
    print(f"Hierarchical clustering analysis complete!")
    print(f"{'='*60}")
    
    return cluster_labels, results


# Module exports
__all__ = [
    'cluster_species_hierarchical',
    'hellinger_transform',
    'chord_transform',
    'octave_transform'
]
