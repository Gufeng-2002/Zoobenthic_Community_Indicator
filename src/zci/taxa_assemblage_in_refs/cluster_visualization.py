"""
Visualization tools for hierarchical clustering results of reference sites.

This module provides functions to visualize:
1. Geographic distribution of reference sites with cluster assignments
2. Environmental variables across clusters
3. Taxa composition across clusters

Author: Generated for Thesis Project
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Dict, List, Tuple
import warnings
from scipy import stats
import geopandas as gpd
import sys
import os


def _plot_rivers_lakes_background(ax, annotating=False):
    """Plot lakes and rivers background with absolute paths."""
    try:
        # Get the path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        data_dir = os.path.join(project_root, 'data', 'maps')
        
        # Read shapefiles with absolute paths
        lake_stclair = gpd.read_file(os.path.join(data_dir, 'lake_stclair', 'lake_stclair.shp'))
        lake_erie = gpd.read_file(os.path.join(data_dir, 'lake_erie', 'lake_erie.shp'))
        detroit_river = gpd.read_file(os.path.join(data_dir, 'detroit_river_aoc_shapefile', 'AOC_MI_Detroit_2021.shp'))
        stclair_river = gpd.read_file(os.path.join(data_dir, 'aoc_mi_stclair_2021', 'AOC_MI_StClair_2021.shp'))
        lake_huron = gpd.read_file(os.path.join(data_dir, 'lake_huron', 'lake_huron.shp'))
        
        # Plot water bodies
        lake_stclair.plot(ax=ax, color='lightblue', edgecolor='none', alpha=0.5)
        lake_erie.plot(ax=ax, color='lightblue', edgecolor='none', alpha=0.5)
        lake_huron.plot(ax=ax, color='lightblue', edgecolor='none', alpha=0.5)
        detroit_river.plot(ax=ax, color='lightblue', edgecolor='none')
        stclair_river.plot(ax=ax, color='lightblue', edgecolor='none')
        
        ax.set_ylim(42, 43.1)
        ax.set_xlim(-83.3, -82.3)
        
        if annotating:
            ax.text(-83.0, 42.2, 'Detroit River', fontsize=8, color='gray', style='italic')
            ax.text(-82.85, 42.9, 'St. Clair River', fontsize=8, color='gray', style='italic')
            ax.text(-82.55, 42.05, 'Lake Erie', fontsize=8, color='gray', style='italic')
            ax.text(-83.0, 42.5, 'Lake St. Clair', fontsize=8, color='gray', style='italic')
            ax.text(-82.6, 43.05, 'Lake Huron', fontsize=8, color='gray', style='italic')
        
        return True
    except Exception as e:
        print(f"Warning: Could not plot rivers/lakes: {e}")
        return False


def visualize_cluster_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    top_n_taxa: int = 16,
    env_variables: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 10),
    cluster_colors: Optional[Dict[int, str]] = None,
    standardize_env: bool = False
) -> plt.Figure:
    """
    Create a comprehensive 3-panel visualization of cluster analysis results.
    
    Displays:
    - Left panel: Map of reference sites colored by cluster assignment
    - Upper right: Environmental variable means across clusters
    - Lower right: Top taxa composition (Hellinger transformed) across clusters
    
    Parameters:
    ----------
    raw_data : pd.DataFrame
        Raw data containing cluster assignments, coordinates, and if_ref column
    multiindex_data : pd.DataFrame
        Multi-index data with environmental and taxa information
    cluster_column : str, default='clusters'
        Name of the cluster assignment column in raw_data
    top_n_taxa : int, default=15
        Number of top taxa to display in composition plot
    env_variables : list of str, optional
        Specific environmental variables to plot. If None, uses common ones.
        Example: ['depth_m', 'velocity_m_s', 'substrate_index']
    figsize : tuple, default=(18, 10)
        Figure size (width, height)
    cluster_colors : dict, optional
        Mapping of cluster IDs to colors. If None, uses default color scheme.
        Example: {0: 'orange', 1: 'green', 2: 'red'}
    standardize_env : bool, default=False
        If True, plot z-scores of environmental variables.
        If False, plot raw mean values.
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with 3 panels
        
    Examples:
    --------
    >>> fig = visualize_cluster_analysis(
    ...     raw_data, data,
    ...     cluster_column='clusters',
    ...     top_n_taxa=15
    ... )
    >>> fig.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    
    Notes:
    -----
    - Requires geographic coordinates (Longitude, Latitude) in raw_data
    - Only plots reference sites (if_ref == True)
    - Taxa composition uses Hellinger-transformed values
    """
    # Filter for reference sites only
    ref_mask = raw_data['if_ref'] == True
    ref_data = raw_data[ref_mask].copy()
    ref_multiindex = multiindex_data[ref_mask].copy()
    
    # Check if cluster column exists
    if cluster_column not in ref_data.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in raw_data")
    
    # Get number of clusters
    n_clusters = ref_data[cluster_column].nunique()
    cluster_ids = sorted(ref_data[cluster_column].unique())
    
    # Set default colors if not provided (matching pasted figure)
    if cluster_colors is None:
        # Blue, orange, green to match the reference figure exactly
        default_colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#DC143C', '#9370DB', '#FFD700']
        cluster_colors = {cid: default_colors[i % len(default_colors)] 
                         for i, cid in enumerate(cluster_ids)}
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.05)  # Reduced wspace from 0.3 to 0.2
    
    # Left panel: Map
    ax_map = fig.add_subplot(gs[:, 0])
    _plot_cluster_map(ref_data, cluster_column, cluster_colors, cluster_ids, ax_map)
    
    # Upper right: Environmental variables
    ax_env = fig.add_subplot(gs[0, 1])
    _plot_environmental_variables(ref_data, ref_multiindex, cluster_column, 
                                  cluster_colors, cluster_ids, env_variables, ax_env,
                                  standardize=standardize_env)
    
    # Lower right: Taxa composition
    ax_taxa = fig.add_subplot(gs[1, 1])
    _plot_taxa_composition(ref_multiindex, ref_data, cluster_column, 
                          cluster_colors, cluster_ids, top_n_taxa, ax_taxa)
    
    # Add overall title
    fig.suptitle('Reference Sites Cluster Analysis - Geographic, Environmental & Taxa Patterns',
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def _plot_cluster_map(ref_data: pd.DataFrame,
                     cluster_column: str,
                     cluster_colors: Dict[int, str],
                     cluster_ids: List[int],
                     ax: plt.Axes) -> None:
    """Plot reference sites on a map colored by cluster assignment."""
    # Check for coordinate columns
    if 'Longitude' not in ref_data.columns or 'Latitude' not in ref_data.columns:
        ax.text(0.5, 0.5, 'Geographic coordinates\nnot available',
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Reference Sites by Cluster', fontsize=12, fontweight='bold')
        return
    
    # Plot lakes and rivers background
    if _plot_rivers_lakes_background(ax, annotating=True):
        # Map styling is handled by the background function
        pass
    else:
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    
    # Plot each cluster
    for cluster_id in cluster_ids:
        cluster_mask = ref_data[cluster_column] == cluster_id
        cluster_sites = ref_data[cluster_mask]
        
        ax.scatter(cluster_sites['Longitude'], cluster_sites['Latitude'],
                  c=cluster_colors[cluster_id], s=120, alpha=0.8,
                  edgecolors='black', linewidth=1.5,
                  label=f'Cluster {int(cluster_id) + 1} (n={len(cluster_sites)})',
                  zorder=3)
        
        # Add site labels near scatter points
        for idx, (lon, lat) in enumerate(zip(cluster_sites['Longitude'], cluster_sites['Latitude'])):
            site_name = cluster_sites.index[idx]
            ax.annotate(site_name, xy=(lon, lat), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.7,
                       ha='left', va='bottom')
    # Add axis labels
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    
    # Legend - upper left with larger font
    ax.legend(loc='upper left', fontsize=15, framealpha=0.9)


def _plot_environmental_variables(ref_data: pd.DataFrame,
                                  ref_multiindex: pd.DataFrame,
                                  cluster_column: str,
                                  cluster_colors: Dict[int, str],
                                  cluster_ids: List[int],
                                  env_variables: Optional[List[str]],
                                  ax: plt.Axes,
                                  standardize: bool = False) -> None:
    """
    Plot environmental variable means across clusters.
    
    Parameters:
    ----------
    standardize : bool, default=False
        If True, plot z-scores of environmental variables.
        If False, plot raw mean values.
    """
    # Default environmental variables if not specified
    if env_variables is None:
        # Try to find common environmental variables
        possible_env_vars = ['depth_m', 'velocity_m_s', 'substrate_index', 
                            'temperature_c', 'dissolved_oxygen_mg_l']
        
        # Check which ones exist in the data
        if 'env' in ref_multiindex.columns.get_level_values(0):
            env_cols = ref_multiindex.loc[:, ref_multiindex.columns.get_level_values(0) == 'env']
            available_vars = env_cols.columns.get_level_values(-1).unique().tolist()
            env_variables = [v for v in possible_env_vars if v in available_vars][:5]
        
        if not env_variables:
            # Fallback: use any numeric columns from raw_data
            numeric_cols = ref_data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Longitude', 'Latitude', cluster_column, 'if_ref', 'Pollution_Score']
            env_variables = [col for col in numeric_cols if col not in exclude_cols][:5]
    
    if not env_variables:
        ax.text(0.5, 0.5, 'No environmental\nvariables found',
               ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Environmental Variables', fontsize=12, fontweight='bold')
        return
    
    # Calculate means for each cluster
    cluster_means = []
    actual_vars_used = []  # Track which variables actually have data
    
    for cluster_id in cluster_ids:
        cluster_mask = ref_data[cluster_column] == cluster_id
        
        # Try to get data from multiindex first, then raw_data
        var_means = []
        for var in env_variables:
            value_found = False
            # Try multiindex
            try:
                if 'env' in ref_multiindex.columns.get_level_values(0):
                    env_data = ref_multiindex.loc[cluster_mask, 
                                                  ref_multiindex.columns.get_level_values(0) == 'env']
                    if var in env_data.columns.get_level_values(-1):
                        var_col = [col for col in env_data.columns if col[-1] == var][0]
                        mean_val = env_data[var_col].mean()
                        if not np.isnan(mean_val):
                            var_means.append(mean_val)
                            value_found = True
                            if cluster_id == cluster_ids[0]:
                                actual_vars_used.append(var)
                            continue
            except Exception as e:
                pass
            
            # Fallback to raw_data
            if not value_found and var in ref_data.columns:
                mean_val = ref_data.loc[cluster_mask, var].mean()
                if not np.isnan(mean_val):
                    var_means.append(mean_val)
                    value_found = True
                    if cluster_id == cluster_ids[0]:
                        actual_vars_used.append(var)
                else:
                    var_means.append(0)
                    if cluster_id == cluster_ids[0]:
                        actual_vars_used.append(var)
            elif not value_found:
                var_means.append(0)
                if cluster_id == cluster_ids[0]:
                    actual_vars_used.append(var)
        
        cluster_means.append(var_means)
    
    # Update env_variables to only those actually used
    env_variables = actual_vars_used if actual_vars_used else env_variables
    
    # Convert to array for plotting
    cluster_means = np.array(cluster_means)
    
    # Check if we have any non-zero data
    if cluster_means.sum() == 0:
        ax.text(0.5, 0.5, 'Environmental data\nnot available',
               ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Environmental Variables', fontsize=12, fontweight='bold')
        return
    
    # Calculate means and SEM for each variable across all reference sites
    plot_means = []
    sem_values = []
    
    for i, var in enumerate(env_variables):
        # Collect all values for this variable across clusters
        all_values = []
        for cluster_id in cluster_ids:
            cluster_mask = ref_data[cluster_column] == cluster_id
            if var in ref_data.columns:
                vals = ref_data.loc[cluster_mask, var].dropna().values
                all_values.extend(vals)
        
        if len(all_values) > 0:
            # Calculate overall mean and std for z-score (if standardizing)
            overall_mean = np.mean(all_values)
            overall_std = np.std(all_values, ddof=1)
            
            # Calculate means/z-scores for each cluster
            cluster_values = []
            cluster_sems = []
            for cluster_id in cluster_ids:
                cluster_mask = ref_data[cluster_column] == cluster_id
                if var in ref_data.columns:
                    cluster_vals = ref_data.loc[cluster_mask, var].dropna().values
                    if len(cluster_vals) > 0:
                        cluster_mean = np.mean(cluster_vals)
                        sem = stats.sem(cluster_vals)
                        
                        if standardize:
                            # Z-score
                            z_score = (cluster_mean - overall_mean) / overall_std if overall_std > 0 else 0
                            cluster_values.append(z_score)
                            cluster_sems.append(sem / overall_std if overall_std > 0 else 0)  # SEM in z-score units
                        else:
                            # Raw mean
                            cluster_values.append(cluster_mean)
                            cluster_sems.append(sem)
                    else:
                        cluster_values.append(0)
                        cluster_sems.append(0)
            
            plot_means.append(cluster_values)
            sem_values.append(cluster_sems)
        else:
            plot_means.append([0] * len(cluster_ids))
            sem_values.append([0] * len(cluster_ids))
    
    plot_means = np.array(plot_means).T  # Transpose to get cluster x variable
    sem_values = np.array(sem_values).T
    
    # Create grouped bar plot with asymmetric error bars
    x = np.arange(len(env_variables))
    width = 0.25
    
    for i, cluster_id in enumerate(cluster_ids):
        offset = (i - len(cluster_ids)/2 + 0.5) * width
        
        # Create asymmetric error bars: upper only for positive, lower only for negative
        lower_errors = np.zeros_like(sem_values[i])
        upper_errors = np.zeros_like(sem_values[i])
        
        for j in range(len(plot_means[i])):
            if plot_means[i][j] >= 0:
                # Positive value: show only upper error bar
                upper_errors[j] = sem_values[i][j]
                lower_errors[j] = 0
            else:
                # Negative value: show only lower error bar
                upper_errors[j] = 0
                lower_errors[j] = sem_values[i][j]
        
        yerr_array = np.array([lower_errors, upper_errors])
        
        ax.bar(x + offset, plot_means[i], width,
              yerr=yerr_array,
              label=f'Cluster {int(cluster_id) + 1}',
              color=cluster_colors[cluster_id], alpha=0.8, edgecolor='black',
              error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})
    
    # Clean up variable names for display - create short, compact names
    clean_names = []
    for var in env_variables:
        # Create shorter versions of common variable names
        short_name = var.replace('Measured Depth (m)', 'Depth')\
                        .replace('Velocity  at bottom (m/sec)_Imputed', 'Velocity')\
                        .replace('Water DO Bottom (mg/L)', 'DO')\
                        .replace('Temperature (oC)', 'Temp')\
                        .replace('MPS (Phi)', 'Sediment')\
                        .replace('LOI (%)', 'LOI')\
                        .replace('_', ' ').title()
        clean_names.append(short_name)
    
    if standardize:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Mean z-score (± SEM)', fontsize=11, fontweight='bold')
        ax.set_title('(A) Standardized Habitat Features Across Clusters',
                    fontsize=12, fontweight='bold', loc='left')
    else:
        ax.set_ylabel('Mean Value (± SEM)', fontsize=11, fontweight='bold')
        ax.set_title('(A) Habitat Features Across Clusters',
                    fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=30, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def _plot_taxa_composition(ref_multiindex: pd.DataFrame,
                          ref_data: pd.DataFrame,
                          cluster_column: str,
                          cluster_colors: Dict[int, str],
                          cluster_ids: List[int],
                          top_n_taxa: int,
                          ax: plt.Axes) -> None:
    """Plot taxa composition (Hellinger transformed) across clusters."""
    # Extract taxa data
    if 'taxa' not in ref_multiindex.columns.get_level_values(0):
        ax.text(0.5, 0.5, 'Taxa data not found',
               ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Taxa Composition', fontsize=12, fontweight='bold')
        return
    
    taxa_data = ref_multiindex.loc[:, ref_multiindex.columns.get_level_values(0) == 'taxa']
    
    # Simplify column names
    taxa_data_simple = taxa_data.copy()
    taxa_data_simple.columns = taxa_data_simple.columns.get_level_values(-1)
    
    # Calculate mean abundance for each taxon across all reference sites
    # to identify the most abundant taxa
    overall_means = taxa_data_simple.mean(axis=0).sort_values(ascending=False)
    top_taxa = overall_means.head(top_n_taxa).index.tolist()
    
    # Calculate mean abundance and SEM for each cluster
    cluster_taxa_means = []
    cluster_taxa_sems = []
    for cluster_id in cluster_ids:
        cluster_mask = ref_data[cluster_column] == cluster_id
        cluster_data = taxa_data_simple.loc[cluster_mask, top_taxa]
        cluster_mean = cluster_data.mean(axis=0)
        cluster_sem = cluster_data.sem(axis=0)  # Standard error of mean
        cluster_taxa_means.append(cluster_mean.values)
        cluster_taxa_sems.append(cluster_sem.values)
    
    cluster_taxa_means = np.array(cluster_taxa_means)
    cluster_taxa_sems = np.array(cluster_taxa_sems)
    
    # Create grouped bar plot with asymmetric error bars (upper only since all values are positive)
    x = np.arange(len(top_taxa))
    width = 0.25
    
    for i, cluster_id in enumerate(cluster_ids):
        offset = (i - len(cluster_ids)/2 + 0.5) * width
        
        # For taxa (all positive values), show only upper error bars
        lower_errors = np.zeros_like(cluster_taxa_sems[i])
        upper_errors = cluster_taxa_sems[i]
        yerr_array = np.array([lower_errors, upper_errors])
        
        ax.bar(x + offset, cluster_taxa_means[i], width,
              yerr=yerr_array,
              label=f'Cluster {int(cluster_id) + 1}',
              color=cluster_colors[cluster_id], alpha=0.8, edgecolor='black',
              error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})
    
    # Clean up taxa names for display (abbreviate if too long)
    clean_taxa = [name[:15] + '...' if len(name) > 15 else name for name in top_taxa]
    
    ax.set_ylabel('Mean Hellinger Abundance (± SEM)', fontsize=11, fontweight='bold')
    ax.set_title(f'(B) Reference Sites: Taxa by Cluster',
                fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_taxa, rotation=30, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_cluster_dendrogram_with_map(
    raw_data: pd.DataFrame,
    linkage_matrix: np.ndarray,
    cluster_labels: pd.Series,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Create a 2-panel visualization with dendrogram and geographic map.
    
    Parameters:
    ----------
    raw_data : pd.DataFrame
        Raw data with coordinates and if_ref column
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    cluster_labels : pd.Series
        Cluster assignments for sites
    figsize : tuple, default=(16, 8)
        Figure size
        
    Returns:
    -------
    fig : plt.Figure
        Matplotlib figure with dendrogram (left) and map (right)
    """
    from scipy.cluster.hierarchy import dendrogram
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Dendrogram
    ref_mask = raw_data['if_ref'] == True
    ref_sites = raw_data[ref_mask].index
    
    dendrogram(linkage_matrix, labels=ref_sites, leaf_rotation=90, ax=ax1)
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Reference Site', fontsize=11)
    ax1.set_ylabel('Linkage Distance', fontsize=11)
    
    # Right: Map
    n_clusters = cluster_labels.nunique()
    cluster_colors = {i: plt.cm.Set1(i) for i in range(n_clusters)}
    
    _plot_cluster_map(raw_data[ref_mask], 'clusters', cluster_colors, 
                     sorted(cluster_labels.unique()), ax2)
    
    plt.tight_layout()
    return fig
