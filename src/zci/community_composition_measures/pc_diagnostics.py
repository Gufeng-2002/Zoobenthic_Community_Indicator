"""
PC Diagnostics Module for Community Composition Analysis

This module provides functions for:
1. PC loadings plots for PCs accounting for >5% variance per cluster
2. Multiple regression analysis between pollution scores and individual PCs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings


def plot_pc_loadings_by_cluster(
    pca_results: Dict,
    variance_threshold_pct: float = 5.0,
    colors: Optional[Dict] = None,
    figsize_per_pc: Tuple[float, float] = (4.0, 5.0),
    dpi: int = 150,
    top_n_taxa: Optional[int] = None,
    show_values: bool = False
) -> Dict[int, plt.Figure]:
    """
    Create PC loadings plots for PCs accounting for >variance_threshold% per cluster.
    
    For each cluster, creates a figure with m vertical barplots showing the loadings
    of all m PCs that explain over the variance threshold. The x-ticks (taxa) are
    in the same order across all barplots within each figure.
    
    Parameters:
    -----------
    pca_results : dict
        Dictionary from fit_cluster_pcas() containing:
        - 'pca_models': {cluster_id: fitted PCA model}
        - 'feature_names': {cluster_id: array of taxa/feature names}
        - 'variance_explained': {cluster_id: array of variance explained per PC}
    variance_threshold_pct : float
        Minimum variance explained (%) for a PC to be included (default: 5.0)
    colors : dict, optional
        {cluster_id: color}. If None, uses default color palette
    figsize_per_pc : tuple
        (width, height) per PC subplot
    dpi : int
        Figure resolution
    top_n_taxa : int, optional
        If provided, only show the top N taxa by mean absolute loading across PCs
    show_values : bool
        Whether to show loading values on bars
        
    Returns:
    --------
    dict
        {cluster_id: matplotlib figure}
    """
    pca_models = pca_results.get('pca_models', {})
    feature_names = pca_results.get('feature_names', {})
    variance_explained = pca_results.get('variance_explained', {})
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    cluster_figures = {}
    
    for cluster in sorted(pca_models.keys()):
        pca = pca_models[cluster]
        taxa_names = feature_names.get(cluster, np.array([]))
        var_exp = variance_explained.get(cluster, np.array([]))
        
        if pca is None or len(taxa_names) == 0:
            continue
        
        # Get loadings (components_: shape = (n_components, n_features))
        loadings = pca.components_
        
        # Find PCs that exceed the variance threshold
        significant_pcs = []
        for pc_idx, var_pct in enumerate(var_exp):
            if var_pct >= variance_threshold_pct:
                significant_pcs.append(pc_idx)
        
        if len(significant_pcs) == 0:
            # No PCs exceed threshold, skip this cluster
            warnings.warn(f"Cluster {cluster}: No PCs explain ≥{variance_threshold_pct}% variance")
            continue
        
        n_pcs = len(significant_pcs)
        n_taxa = len(taxa_names)
        
        # Determine taxa order: sort by mean absolute loading across significant PCs
        mean_abs_loadings = np.mean(np.abs(loadings[significant_pcs, :]), axis=0)
        sorted_indices = np.argsort(mean_abs_loadings)[::-1]  # Descending order
        
        # Filter to top N taxa if specified
        if top_n_taxa is not None and top_n_taxa < n_taxa:
            sorted_indices = sorted_indices[:top_n_taxa]
            n_taxa = top_n_taxa
        
        sorted_taxa_names = taxa_names[sorted_indices]
        
        # Create figure with vertical barplots side by side
        fig_width = figsize_per_pc[0] * n_pcs
        fig_height = figsize_per_pc[1]
        
        fig, axes = plt.subplots(1, n_pcs, figsize=(fig_width, fig_height), dpi=dpi, sharey=True)
        if n_pcs == 1:
            axes = [axes]
        
        color = colors.get(cluster, '#1f77b4')
        
        # Y positions for taxa (horizontal bars)
        y_positions = np.arange(n_taxa)
        
        for ax_idx, pc_idx in enumerate(significant_pcs):
            ax = axes[ax_idx]
            
            # Get loadings for this PC, reordered by our consistent taxa order
            pc_loadings = loadings[pc_idx, sorted_indices]
            var_pct = var_exp[pc_idx]
            
            # Create horizontal bar plot
            bar_colors = [color if v >= 0 else '#d62728' for v in pc_loadings]
            bars = ax.barh(y_positions, pc_loadings, color=bar_colors, 
                          edgecolor='white', linewidth=0.5, alpha=0.8)
            
            # Add value labels if requested
            if show_values:
                for i, (bar, val) in enumerate(zip(bars, pc_loadings)):
                    if abs(val) > 0.01:  # Only show significant values
                        x_pos = val + 0.02 if val >= 0 else val - 0.02
                        ha = 'left' if val >= 0 else 'right'
                        ax.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, fontsize=7)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
            
            # Set title with PC number and variance explained
            ax.set_title(f'PC{pc_idx + 1}\n({var_pct:.1f}%)', 
                        fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Loading', fontsize=10)
            
            # Only show y-axis labels on the leftmost subplot
            if ax_idx == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(sorted_taxa_names, fontsize=8)
                ax.set_ylabel('Taxa', fontsize=10)
            else:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([])
            
            # Grid for readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Set consistent x-limits across subplots
            max_abs_loading = np.max(np.abs(loadings[significant_pcs, :])) * 1.1
            ax.set_xlim(-max_abs_loading, max_abs_loading)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Invert y-axis so highest loadings are at top
        axes[0].invert_yaxis()
        
        # Add overall title
        fig.suptitle(f'Cluster {int(cluster)}: PC Loadings\n(PCs explaining ≥{variance_threshold_pct}% variance)',
                    fontsize=13, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        cluster_figures[cluster] = fig
    
    return cluster_figures


def calculate_pc_pollution_regressions(
    pca_results: Dict,
    projected_coords: Dict[int, pd.DataFrame],
    raw_data: pd.DataFrame,
    pollution_column: str = 'Pollution_Score',
    cluster_column: str = 'clusters',
    variance_threshold: float = 5.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform individual regressions between each PC axis and pollution score.
    
    For each cluster, identifies PCs explaining >threshold% variance,
    then fits individual linear regressions between pollution score and
    each PC axis.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas()
    projected_coords : dict
        {cluster_id: DataFrame with PC coordinates for projected sites}
    raw_data : pd.DataFrame
        Raw data with pollution scores
    pollution_column : str
        Column name for pollution scores
    cluster_column : str
        Column name for cluster labels
    variance_threshold : float
        Minimum variance explained (%) to include a PC (default: 5.0%)
        
    Returns:
    --------
    tuple
        (summary_table: DataFrame, detailed_results: dict)
        
        summary_table columns:
        - Cluster, PC, Variance_Explained_Pct, R_squared, Adj_R_squared,
          Coefficient, Std_Error, T_statistic, P_value, N_sites
          
        detailed_results:
        - {cluster_id: {pc_name: {'model_stats': dict, 'residuals': array, ...}}}
    """
    training_coords = pca_results['pca_coordinates']
    variance_explained = pca_results['variance_explained']
    
    results_list = []
    detailed_results = {}
    
    clusters = sorted(training_coords.keys())
    
    for cluster in clusters:
        train_coords = training_coords[cluster].copy()
        proj_coords = projected_coords.get(cluster, pd.DataFrame())
        var_exp = variance_explained.get(cluster, np.array([]))
        
        # Combine training and projected coordinates
        if not proj_coords.empty:
            all_coords = pd.concat([train_coords, proj_coords])
        else:
            all_coords = train_coords
        
        # Get pollution scores for these sites
        pollution_scores = raw_data.loc[all_coords.index, pollution_column].dropna()
        
        # Find common sites
        common_sites = all_coords.index.intersection(pollution_scores.index)
        
        if len(common_sites) < 5:
            warnings.warn(f"Cluster {cluster}: Not enough sites ({len(common_sites)}) for regression")
            continue
        
        all_coords = all_coords.loc[common_sites]
        pollution_scores = pollution_scores.loc[common_sites]
        
        detailed_results[cluster] = {}
        
        # Iterate over PCs
        for pc_idx, var_pct in enumerate(var_exp):
            if var_pct < variance_threshold:
                continue
                
            pc_col = f'PC{pc_idx + 1}'
            if pc_col not in all_coords.columns:
                continue
            
            pc_values = all_coords[pc_col].values
            poll_values = pollution_scores.values
            
            # Remove any NaN
            valid_mask = ~(np.isnan(pc_values) | np.isnan(poll_values))
            pc_valid = pc_values[valid_mask]
            poll_valid = poll_values[valid_mask]
            
            n = len(pc_valid)
            if n < 5:
                continue
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(poll_valid, pc_valid)
            
            r_squared = r_value ** 2
            # Adjusted R-squared
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
            
            # T-statistic
            t_stat = slope / std_err if std_err > 0 else np.nan
            
            # Store results
            results_list.append({
                'Cluster': int(cluster),
                'PC': pc_col,
                'Variance_Explained_Pct': var_pct,
                'R_squared': r_squared,
                'Adj_R_squared': adj_r_squared,
                'Coefficient': slope,
                'Std_Error': std_err,
                'T_statistic': t_stat,
                'P_value': p_value,
                'N_sites': n
            })
            
            # Store detailed results
            residuals = pc_valid - (intercept + slope * poll_valid)
            detailed_results[cluster][pc_col] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'p_value': p_value,
                'residuals': residuals,
                'predicted': intercept + slope * poll_valid,
                'actual': pc_valid,
                'pollution': poll_valid
            }
    
    # Create summary DataFrame
    summary_table = pd.DataFrame(results_list)
    
    if not summary_table.empty:
        summary_table = summary_table.sort_values(['Cluster', 'PC']).reset_index(drop=True)
    
    return summary_table, detailed_results


def plot_pc_pollution_regressions(
    regression_table: pd.DataFrame,
    detailed_results: Dict,
    colors: Optional[Dict] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    max_pcs_per_row: int = 4
) -> plt.Figure:
    """
    Visualize PC vs Pollution regressions for all clusters and significant PCs.
    
    Parameters:
    -----------
    regression_table : pd.DataFrame
        Output from calculate_pc_pollution_regressions()
    detailed_results : dict
        Detailed results from calculate_pc_pollution_regressions()
    colors : dict, optional
        {cluster_id: color}
    figsize : tuple, optional
        Figure size. If None, calculated automatically
    dpi : int
        Figure resolution
    max_pcs_per_row : int
        Maximum number of PC plots per row for each cluster
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure
    """
    if regression_table.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No regression results to display',
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    clusters = sorted(regression_table['Cluster'].unique())
    
    # Calculate subplot grid
    n_plots_per_cluster = regression_table.groupby('Cluster').size().to_dict()
    total_rows = sum((n + max_pcs_per_row - 1) // max_pcs_per_row for n in n_plots_per_cluster.values())
    
    if figsize is None:
        figsize = (4 * max_pcs_per_row, 3.5 * total_rows)
    
    fig, axes = plt.subplots(total_rows, max_pcs_per_row, figsize=figsize, dpi=dpi)
    if total_rows == 1:
        axes = [axes]
    if max_pcs_per_row == 1:
        axes = [[ax] for ax in axes]
    
    row_idx = 0
    
    for cluster in clusters:
        cluster_data = regression_table[regression_table['Cluster'] == cluster]
        color = colors.get(cluster, '#1f77b4')
        
        for pc_idx, (_, row) in enumerate(cluster_data.iterrows()):
            pc_name = row['PC']
            col_idx = pc_idx % max_pcs_per_row
            current_row = row_idx + pc_idx // max_pcs_per_row
            
            ax = axes[current_row][col_idx]
            
            # Get detailed data
            if cluster in detailed_results and pc_name in detailed_results[cluster]:
                details = detailed_results[cluster][pc_name]
                
                # Scatter plot
                ax.scatter(details['pollution'], details['actual'],
                          c=color, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
                
                # Regression line
                x_line = np.linspace(details['pollution'].min(), details['pollution'].max(), 100)
                y_line = details['intercept'] + details['slope'] * x_line
                ax.plot(x_line, y_line, color='red', linewidth=2, linestyle='-')
                
                # Add regression stats
                stats_text = (f'R² = {row["R_squared"]:.3f}\n'
                             f'p = {row["P_value"]:.4f}\n'
                             f'β = {row["Coefficient"]:.3f}')
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Pollution Score', fontsize=9)
            ax.set_ylabel(pc_name, fontsize=9)
            ax.set_title(f'Cluster {int(cluster)}: {pc_name} ({row["Variance_Explained_Pct"]:.1f}%)',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots in this cluster's rows
        n_pcs = len(cluster_data)
        n_rows_for_cluster = (n_pcs + max_pcs_per_row - 1) // max_pcs_per_row
        
        for extra_idx in range(n_pcs, n_rows_for_cluster * max_pcs_per_row):
            col_idx = extra_idx % max_pcs_per_row
            current_row = row_idx + extra_idx // max_pcs_per_row
            axes[current_row][col_idx].axis('off')
        
        row_idx += n_rows_for_cluster
    
    fig.suptitle('PC vs Pollution Score Regressions by Cluster\n(PCs explaining ≥5% variance)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def create_pc_regression_summary_table(
    regression_table: pd.DataFrame,
    format_pvalue: bool = True
) -> pd.DataFrame:
    """
    Create a formatted summary table of PC-Pollution regressions for display.
    
    Parameters:
    -----------
    regression_table : pd.DataFrame
        Output from calculate_pc_pollution_regressions()
    format_pvalue : bool
        Whether to format p-values with significance stars
        
    Returns:
    --------
    pd.DataFrame
        Formatted summary table
    """
    if regression_table.empty:
        return pd.DataFrame()
    
    df = regression_table.copy()
    
    # Add significance column
    def significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        elif p < 0.1:
            return '.'
        else:
            return ''
    
    if format_pvalue:
        df['Significance'] = df['P_value'].apply(significance_stars)
    
    # Round numeric columns
    numeric_cols = ['Variance_Explained_Pct', 'R_squared', 'Adj_R_squared', 
                   'Coefficient', 'Std_Error', 'T_statistic', 'P_value']
    
    for col in numeric_cols:
        if col in df.columns:
            if col == 'P_value':
                df[col] = df[col].apply(lambda x: f'{x:.4f}' if x >= 0.0001 else '<0.0001')
            else:
                df[col] = df[col].round(4)
    
    return df

def extract_significant_pc_loadings(
    pca_results: Dict,
    regression_table: pd.DataFrame,
    feature_names: Optional[Dict[int, np.ndarray]] = None,
    p_value_threshold: float = 0.05,
    top_n_taxa: Optional[int] = 10
) -> pd.DataFrame:
    """
    Extract and display loadings of PCs significant in regression with pollution.
    
    For each significant PC (from regression analysis), shows the taxa with
    the highest absolute loadings.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas()
    regression_table : pd.DataFrame
        Output from calculate_pc_pollution_regressions() with R_squared, P_value, etc.
    feature_names : dict, optional
        {cluster_id: array of taxa names}. If None, uses generic names.
    p_value_threshold : float
        P-value threshold for considering PC significant (default: 0.05)
    top_n_taxa : int, optional
        Number of top taxa to show per PC (by absolute loading).
        If None, shows all taxa.
        
    Returns:
    --------
    pd.DataFrame
        Table with columns: Cluster, PC, Taxon, Loading, Loading_Abs,
        R_squared, P_value
    """
    pca_models = pca_results.get('pca_models', {})
    feature_names_dict = feature_names or pca_results.get('feature_names', {})
    
    # Filter regression results for significant PCs
    if regression_table.empty:
        return pd.DataFrame()
    
    significant_pcs = regression_table[regression_table['P_value'] < p_value_threshold].copy()
    
    if significant_pcs.empty:
        return pd.DataFrame()
    
    results = []
    
    for _, row in significant_pcs.iterrows():
        cluster = int(row['Cluster'])
        pc_name = row['PC']
        pc_idx = int(pc_name.replace('PC', '')) - 1
        
        if cluster not in pca_models or pc_idx < 0:
            continue
        
        pca = pca_models[cluster]
        taxa = feature_names_dict.get(cluster, None)
        
        if taxa is None:
            # Generate generic taxa names
            n_features = pca.n_features_in_
            taxa = np.array([f'Feature_{i+1}' for i in range(n_features)])
        
        # Get loadings for this PC
        loadings = pca.components_[pc_idx, :]
        
        # Sort by absolute loading
        abs_loadings = np.abs(loadings)
        sorted_idx = np.argsort(abs_loadings)[::-1]  # Descending order
        
        # Limit to top_n_taxa if specified
        if top_n_taxa is not None:
            sorted_idx = sorted_idx[:top_n_taxa]
        
        # Create entries for significant taxa
        for idx in sorted_idx:
            loading = loadings[idx]
            if abs(loading) < 0.01:  # Skip very small loadings
                continue
            
            result = {
                'Cluster': cluster,
                'PC': pc_name,
                'Taxon': taxa[idx],
                'Loading': loading,
                'Loading_Abs': abs_loadings[idx],
                'R_squared': row['R_squared'],
                'P_value': row['P_value']
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Sort by R_squared (descending), then by Loading_Abs (descending)
    df = df.sort_values(['R_squared', 'Loading_Abs'], ascending=[False, False])
    df = df.reset_index(drop=True)
    
    # Format numeric columns
    df['Loading'] = df['Loading'].round(4)
    df['Loading_Abs'] = df['Loading_Abs'].round(4)
    df['R_squared'] = df['R_squared'].round(4)
    
    return df


def calculate_pollution_pc_vs_species_pc_regressions(
    pca_results: Dict,
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    pollution_pc_prefix: str = 'Pollution_PC',
    species_pc_variance_threshold: float = 5.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate regressions between individual pollution PCs and species PCs for each cluster.
    
    For each cluster, this function performs linear regressions between each pollution PC
    (from contamination assessment) and each species PC (from community composition PCA),
    providing a detailed analysis of which pollution gradients affect which community
    composition axes.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas() in community composition analysis, containing:
        - 'pca_coordinates': {cluster_id: DataFrame of site PC coordinates}
        - 'variance_explained': {cluster_id: array of % variance per PC}
    raw_data : pd.DataFrame
        Raw data containing pollution PC columns (e.g., 'Pollution_PC1', 'Pollution_PC2', etc.)
        and cluster assignments
    cluster_column : str
        Column name for cluster assignments
    pollution_pc_prefix : str
        Prefix for pollution PC columns in raw_data (default: 'Pollution_PC')
    species_pc_variance_threshold : float
        Minimum variance explained (%) for a species PC to be included
        
    Returns:
    --------
    tuple
        (regression_table, detailed_results)
        - regression_table: DataFrame with columns [Cluster, Pollution_PC, Species_PC, 
          R_squared, P_value, Slope, Intercept, N_sites, Species_PC_Variance]
        - detailed_results: Dict with full regression details per cluster
    """
    pca_coordinates = pca_results.get('pca_coordinates', {})
    variance_explained = pca_results.get('variance_explained', {})
    
    # Find pollution PC columns in raw_data
    pollution_pc_cols = [col for col in raw_data.columns if col.startswith(pollution_pc_prefix)]
    
    if not pollution_pc_cols:
        raise ValueError(f"No pollution PC columns found with prefix '{pollution_pc_prefix}' in raw_data")
    
    results = []
    detailed_results = {}
    
    for cluster in sorted(pca_coordinates.keys()):
        coords = pca_coordinates[cluster]
        var_exp = variance_explained.get(cluster, np.array([]))
        
        if coords.empty:
            continue
        
        detailed_results[cluster] = {}
        
        # Get sites in this cluster
        cluster_mask = raw_data[cluster_column] == cluster
        cluster_sites = raw_data[cluster_mask].index
        common_sites = coords.index.intersection(cluster_sites)
        
        if len(common_sites) < 5:
            continue
        
        # Filter species PCs by variance threshold
        species_pc_cols = []
        for i, var_pct in enumerate(var_exp):
            if var_pct >= species_pc_variance_threshold:
                pc_name = f'PC{i+1}'
                if pc_name in coords.columns:
                    species_pc_cols.append((pc_name, var_pct))
        
        # Perform regressions for each pollution PC vs each species PC
        for pol_pc in pollution_pc_cols:
            pollution_values = raw_data.loc[common_sites, pol_pc].values
            
            # Skip if pollution PC has NaN
            valid_pollution = ~np.isnan(pollution_values)
            
            for species_pc, var_pct in species_pc_cols:
                species_values = coords.loc[common_sites, species_pc].values
                
                # Combine valid masks
                valid = valid_pollution & ~np.isnan(species_values)
                
                if valid.sum() < 5:
                    continue
                
                x = pollution_values[valid]
                y = species_values[valid]
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared = r_value ** 2
                
                result = {
                    'Cluster': int(cluster),
                    'Pollution_PC': pol_pc.replace(pollution_pc_prefix, 'PC'),
                    'Species_PC': species_pc,
                    'R_squared': r_squared,
                    'P_value': p_value,
                    'Slope': slope,
                    'Intercept': intercept,
                    'Std_Error': std_err,
                    'N_sites': valid.sum(),
                    'Species_PC_Variance': var_pct
                }
                results.append(result)
                
                # Store detailed results
                key = f"{pol_pc}_vs_{species_pc}"
                detailed_results[cluster][key] = {
                    'x': x,
                    'y': y,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value
                }
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Sort by P_value (ascending) then R_squared (descending)
        df = df.sort_values(['Cluster', 'P_value', 'R_squared'], 
                           ascending=[True, True, False]).reset_index(drop=True)
    
    return df, detailed_results


def plot_pollution_pc_vs_species_pc_regressions(
    regression_table: pd.DataFrame,
    detailed_results: Dict,
    colors: Optional[Dict] = None,
    figsize: Tuple[int, int] = (20, 16),
    p_value_threshold: float = 0.05
) -> plt.Figure:
    """
    Create a visualization of pollution PC vs species PC regressions.
    
    Parameters:
    -----------
    regression_table : pd.DataFrame
        Output from calculate_pollution_pc_vs_species_pc_regressions()
    detailed_results : dict
        Detailed results containing x, y values for each regression
    colors : dict, optional
        {cluster_id: color}
    figsize : tuple
        Figure size
    p_value_threshold : float
        Threshold for highlighting significant relationships
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with regression plots
    """
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    if regression_table.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No significant regressions found", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Get unique clusters and create subplots
    clusters = sorted(regression_table['Cluster'].unique())
    n_clusters = len(clusters)
    
    # Get unique pollution PCs and species PCs
    pollution_pcs = sorted(regression_table['Pollution_PC'].unique())
    species_pcs = sorted(regression_table['Species_PC'].unique())
    
    n_pol_pcs = len(pollution_pcs)
    n_species_pcs = len(species_pcs)
    
    # Create figure with grid: rows = pollution PCs, cols = species PCs, hue = clusters
    fig, axes = plt.subplots(n_pol_pcs, n_species_pcs, 
                              figsize=(n_species_pcs * 4, n_pol_pcs * 3.5),
                              squeeze=False)
    
    fig.suptitle('Pollution PC vs Species PC Regressions by Cluster\n(Significant relationships highlighted)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for i, pol_pc in enumerate(pollution_pcs):
        for j, species_pc in enumerate(species_pcs):
            ax = axes[i, j]
            
            for cluster in clusters:
                key = f"Pollution_{pol_pc}_vs_{species_pc}"
                
                if cluster not in detailed_results or key not in detailed_results[cluster]:
                    continue
                
                details = detailed_results[cluster][key]
                x, y = details['x'], details['y']
                r_sq = details['r_squared']
                p_val = details['p_value']
                slope = details['slope']
                intercept = details['intercept']
                
                color = colors.get(cluster, '#333333')
                
                # Determine if significant
                is_significant = p_val < p_value_threshold
                alpha = 0.8 if is_significant else 0.3
                marker_size = 40 if is_significant else 20
                
                # Plot scatter
                ax.scatter(x, y, c=[color], alpha=alpha, s=marker_size, 
                          label=f'Cluster {cluster}')
                
                # Plot regression line if significant
                if is_significant:
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=color, linestyle='--', 
                           linewidth=2, alpha=0.8)
                    
                    # Add annotation
                    ax.annotate(f'C{cluster}: R²={r_sq:.2f}*', 
                               xy=(0.02, 0.98 - cluster * 0.12),
                               xycoords='axes fraction',
                               fontsize=8, color=color, fontweight='bold',
                               verticalalignment='top')
            
            ax.set_xlabel(f'{pol_pc}' if i == n_pol_pcs - 1 else '')
            ax.set_ylabel(f'{species_pc}' if j == 0 else '')
            ax.set_title(f'{pol_pc} vs {species_pc}', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=colors.get(c, '#333333'), 
                          markersize=10, label=f'Cluster {c}')
               for c in clusters]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.tight_layout()
    
    return fig


def create_pollution_species_pc_summary_table(
    regression_table: pd.DataFrame,
    p_value_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Create a formatted summary table of pollution PC vs species PC regressions.
    
    Parameters:
    -----------
    regression_table : pd.DataFrame
        Output from calculate_pollution_pc_vs_species_pc_regressions()
    p_value_threshold : float
        Threshold for marking significant relationships
        
    Returns:
    --------
    pd.DataFrame
        Formatted summary table with significance markers
    """
    if regression_table.empty:
        return pd.DataFrame()
    
    df = regression_table.copy()
    
    # Add significance markers
    df['Significant'] = df['P_value'] < p_value_threshold
    df['Significance'] = df['P_value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    )
    
    # Format numeric columns
    df['R²'] = df['R_squared'].apply(lambda x: f"{x:.4f}")
    df['p-value'] = df['P_value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
    df['Slope'] = df['Slope'].apply(lambda x: f"{x:.4f}")
    df['Var%'] = df['Species_PC_Variance'].apply(lambda x: f"{x:.1f}%")
    
    # Select and reorder columns
    result = df[['Cluster', 'Pollution_PC', 'Species_PC', 'R²', 'p-value', 
                 'Significance', 'Slope', 'Var%', 'N_sites']].copy()
    
    return result