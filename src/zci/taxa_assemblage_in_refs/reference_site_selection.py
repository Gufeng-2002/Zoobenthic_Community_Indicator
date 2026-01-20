"""
Reference Site Selection Module

This module provides functions to select reference sites (least-polluted sites) from
the complete dataset based on pollution scores, with visualization of the selection
criteria and spatial distribution.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind


def select_reference_sites(
    raw_data,
    multiindex_data,
    percentile=50,
    pollution_col='SumRel',
    lon_col='Longitude',
    lat_col='Latitude',
    waterbody_col='Waterbody'
):
    """
    Select reference sites based on lowest pollution scores.
    
    This function identifies the least-polluted sites (bottom p-th percentile) as
    reference sites for further ecological analysis. It adds an 'if_ref' indicator
    column to both datasets and creates visualizations showing the spatial distribution
    and pollution score distribution of selected sites.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw dataframe without multi-index columns
    multiindex_data : pd.DataFrame
        Dataframe with multi-index columns
    percentile : float, default=50
        Percentile threshold for reference site selection (e.g., 50 means bottom 50%)
    pollution_col : str, default='SumRel'
        Name of the pollution score column
    lon_col : str, default='Longitude'
        Name of the longitude column
    lat_col : str, default='Latitude'
        Name of the latitude column
    waterbody_col : str, default='Waterbody'
        Name of the waterbody identifier column
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'raw_data': Updated raw dataframe with 'if_ref' column
        - 'multiindex_data': Updated multi-index data with if_ref indicator
        - 'reference_summary': Dictionary with selection statistics
        - 'fig': Combined visualization figure (map + ECDF)
    
    Example:
    --------
    >>> results = select_reference_sites(raw_data, multiindex_data, percentile=50)
    >>> updated_raw_data = results['raw_data']
    >>> fig = results['fig']
    >>> fig.show()
    """
    
    # Create copies to avoid modifying originals
    raw_data = raw_data.copy()
    multiindex_data = multiindex_data.copy()
    
    # Validate pollution column exists
    if pollution_col not in raw_data.columns:
        raise ValueError(f"Pollution column '{pollution_col}' not found in raw_data")
    
    # =====================================================================
    # Step 1: Select reference sites based on percentile threshold
    # =====================================================================
    print("="*70)
    print("REFERENCE SITE SELECTION")
    print("="*70)
    
    pollution_scores = raw_data[pollution_col]
    n_reference_sites = int(len(pollution_scores) * percentile / 100)
    reference_threshold = pollution_scores.nsmallest(n_reference_sites).max()
    
    # Create boolean mask for reference sites
    is_reference = pollution_scores <= reference_threshold
    
    # Add 'if_ref' column to raw_data
    raw_data['if_ref'] = is_reference.astype(int)
    
    # Add to multi-index data
    multiindex_col = pd.MultiIndex.from_tuples([('sites', 'classification', 'if_ref')])
    if_ref_df = pd.DataFrame(
        is_reference.astype(int).values,
        index=raw_data.index,
        columns=multiindex_col
    )
    
    from zci.data_process.dataframe_ops import concat_blocks
    multiindex_data = concat_blocks([multiindex_data, if_ref_df])
    
    # Get reference and non-reference data
    reference_sites = raw_data[is_reference]
    non_reference_sites = raw_data[~is_reference]
    
    print(f"\nSelection Criteria:")
    print(f"  Percentile threshold: {percentile}%")
    print(f"  Pollution score threshold: {reference_threshold:.3f}")
    
    print(f"\nSelected Sites:")
    print(f"  Reference sites: {n_reference_sites} ({percentile}% of {len(raw_data)} total)")
    print(f"  Non-reference sites: {len(non_reference_sites)}")
    
    print(f"\nPollution Score Ranges:")
    print(f"  Reference sites: {reference_sites[pollution_col].min():.3f} to {reference_sites[pollution_col].max():.3f}")
    print(f"  Non-reference sites: {non_reference_sites[pollution_col].min():.3f} to {non_reference_sites[pollution_col].max():.3f}")
    print(f"  All sites: {pollution_scores.min():.3f} to {pollution_scores.max():.3f}")
    
    # Waterbody distribution
    print(f"\nWaterbody Distribution (Reference Sites):")
    for wb, count in reference_sites[waterbody_col].value_counts().items():
        print(f"  {wb}: {count} sites")
    
    # =====================================================================
    # Step 2: Create combined visualization
    # =====================================================================
    print("\n" + "="*70)
    print("Creating visualizations...")
    
    fig = _create_combined_visualizations(
        raw_data, reference_sites, non_reference_sites,
        pollution_col, lon_col, lat_col, percentile, reference_threshold
    )
    
    # =====================================================================
    # Step 3: Compile summary statistics
    # =====================================================================
    summary = {
        'n_total_sites': len(raw_data),
        'n_reference_sites': n_reference_sites,
        'n_non_reference_sites': len(non_reference_sites),
        'percentile': percentile,
        'pollution_threshold': reference_threshold,
        'reference_pollution_range': (reference_sites[pollution_col].min(), 
                                       reference_sites[pollution_col].max()),
        'non_reference_pollution_range': (non_reference_sites[pollution_col].min(), 
                                           non_reference_sites[pollution_col].max()),
        'waterbody_counts': reference_sites[waterbody_col].value_counts().to_dict()
    }
    
    print("\n" + "="*70)
    print("REFERENCE SITE SELECTION COMPLETED")
    print("="*70)
    
    return {
        'raw_data': raw_data,
        'multiindex_data': multiindex_data,
        'reference_summary': summary,
        'fig': fig
    }


def _create_combined_visualizations(raw_data, reference_sites, non_reference_sites,
                                      pollution_col, lon_col, lat_col, percentile, 
                                      reference_threshold):
    """
    Create combined visualization with spatial map (left) and ECDF (right).
    
    Left panel: Spatial distribution of reference vs non-reference sites
    Right panel: Empirical cumulative distribution of pollution scores
    """
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 7), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
    ax_map = fig.add_subplot(gs[0])
    ax_ecdf = fig.add_subplot(gs[1])
    
    # =====================================================================
    # LEFT PANEL: Spatial Map
    # =====================================================================
    
    # Load geometries for map
    try:
        lake_stclair = gpd.read_file("../data/maps/lake_stclair/lake_stclair.shp").to_crs(epsg=4326)
        detroit_river = gpd.read_file("../data/maps/detroit_river_aoc_shapefile/AOC_MI_Detroit_2021.shp").to_crs(epsg=4326)
        stclair_river = gpd.read_file("../data/maps/aoc_mi_stclair_2021/AOC_MI_StClair_2021.shp").to_crs(epsg=4326)
        
        # Plot water bodies
        lake_stclair.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
        detroit_river.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
        stclair_river.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
        
        # Add annotations
        ax_map.text(-83.0, 42.2, 'Detroit River', fontsize=10, color='gray', style='italic')
        ax_map.text(-82.85, 42.9, 'St. Clair River', fontsize=10, color='gray', style='italic')
        ax_map.text(-82.55, 42.05, 'Lake Erie', fontsize=10, color='gray', style='italic')
        ax_map.text(-83.0, 42.5, 'Lake St. Clair', fontsize=10, color='gray', style='italic')
        
        ax_map.set_ylim(42, 43.1)
        ax_map.set_xlim(-83.3, -82.3)
    except FileNotFoundError:
        print("Warning: Map shapefiles not found, creating map without background")
    
    # Plot non-reference sites (gray, smaller)
    ax_map.scatter(
        non_reference_sites[lon_col], 
        non_reference_sites[lat_col],
        c='lightgray',
        s=60,
        alpha=0.6,
        edgecolors='gray',
        linewidths=0.5,
        label=f'Non-reference sites (n={len(non_reference_sites)})',
        zorder=2
    )
    
    # Plot reference sites (green, larger)
    ax_map.scatter(
        reference_sites[lon_col], 
        reference_sites[lat_col],
        c='green',
        s=120,
        alpha=0.9,
        edgecolors='darkgreen',
        linewidths=1.5,
        label=f'Reference sites (n={len(reference_sites)})',
        zorder=3
    )
    
    # Labels
    ax_map.set_title(f'Reference Sites Selection\nSpatial Distribution ({percentile}% Least-Polluted)', 
                     fontsize=13, fontweight='bold', pad=15)
    ax_map.set_xlabel('Longitude', fontsize=11)
    ax_map.set_ylabel('Latitude', fontsize=11)
    ax_map.grid(linestyle='--', alpha=0.4)
    ax_map.legend(loc='upper right', fontsize=10)
    
    # =====================================================================
    # RIGHT PANEL: ECDF with Reference Sites Highlighted
    # =====================================================================
    
    # Sort pollution scores for ECDF
    sorted_scores = raw_data[pollution_col].sort_values()
    n_sites = len(sorted_scores)
    cumulative_prob = np.arange(1, n_sites + 1) / n_sites
    
    # Plot ECDF as step plot
    ax_ecdf.step(sorted_scores, cumulative_prob, where='post', linewidth=2.5, 
                 color='darkblue', alpha=0.8, label='All sites ECDF')
    
    # Add reference threshold lines
    reference_percentile = percentile / 100
    ax_ecdf.axhline(y=reference_percentile, color='red', linestyle='--', linewidth=2, 
                    label=f'{percentile}% threshold', zorder=4)
    ax_ecdf.axvline(x=reference_threshold, color='red', linestyle='--', linewidth=2, 
                    alpha=0.7, zorder=4)
    
    # Highlight reference region
    ax_ecdf.fill_between(
        sorted_scores[sorted_scores <= reference_threshold], 
        0, reference_percentile, 
        alpha=0.3, color='green', 
        label=f'Reference sites (n={len(reference_sites)})',
        zorder=1
    )
    
    # Add percentile markers
    percentiles = [10, 25, 50, 75, 90]
    for pct in percentiles:
        score_at_pct = np.percentile(sorted_scores, pct)
        ax_ecdf.axhline(y=pct/100, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax_ecdf.text(sorted_scores.max() * 1.02, pct/100, f'{pct}%', 
                     fontsize=9, color='gray', va='center')
    
    # Add annotation box
    ax_ecdf.text(reference_threshold + (sorted_scores.max() - reference_threshold) * 0.15, 
                 reference_percentile + 0.05, 
                 f'Threshold: {reference_threshold:.3f}\n({percentile}% of sites)', 
                 fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='red', linewidth=1.5))
    
    # Labels
    ax_ecdf.set_title('Pollution Score Distribution\nEmpirical Cumulative Distribution Function', 
                      fontsize=13, fontweight='bold', pad=15)
    ax_ecdf.set_xlabel(f'Pollution Score ({pollution_col})', fontsize=11)
    ax_ecdf.set_ylabel('Cumulative Probability', fontsize=11)
    ax_ecdf.grid(True, alpha=0.3)
    ax_ecdf.legend(fontsize=10, loc='lower right')
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.set_xlim(sorted_scores.min() - 0.5, sorted_scores.max() + 1)
    
    plt.tight_layout()
    
    return fig


def compare_habitat_variables(
    raw_data,
    habitat_variables=None,
    if_ref_col='if_ref',
    compare_with_all = True
):
    """
    Perform t-tests comparing habitat variables between reference and all sites.
    
    This function compares the distributions of habitat/environmental variables between
    reference sites and all sites using independent t-tests. It returns both a formatted
    pandas DataFrame and a LaTeX table string.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw dataframe with 'if_ref' column indicating reference sites
    habitat_variables : list, optional
        List of habitat variable column names to compare
        If None, uses default set of environmental features
    if_ref_col : str, default='if_ref'
        Name of the reference site indicator column
    compare_with_all: whether to apply t-test of ref-sites aginst all sites or the non-ref sites
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'results_df': DataFrame with test statistics
        - 'latex_table': LaTeX-formatted table string
    
    Example:
    --------
    >>> results = compare_habitat_variables(raw_data)
    >>> print(results['results_df'])
    >>> print(results['latex_table'])
    """
    
    # Default habitat variables if not specified
    if habitat_variables is None:
        habitat_variables = [
            'Temperature (oC)', 
            'Measured Depth (m)', 
            'Water DO Bottom (mg/L)', 
            'LOI (%)', 
            'MPS (Phi)', 
            'Velocity  at bottom (m/sec)_Imputed'
        ]
    
    # Validate if_ref column exists
    if if_ref_col not in raw_data.columns:
        raise ValueError(f"Reference indicator column '{if_ref_col}' not found in raw_data")
    
    # Get reference and all sites data
    reference_data = raw_data[raw_data[if_ref_col] == 1]
    if compare_with_all:
        compared_data = raw_data
    else:
        compared_data = raw_data[raw_data[if_ref_col] != 1]
    
    print("="*70)
    print("HABITAT VARIABLE COMPARISON: REFERENCE VS ALL SITES")
    print("="*70)
    print(f"\nReference sites: {len(reference_data)}")
    print(f"Total sites: {len(compared_data)}")
    print(f"Variables tested: {len(habitat_variables)}")
    
    # Perform t-tests and collect results
    results = []
    
    for variable in habitat_variables:
        # Check if variable exists in data
        if variable not in raw_data.columns:
            print(f"Warning: Variable '{variable}' not found in data, skipping...")
            continue
        
        # Get data for reference and all sites (drop NaN)
        ref_vals = reference_data[variable].dropna()
        compared_vals = compared_data[variable].dropna()
        
        # Skip if insufficient data
        if len(ref_vals) < 2 or len(compared_vals) < 2:
            print(f"Warning: Insufficient data for '{variable}', skipping...")
            continue
        
        # Perform independent t-test (Welch's t-test, unequal variances)
        t_stat, p_val = ttest_ind(ref_vals, compared_vals, equal_var=False)
        
        # Calculate descriptive statistics
        ref_mean = ref_vals.mean()
        ref_std = ref_vals.std()
        all_mean = compared_vals.mean()
        all_std = compared_vals.std()
        
        # Determine significance level
        if p_val < 0.001:
            significance = "***"
            sig_level = "p < 0.001"
        elif p_val < 0.01:
            significance = "**"
            sig_level = "p < 0.01"
        elif p_val < 0.05:
            significance = "*"
            sig_level = "p < 0.05"
        else:
            significance = "NS"
            sig_level = "NS"
        
        # Add to results
        results.append({
            'Feature': variable,
            'Ref_Mean': ref_mean,
            'Ref_Std': ref_std,
            'All_Mean': all_mean,
            'All_Std': all_std,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Significance': significance,
            'Sig_Level': sig_level
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print("\n" + "="*100)
    print(f"{'Feature':<30} {'Ref Mean':<10} {'Ref Std':<10} {'All Mean':<10} {'All Std':<10} {'T-Stat':<10} {'P-Value':<12} {'Sig':<5}")
    print("-"*100)
    
    for _, row in results_df.iterrows():
        print(f"{row['Feature']:<30} {row['Ref_Mean']:<10.3f} {row['Ref_Std']:<10.3f} "
              f"{row['All_Mean']:<10.3f} {row['All_Std']:<10.3f} {row['T_Stat']:<10.3f} "
              f"{row['P_Value']:<12.4f} {row['Significance']:<5}")
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, NS p>=0.05")
    print(f"\nSummary: {len(results_df[results_df['Significance'] != 'NS'])} out of {len(results_df)} features show significant differences")
    
    # Generate LaTeX table
    latex_table = _generate_latex_table(results_df)
    
    return {
        'results_df': results_df,
        'latex_table': latex_table
    }


def _generate_latex_table(results_df):
    """
    Generate a formatted LaTeX table from t-test results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with t-test results
    
    Returns:
    --------
    str : LaTeX table string
    """
    
    latex = []
    
    # Table preamble
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{Comparison of Habitat Variables: Reference Sites vs All Sites}")
    latex.append(r"  \label{tab:habitat_comparison}")
    latex.append(r"  \begin{tabular}{lccccccl}")
    latex.append(r"    \toprule")
    
    # Header
    latex.append(r"    \textbf{Feature} & \textbf{Ref Mean} & \textbf{Ref SD} & "
                 r"\textbf{All Mean} & \textbf{All SD} & \textbf{t-stat} & "
                 r"\textbf{p-value} & \textbf{Sig.} \\")
    latex.append(r"    \midrule")
    
    # Data rows
    for _, row in results_df.iterrows():
        # Format feature name (escape underscores and special chars)
        feature = row['Feature'].replace('_', r'\_').replace('%', r'\%')
        
        # Format numbers
        ref_mean = f"{row['Ref_Mean']:.2f}"
        ref_std = f"{row['Ref_Std']:.2f}"
        all_mean = f"{row['All_Mean']:.2f}"
        all_std = f"{row['All_Std']:.2f}"
        t_stat = f"{row['T_Stat']:.2f}"
        
        # Format p-value
        if row['P_Value'] < 0.001:
            p_val = r"$<$0.001"
        else:
            p_val = f"{row['P_Value']:.3f}"
        
        # Significance
        sig = row['Significance']
        if sig == "***":
            sig = r"$^{***}$"
        elif sig == "**":
            sig = r"$^{**}$"
        elif sig == "*":
            sig = r"$^{*}$"
        
        # Add row
        latex.append(f"    {feature} & {ref_mean} & {ref_std} & {all_mean} & "
                     f"{all_std} & {t_stat} & {p_val} & {sig} \\\\")
    
    # Table footer
    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"  \vspace{0.2cm}")
    latex.append(r"  \begin{tablenotes}")
    latex.append(r"    \small")
    latex.append(r"    \item \textbf{Note:} Significance levels: $^{***}$ p $<$ 0.001, "
                 r"$^{**}$ p $<$ 0.01, $^{*}$ p $<$ 0.05, NS = not significant.")
    latex.append(r"    \item SD = Standard Deviation. t-test performed using Welch's method "
                 r"(unequal variances).")
    latex.append(r"  \end{tablenotes}")
    latex.append(r"\end{table}")
    
    return '\n'.join(latex)
