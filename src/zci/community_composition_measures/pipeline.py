"""
Community Composition Analysis Pipeline

This module provides the complete pipeline function that integrates all
community composition analysis steps. The pipeline is decomposed into
separate blocks for:
1. PCA analysis on species data
2. ZCI computation
3. Regression analysis
4. Visualizations
5. Data updating with results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

from .pca_analysis import fit_cluster_pcas, hellinger_transform, transform_sites_with_pca
from .taxa_loadings import plot_taxa_loadings_stacked, plot_taxa_loadings_comparison, plot_taxa_loadings_consistent
from .zci_calculation import (
    calculate_zci_all_clusters, 
    project_sites_to_pc_space,
    calculate_zci_for_projected_sites,
    combine_training_and_projected_zci,
    calculate_zci_all_clusters_with_threshold
)
from .visualization import (
    plot_ordination_comparison,
    plot_zci_vs_pollution,
    create_comprehensive_figure
)
from .pc_diagnostics import (
    plot_pc_loadings_by_cluster,
    calculate_pc_pollution_regressions,
    plot_pc_pollution_regressions,
    create_pc_regression_summary_table,
    extract_significant_pc_loadings,
    calculate_pollution_pc_vs_species_pc_regressions,
    plot_pollution_pc_vs_species_pc_regressions,
    create_pollution_species_pc_summary_table
)


# ============================================================================
# BLOCK 1: PCA ANALYSIS BLOCK
# ============================================================================

def run_pca_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    pollution_column: str = 'Pollution_Score',
    training_percentile: float = 100.0,
    variance_threshold: float = 0.70,
    taxa_transformation: str = 'hellinger',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    BLOCK 1: Fit PCA models on species data for each cluster.
    
    This block:
    1. Extracts taxa data from multiindex
    2. Selects training sites based on pollution percentile
    3. Fits separate PCA models per cluster
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with cluster labels and pollution scores
    multiindex_data : pd.DataFrame
        MultiIndex DataFrame with taxa data
    cluster_column : str, default='clusters'
        Column name for cluster labels
    ref_column : str, default='if_ref'
        Column name for reference site indicator
    pollution_column : str, default='Pollution_Score'
        Column name for pollution scores
    training_percentile : float, default=100.0
        Percentile threshold for training sites (0-100)
    variance_threshold : float, default=0.70
        Cumulative variance threshold for PC selection (0-1)
    taxa_transformation : str, default='hellinger'
        Transformation method for taxa data
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    dict with keys:
        - 'pca_results': Complete PCA results from fit_cluster_pcas()
        - 'training_mask': Boolean mask of training sites
        - 'non_training_mask': Boolean mask of non-training sites
    """
    if verbose:
        print("\n" + "="*80)
        print("BLOCK 1: PCA ANALYSIS")
        print("="*80)
        print(f"Taxa transformation: {taxa_transformation}")
        print(f"Training percentile: {training_percentile}%")
        print(f"Variance threshold: {variance_threshold * 100:.0f}%")
    
    # Extract taxa data from multiindex
    if verbose:
        print("\n--- Extracting taxa data ---")
    
    taxa_mask = multiindex_data.columns.get_level_values(0) == 'taxa'
    taxa_data = multiindex_data.loc[:, taxa_mask].copy()
    taxa_data.columns = taxa_data.columns.get_level_values(-1)
    
    if verbose:
        print(f"Taxa data shape: {taxa_data.shape}")
    
    # Create training mask based on pollution percentile
    if verbose:
        print("\n--- Selecting training sites ---")
    
    if training_percentile >= 100:
        training_mask = pd.Series(True, index=raw_data.index)
        if verbose:
            print("Using all sites for PCA training (percentile=100)")
    else:
        threshold = np.percentile(raw_data[pollution_column].dropna(), training_percentile)
        training_mask = raw_data[pollution_column] <= threshold
        if verbose:
            print(f"Pollution threshold (p{training_percentile}): {threshold:.3f}")
            print(f"Training sites: {training_mask.sum()} / {len(raw_data)}")
    
    # Fit PCA models
    if verbose:
        print("\n--- Fitting PCA models ---")
    
    cluster_labels = raw_data[cluster_column]
    
    pca_results = fit_cluster_pcas(
        taxa_data=taxa_data,
        cluster_labels=cluster_labels,
        site_mask=training_mask,
        variance_threshold=variance_threshold,
        taxa_transformation=taxa_transformation,
        random_state=random_state,
        verbose=verbose
    )
    
    return {
        'pca_results': pca_results,
        'training_mask': training_mask,
        'non_training_mask': ~training_mask,
        'taxa_data': taxa_data,
        'cluster_labels': cluster_labels
    }


# ============================================================================
# BLOCK 2: ZCI COMPUTATION BLOCK
# ============================================================================

def compute_zci_for_all_sites(
    pca_results: Dict[str, Any],
    raw_data: pd.DataFrame,
    pca_block_data: Dict[str, Any],
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    pollution_column: str = 'Pollution_Score',
    zci_method: str = 'default',
    least_pollution_threshold: Optional[float] = None,
    zci_pc_planes: Optional[Dict[int, Tuple[int, ...]]] = None,
    normalize_pcs: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    BLOCK 2: Calculate ZCI (compositional index) for all sites.
    
    This block:
    1. Calculates ZCI for training sites
    2. Projects non-training sites to PC space
    3. Calculates ZCI for projected sites
    4. Combines all ZCI values
    
    Parameters
    ----------
    pca_results : dict
        Results from Block 1 (PCA analysis)
    raw_data : pd.DataFrame
        Raw data with cluster and pollution information
    pca_block_data : dict
        Additional data from Block 1 (masks, labels)
    cluster_column : str, default='clusters'
        Column name for cluster labels
    ref_column : str, default='if_ref'
        Column name for reference site indicator
    pollution_column : str, default='Pollution_Score'
        Column name for pollution scores
    zci_method : str, default='default'
        Method for ZCI calculation: 'default' or 'least_pollution_threshold'
    least_pollution_threshold : float, optional
        Percentile threshold for least-polluted sites (0-100)
    zci_pc_planes : dict, optional
        {cluster_id: (pc1, pc2, ...)} for PC selection per cluster
    normalize_pcs : bool, default=True
        Whether to z-score normalize PC coordinates
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    dict with keys:
        - 'training_zci_df': ZCI for training sites
        - 'projected_zci_df': ZCI for projected sites
        - 'combined_zci_df': Combined ZCI for all sites
        - 'reference_points': {cluster_id: reference point array}
        - 'projected_coords': {cluster_id: DataFrame of projected coordinates}
    """
    if verbose:
        print("\n" + "="*80)
        print("BLOCK 2: ZCI COMPUTATION")
        print("="*80)
        print(f"ZCI method: {zci_method}")
        print(f"Normalize PCs: {normalize_pcs}")
        if zci_method == 'least_pollution_threshold' and least_pollution_threshold is not None:
            print(f"Least pollution threshold: {least_pollution_threshold}%")
    
    training_mask = pca_block_data['training_mask']
    non_training_mask = pca_block_data['non_training_mask']
    cluster_labels = pca_block_data['cluster_labels']
    taxa_data = pca_block_data['taxa_data']
    
    # Calculate ZCI for training sites
    if verbose:
        print("\n--- Calculating ZCI for training sites ---")
    
    if zci_method == 'least_pollution_threshold' and least_pollution_threshold is not None:
        training_zci_df = calculate_zci_all_clusters_with_threshold(
            pca_results=pca_results,
            raw_data=raw_data,
            cluster_column=cluster_column,
            pollution_column=pollution_column,
            least_pollution_threshold=least_pollution_threshold,
            pc_planes=zci_pc_planes,
            normalize_pcs=normalize_pcs
        )
    else:
        training_zci_df = calculate_zci_all_clusters(
            pca_results=pca_results,
            raw_data=raw_data,
            cluster_column=cluster_column,
            ref_column=ref_column,
            pollution_column=pollution_column,
            use_ref_centroid=True,
            normalize_pcs=normalize_pcs
        )
    
    if verbose:
        print(f"Training ZCI calculated for {len(training_zci_df)} sites")
    
    # Extract reference points
    reference_points = {}
    for cluster in pca_results['pca_coordinates'].keys():
        coords = pca_results['pca_coordinates'][cluster]
        if zci_method == 'least_pollution_threshold' and least_pollution_threshold is not None:
            cluster_mask = raw_data[cluster_column] == cluster
            site_pollution = raw_data.loc[coords.index.intersection(raw_data.index), pollution_column].dropna()
            if len(site_pollution) > 0:
                threshold_score = np.percentile(site_pollution, least_pollution_threshold)
                least_polluted = site_pollution[site_pollution <= threshold_score].index.tolist()
                if least_polluted:
                    if zci_pc_planes and cluster in zci_pc_planes:
                        pc_cols = [f'PC{i}' for i in zci_pc_planes[cluster]]
                    else:
                        pc_cols = [c for c in coords.columns if c.startswith('PC')]
                    available_pcs = [c for c in pc_cols if c in coords.columns]
                    reference_points[cluster] = coords.loc[least_polluted, available_pcs].mean().values
                else:
                    reference_points[cluster] = coords.mean().values
            else:
                reference_points[cluster] = coords.mean().values
        else:
            ref_mask = raw_data[ref_column].isin([True, 1])
            cluster_mask = raw_data[cluster_column] == cluster
            ref_sites = raw_data[cluster_mask & ref_mask].index.tolist()
            available_refs = [s for s in ref_sites if s in coords.index]
            if available_refs:
                reference_points[cluster] = coords.loc[available_refs].mean().values
            else:
                reference_points[cluster] = coords.mean().values
    
    # Project non-training sites
    if verbose:
        print("\n--- Projecting non-training sites ---")
    
    projected_coords = project_sites_to_pc_space(
        taxa_hellinger=pca_results['taxa_hellinger'],
        pca_models=pca_results['pca_models'],
        sites_to_project=non_training_mask,
        cluster_labels=cluster_labels,
        training_sites=pca_results['training_sites']
    )
    
    n_projected = sum(len(df) for df in projected_coords.values())
    if verbose:
        print(f"Projected {n_projected} sites onto PC space")
    
    # Calculate ZCI for projected sites
    if verbose:
        print("\n--- Calculating ZCI for projected sites ---")
    
    training_coords_stats = {}
    if normalize_pcs:
        for cluster, coords in pca_results['pca_coordinates'].items():
            training_coords_stats[cluster] = {
                'mean': coords.mean(),
                'std': coords.std()
            }
    
    projected_zci_df = calculate_zci_for_projected_sites(
        projected_coordinates=projected_coords,
        reference_points=reference_points,
        raw_data=raw_data,
        cluster_column=cluster_column,
        pollution_column=pollution_column,
        normalize_pcs=normalize_pcs,
        training_coords_stats=training_coords_stats
    )
    
    if verbose:
        print(f"Projected ZCI calculated for {len(projected_zci_df)} sites")
    
    # Combine training and projected ZCI
    combined_zci_df = combine_training_and_projected_zci(training_zci_df, projected_zci_df)
    
    if verbose:
        print(f"Total ZCI records: {len(combined_zci_df)}")
    
    return {
        'training_zci_df': training_zci_df,
        'projected_zci_df': projected_zci_df,
        'combined_zci_df': combined_zci_df,
        'reference_points': reference_points,
        'projected_coords': projected_coords
    }


# ============================================================================
# BLOCK 3: REGRESSION ANALYSIS BLOCK
# ============================================================================

def run_regression_analyses(
    pca_results: Dict[str, Any],
    projected_coords: Dict[str, pd.DataFrame],
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    pollution_column: str = 'Pollution_Score',
    pollution_pc_prefix: str = 'Pollution_PC',
    pc_variance_threshold: float = 5.0,
    top_n_taxa: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    BLOCK 3: Regression analysis between species PCs and pollution.
    
    This block:
    1. Calculates PC-Pollution regressions
    2. Calculates Pollution PC vs Species PC regressions
    3. Extracts significant loadings
    
    Parameters
    ----------
    pca_results : dict
        Results from Block 1 (PCA analysis)
    projected_coords : dict
        Projected coordinates from Block 2
    raw_data : pd.DataFrame
        Raw data with pollution information
    cluster_column : str, default='clusters'
        Column name for cluster labels
    pollution_column : str, default='Pollution_Score'
        Column name for pollution scores
    pollution_pc_prefix : str, default='Pollution_PC'
        Prefix for pollution PC column names
    pc_variance_threshold : float, default=5.0
        Minimum variance % for PC inclusion in regression
    top_n_taxa : int, default=10
        Number of top taxa to include in analysis
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    dict with keys:
        - 'pc_regression_table': PC-Pollution regression results
        - 'pc_regression_details': Detailed regression information
        - 'significant_pc_loadings': Loadings of significant PCs
        - 'pollution_vs_species_pc_table': Pollution PC vs Species PC results
        - 'pollution_vs_species_pc_details': Detailed cross-PC regression info
    """
    if verbose:
        print("\n" + "="*80)
        print("BLOCK 3: REGRESSION ANALYSIS")
        print("="*80)
        print(f"PC variance threshold: {pc_variance_threshold}%")
    
    pc_regression_table = pd.DataFrame()
    pc_regression_details = {}
    significant_pc_loadings = pd.DataFrame()
    pollution_vs_species_pc_table = pd.DataFrame()
    pollution_vs_species_pc_details = {}
    
    # PC-Pollution regression
    if verbose:
        print("\n--- Running PC-Pollution regression analysis ---")
    
    pc_regression_table, pc_regression_details = calculate_pc_pollution_regressions(
        pca_results=pca_results,
        projected_coords=projected_coords,
        raw_data=raw_data,
        pollution_column=pollution_column,
        cluster_column=cluster_column,
        variance_threshold=pc_variance_threshold
    )
    
    if not pc_regression_table.empty:
        if verbose:
            print(f"Found {len(pc_regression_table)} PCs with ≥{pc_variance_threshold}% variance")
        
        # Extract significant PC loadings (p < 0.05)
        significant_pc_loadings = extract_significant_pc_loadings(
            pca_results=pca_results,
            regression_table=pc_regression_table,
            p_value_threshold=0.05,
            top_n_taxa=top_n_taxa
        )
        
        if verbose and not significant_pc_loadings.empty:
            n_sig_pcs = significant_pc_loadings[['Cluster', 'PC']].drop_duplicates().shape[0]
            print(f"Found {n_sig_pcs} significant PCs (p < 0.05)")
    
    # Pollution PC vs Species PC regression
    if verbose:
        print("\n--- Running Pollution PC vs Species PC regression analysis ---")
    
    pollution_pc_cols = [c for c in raw_data.columns if c.startswith(pollution_pc_prefix)]
    
    if pollution_pc_cols:
        pollution_vs_species_pc_table, pollution_vs_species_pc_details = calculate_pollution_pc_vs_species_pc_regressions(
            pca_results=pca_results,
            raw_data=raw_data,
            cluster_column=cluster_column,
            pollution_pc_prefix=pollution_pc_prefix,
            species_pc_variance_threshold=pc_variance_threshold
        )
        
        if not pollution_vs_species_pc_table.empty and verbose:
            n_regressions = len(pollution_vs_species_pc_table)
            n_significant = (pollution_vs_species_pc_table['P_value'] < 0.05).sum()
            print(f"Tested {n_regressions} Pollution PC vs Species PC relationships")
            print(f"Found {n_significant} significant relationships (p < 0.05)")
    else:
        if verbose:
            print(f"No pollution PC columns found with prefix '{pollution_pc_prefix}'")
    
    return {
        'pc_regression_table': pc_regression_table,
        'pc_regression_details': pc_regression_details,
        'significant_pc_loadings': significant_pc_loadings,
        'pollution_vs_species_pc_table': pollution_vs_species_pc_table,
        'pollution_vs_species_pc_details': pollution_vs_species_pc_details
    }


# ============================================================================
# BLOCK 4: VISUALIZATION BLOCK
# ============================================================================

def create_visualizations(
    pca_results: Dict[str, Any],
    projected_coords: Dict[str, pd.DataFrame],
    combined_zci_df: pd.DataFrame,
    raw_data: pd.DataFrame,
    pca_block_data: Dict[str, Any],
    regression_results: Dict[str, Any],
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    pollution_column: str = 'Pollution_Score',
    top_n_taxa: int = 10,
    pc_plane: Tuple[int, int] = (1, 2),
    pc_loadings_top_n_taxa: Optional[int] = None,
    pc_loadings_show_values: bool = False,
    taxa_loadings_pcs: Optional[Dict[int, Tuple[int, ...]]] = None,
    use_consistent_taxa_order: bool = True,
    taxa_loadings_figsize: Tuple[int, int] = (14, 12),
    ordination_figsize: Tuple[int, int] = (18, 6),
    zci_figsize: Tuple[int, int] = (16, 10),
    comprehensive_figsize: Tuple[int, int] = (20, 16),
    colors: Optional[Dict] = None,
    create_taxa_loadings: bool = True,
    create_pc_loadings: bool = True,
    create_ordination: bool = True,
    create_zci: bool = True,
    create_pc_regression: bool = True,
    create_pollution_vs_species_pc: bool = True,
    create_comprehensive: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    BLOCK 4: Create all visualizations.
    
    This block creates publication-ready figures for all analyses:
    1. Taxa loadings plots
    2. PC loadings by cluster
    3. Ordination comparison
    4. ZCI vs pollution plots
    5. PC-Pollution regression plots
    6. Pollution PC vs Species PC plots
    7. Comprehensive multi-panel figure
    
    Parameters
    ----------
    pca_results : dict
        Results from Block 1 (PCA analysis)
    projected_coords : dict
        Projected coordinates from Block 2
    combined_zci_df : pd.DataFrame
        Combined ZCI values from Block 2
    raw_data : pd.DataFrame
        Raw data with site information
    pca_block_data : dict
        Additional data from Block 1
    regression_results : dict
        Results from Block 3 (regression analysis)
    [Other parameters control visualization options and appearance]
    
    Returns
    -------
    dict
        {figure_name: matplotlib Figure}
    """
    if verbose:
        print("\n" + "="*80)
        print("BLOCK 4: VISUALIZATIONS")
        print("="*80)
    
    # Default colors
    if colors is None:
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
                  3: '#d62728', 4: '#9467bd', 5: '#8c564b'}
    
    figures = {}
    
    if create_taxa_loadings:
        if verbose:
            print("\n--- Creating taxa loadings plot ---")
        
        if use_consistent_taxa_order:
            fig_loadings = plot_taxa_loadings_consistent(
                pca_results=pca_results,
                top_n=top_n_taxa,
                figsize=taxa_loadings_figsize,
                colors=colors,
                show_upper_error_only=True,
                taxa_loadings_pcs=taxa_loadings_pcs
            )
        else:
            fig_loadings = plot_taxa_loadings_stacked(
                pca_results=pca_results,
                top_n=top_n_taxa,
                figsize=taxa_loadings_figsize,
                colors=colors,
                taxa_loadings_pcs=taxa_loadings_pcs
            )
        figures['taxa_loadings'] = fig_loadings
    
    if create_pc_loadings:
        if verbose:
            print("\n--- Creating PC loadings by cluster plots ---")
        
        pc_loadings_figs = plot_pc_loadings_by_cluster(
            pca_results=pca_results,
            variance_threshold_pct=5.0,
            colors=colors,
            top_n_taxa=pc_loadings_top_n_taxa,
            show_values=pc_loadings_show_values
        )
        
        for cluster, fig in pc_loadings_figs.items():
            figures[f'pc_loadings_cluster_{cluster}'] = fig
        
        if verbose:
            print(f"Created {len(pc_loadings_figs)} PC loadings figures")
    
    if create_ordination:
        if verbose:
            print("\n--- Creating ordination comparison plot ---")
            print(f"PC plane: PC{pc_plane[0]} vs PC{pc_plane[1]}")
        
        fig_ordination = plot_ordination_comparison(
            training_coords=pca_results['pca_coordinates'],
            projected_coords=projected_coords,
            raw_data=raw_data,
            cluster_column=cluster_column,
            ref_column=ref_column,
            variance_explained=pca_results['variance_explained'],
            colors=colors,
            figsize=ordination_figsize,
            pc_plane=pc_plane
        )
        figures['ordination'] = fig_ordination
    
    if create_zci:
        if verbose:
            print("\n--- Creating ZCI vs pollution plot ---")
        
        fig_zci = plot_zci_vs_pollution(
            zci_df=combined_zci_df,
            colors=colors,
            figsize=zci_figsize
        )
        figures['zci_vs_pollution'] = fig_zci
    
    if create_pc_regression and not regression_results['pc_regression_table'].empty:
        if verbose:
            print("\n--- Creating PC-Pollution regression plots ---")
        
        fig_pc_regression = plot_pc_pollution_regressions(
            regression_table=regression_results['pc_regression_table'],
            detailed_results=regression_results['pc_regression_details'],
            colors=colors
        )
        figures['pc_regression'] = fig_pc_regression
    
    if create_pollution_vs_species_pc and not regression_results['pollution_vs_species_pc_table'].empty:
        if verbose:
            print("\n--- Creating Pollution PC vs Species PC regression plots ---")
        
        fig_pollution_vs_species = plot_pollution_pc_vs_species_pc_regressions(
            regression_table=regression_results['pollution_vs_species_pc_table'],
            detailed_results=regression_results['pollution_vs_species_pc_details'],
            colors=colors,
            p_value_threshold=0.05
        )
        figures['pollution_vs_species_pc'] = fig_pollution_vs_species
    
    if create_comprehensive:
        if verbose:
            print("\n--- Creating comprehensive figure ---")
        
        fig_comprehensive = create_comprehensive_figure(
            pca_results=pca_results,
            projected_coords=projected_coords,
            zci_df=combined_zci_df,
            raw_data=raw_data,
            cluster_column=cluster_column,
            ref_column=ref_column,
            colors=colors,
            figsize=comprehensive_figsize
        )
        figures['comprehensive'] = fig_comprehensive
    
    if verbose:
        print(f"\n✓ Created {len(figures)} figures")
    
    return figures


# ============================================================================
# BLOCK 5: DATA UPDATING BLOCK
# ============================================================================

def update_data_with_zci_and_species_pcs(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    combined_zci_df: pd.DataFrame,
    pca_results: Dict[str, Any],
    projected_coords: Dict[str, pd.DataFrame],
    cluster_column: str = 'clusters',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    BLOCK 5: Update raw_data and multiindex_data with ZCI and species PCs.
    
    This block:
    1. Adds ZCI values to raw_data
    2. Adds Species PC values to raw_data and multiindex_data
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data to update
    multiindex_data : pd.DataFrame
        MultiIndex data to update
    combined_zci_df : pd.DataFrame
        Combined ZCI DataFrame from Block 2
    pca_results : dict
        PCA results from Block 1
    projected_coords : dict
        Projected coordinates from Block 2
    cluster_column : str, default='clusters'
        Column name for cluster labels
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Updated raw_data and multiindex_data
    """
    if verbose:
        print("\n" + "="*80)
        print("BLOCK 5: UPDATING DATA WITH ZCI AND SPECIES PCs")
        print("="*80)
    
    updated_raw_data = raw_data.copy()
    updated_multiindex_data = multiindex_data.copy()
    
    # Add ZCI to raw_data
    if verbose:
        print("\n--- Adding ZCI values to raw_data ---")
    
    if 'Site' in combined_zci_df.columns:
        zci_map = combined_zci_df.set_index('Site')['ZCI'].to_dict()
    else:
        zci_map = combined_zci_df['ZCI'].to_dict() if 'ZCI' in combined_zci_df.columns else {}
    
    updated_raw_data['ZCI'] = updated_raw_data.index.map(zci_map)
    
    if verbose:
        valid_zci = updated_raw_data['ZCI'].notna().sum()
        print(f"✓ Added ZCI for {valid_zci} sites")
        print(f"  ZCI range: [{updated_raw_data['ZCI'].min():.3f}, {updated_raw_data['ZCI'].max():.3f}]")
    
    # Add Species PCs to raw_data
    if verbose:
        print("\n--- Adding Species PC values to raw_data ---")
    
    # Collect all PC coordinates (training + projected)
    all_pc_coords = {}
    for cluster, training_coords in pca_results['pca_coordinates'].items():
        all_pc_coords[cluster] = training_coords.copy()
        
        if cluster in projected_coords:
            projected = projected_coords[cluster]
            # Merge with existing coordinates
            all_pc_coords[cluster] = pd.concat([
                all_pc_coords[cluster],
                projected
            ], axis=0)
    
    # Add PC columns to raw_data
    pc_columns_added = []
    for cluster, coords in all_pc_coords.items():
        cluster_sites = updated_raw_data[updated_raw_data[cluster_column] == cluster].index
        for pc_col in coords.columns:
            if pc_col.startswith('PC'):
                col_name = f'Species_{pc_col}_Cluster{int(cluster)}'
                updated_raw_data[col_name] = np.nan
                updated_raw_data.loc[cluster_sites, col_name] = coords.loc[
                    cluster_sites.intersection(coords.index), pc_col
                ]
                pc_columns_added.append(col_name)
    
    if verbose:
        print(f"✓ Added {len(pc_columns_added)} Species PC columns to raw_data")
    
    # Add Species PCs to multiindex_data
    if verbose:
        print("\n--- Adding Species PC values to multiindex_data ---")
    
    pc_columns_added_multi = 0
    for cluster, coords in all_pc_coords.items():
        for pc_col in coords.columns:
            if pc_col.startswith('PC'):
                multi_col_name = ("species_pc", f"Cluster{int(cluster)}", f"PC{pc_col.replace('PC', '')}")
                
                # Create column if it doesn't exist
                if multi_col_name not in updated_multiindex_data.columns:
                    updated_multiindex_data[multi_col_name] = np.nan
                
                # Update values for each site
                for site_idx in coords.index:
                    if site_idx in updated_multiindex_data.index.get_level_values(0):
                        site_mask = updated_multiindex_data.index.get_level_values(0) == site_idx
                        updated_multiindex_data.loc[site_mask, multi_col_name] = coords.loc[
                            site_idx, pc_col
                        ]
                        pc_columns_added_multi += 1
    
    if verbose:
        print(f"✓ Added Species PC values to multiindex_data")
    
    # Add ZCI to multiindex_data
    if verbose:
        print("\n--- Adding ZCI to multiindex_data ---")
    
    zci_col_name = ("zci_results", "computed", "ZCI")
    if zci_col_name not in updated_multiindex_data.columns:
        updated_multiindex_data[zci_col_name] = np.nan
    
    for site_idx in updated_raw_data.index:
        if site_idx in updated_multiindex_data.index.get_level_values(0):
            site_mask = updated_multiindex_data.index.get_level_values(0) == site_idx
            updated_multiindex_data.loc[site_mask, zci_col_name] = updated_raw_data.loc[site_idx, 'ZCI']
    
    if verbose:
        print(f"✓ Added ZCI to multiindex_data")
    
    if verbose:
        print("\n✓ DATA UPDATE COMPLETED")
    
    return updated_raw_data, updated_multiindex_data


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def community_composition_pipeline(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    pollution_column: str = 'Pollution_Score',
    # PCA parameters
    training_percentile: float = 100.0,
    variance_threshold: float = 0.70,
    taxa_transformation: str = 'hellinger',
    # ZCI parameters
    zci_method: str = 'default',
    least_pollution_threshold: Optional[float] = None,
    zci_pc_planes: Optional[Dict[int, Tuple[int, ...]]] = None,
    normalize_pcs: bool = True,
    # Regression parameters
    pc_variance_threshold: float = 5.0,
    pollution_pc_prefix: str = 'Pollution_PC',
    # Visualization parameters
    create_taxa_loadings_plot: bool = True,
    create_pc_loadings_plot: bool = True,
    create_ordination_plot: bool = True,
    create_zci_plot: bool = True,
    create_pc_regression_plot: bool = True,
    create_pollution_vs_species_pc_plot: bool = True,
    create_comprehensive_plot: bool = True,
    # Visualization style parameters
    use_consistent_taxa_order: bool = True,
    pc_plane: Tuple[int, int] = (1, 2),
    top_n_taxa: int = 10,
    pc_loadings_top_n_taxa: Optional[int] = None,
    pc_loadings_show_values: bool = False,
    taxa_loadings_pcs: Optional[Dict[int, Tuple[int, ...]]] = None,
    taxa_loadings_figsize: Tuple[int, int] = (14, 12),
    ordination_figsize: Tuple[int, int] = (18, 6),
    zci_figsize: Tuple[int, int] = (16, 10),
    comprehensive_figsize: Tuple[int, int] = (20, 16),
    colors: Optional[Dict] = None,
    # Output control
    save_path: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    MAIN PIPELINE: Complete community composition analysis.
    
    This pipeline integrates five decomposed blocks for comprehensive analysis:
    1. Block 1 - PCA Analysis: Fit PCA models on species data per cluster
    2. Block 2 - ZCI Computation: Calculate compositional indices for all sites
    3. Block 3 - Regression Analysis: Analyze PC-pollution relationships
    4. Block 4 - Visualizations: Create publication-ready figures
    5. Block 5 - Data Updating: Update raw_data and multiindex_data with results
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with cluster labels, reference indicator, pollution scores
    multiindex_data : pd.DataFrame
        MultiIndex DataFrame with taxa data
    cluster_column : str, default='clusters'
        Column name for cluster labels
    ref_column : str, default='if_ref'
        Column name for reference site indicator
    pollution_column : str, default='Pollution_Score'
        Column name for pollution scores
    
    PCA Parameters
    ---------------
    training_percentile : float, default=100.0
        Percentile threshold for selecting training sites (0-100)
    variance_threshold : float, default=0.70
        Cumulative variance threshold for PC selection (0-1)
    taxa_transformation : str, default='hellinger'
        Transformation method: 'hellinger', 'chord', 'octave', 'bray-curtis', 'none'
    
    ZCI Parameters
    ---------------
    zci_method : str, default='default'
        'default' or 'least_pollution_threshold'
    least_pollution_threshold : float, optional
        Percentile threshold for least-polluted sites (0-100)
    zci_pc_planes : dict, optional
        {cluster_id: (pc1, pc2, ...)} for PC selection per cluster
    normalize_pcs : bool, default=True
        Whether to z-score normalize PC coordinates
    
    Regression Parameters
    ----------------------
    pc_variance_threshold : float, default=5.0
        Minimum variance % for PC inclusion
    pollution_pc_prefix : str, default='Pollution_PC'
        Prefix for pollution PC column names
    
    Visualization Parameters
    -------------------------
    create_taxa_loadings_plot : bool, default=True
    create_pc_loadings_plot : bool, default=True
    create_ordination_plot : bool, default=True
    create_zci_plot : bool, default=True
    create_pc_regression_plot : bool, default=True
    create_pollution_vs_species_pc_plot : bool, default=True
    create_comprehensive_plot : bool, default=True
    use_consistent_taxa_order : bool, default=True
    top_n_taxa : int, default=10
    pc_plane : tuple, default=(1, 2)
    pc_loadings_top_n_taxa : int, optional
    pc_loadings_show_values : bool, default=False
    taxa_loadings_pcs : dict, optional
    taxa_loadings_figsize : tuple, default=(14, 12)
    ordination_figsize : tuple, default=(18, 6)
    zci_figsize : tuple, default=(16, 10)
    comprehensive_figsize : tuple, default=(20, 16)
    colors : dict, optional
        {cluster_id: color} for visualization
    
    Output Parameters
    ------------------
    save_path : str, optional
        Path to save figures
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Print progress messages
    
    Returns
    -------
    dict
        Complete analysis results containing:
        - 'pca_results': PCA models and coordinates
        - 'pca_block_data': Training/non-training masks and labels
        - 'combined_zci_df': ZCI values for all sites
        - 'projected_coords': Projected site coordinates
        - 'reference_points': Reference centroids by cluster
        - 'regression_results': PC-pollution regression results
        - 'figures': {name: matplotlib Figure}
        - 'updated_raw_data': raw_data with ZCI and Species PCs
        - 'updated_multiindex_data': multiindex_data with ZCI and Species PCs
        - 'summary': Summary statistics dictionary
        - 'pc_regression_table': Regression results table
        - 'significant_pc_loadings': Significant PC loadings
    """
    # Print header
    if verbose:
        print("=" * 80)
        print("COMMUNITY COMPOSITION ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Taxa transformation: {taxa_transformation}")
        print(f"  Training percentile: {training_percentile}%")
        print(f"  Variance threshold: {variance_threshold * 100:.0f}%")
        print(f"  ZCI method: {zci_method}")
    
    # =========================================================================
    # BLOCK 1: PCA ANALYSIS
    # =========================================================================
    pca_block_data = run_pca_analysis(
        raw_data=raw_data,
        multiindex_data=multiindex_data,
        cluster_column=cluster_column,
        ref_column=ref_column,
        pollution_column=pollution_column,
        training_percentile=training_percentile,
        variance_threshold=variance_threshold,
        taxa_transformation=taxa_transformation,
        random_state=random_state,
        verbose=verbose
    )
    
    pca_results = pca_block_data['pca_results']
    
    # =========================================================================
    # BLOCK 2: ZCI COMPUTATION
    # =========================================================================
    zci_block_data = compute_zci_for_all_sites(
        pca_results=pca_results,
        raw_data=raw_data,
        pca_block_data=pca_block_data,
        cluster_column=cluster_column,
        ref_column=ref_column,
        pollution_column=pollution_column,
        zci_method=zci_method,
        least_pollution_threshold=least_pollution_threshold,
        zci_pc_planes=zci_pc_planes,
        normalize_pcs=normalize_pcs,
        verbose=verbose
    )
    
    combined_zci_df = zci_block_data['combined_zci_df']
    projected_coords = zci_block_data['projected_coords']
    reference_points = zci_block_data['reference_points']
    
    # =========================================================================
    # BLOCK 3: REGRESSION ANALYSIS
    # =========================================================================
    regression_results = run_regression_analyses(
        pca_results=pca_results,
        projected_coords=projected_coords,
        raw_data=raw_data,
        cluster_column=cluster_column,
        pollution_column=pollution_column,
        pollution_pc_prefix=pollution_pc_prefix,
        pc_variance_threshold=pc_variance_threshold,
        top_n_taxa=top_n_taxa,
        verbose=verbose
    )
    
    # =========================================================================
    # BLOCK 4: VISUALIZATIONS
    # =========================================================================
    figures = create_visualizations(
        pca_results=pca_results,
        projected_coords=projected_coords,
        combined_zci_df=combined_zci_df,
        raw_data=raw_data,
        pca_block_data=pca_block_data,
        regression_results=regression_results,
        cluster_column=cluster_column,
        ref_column=ref_column,
        pollution_column=pollution_column,
        top_n_taxa=top_n_taxa,
        pc_plane=pc_plane,
        pc_loadings_top_n_taxa=pc_loadings_top_n_taxa,
        pc_loadings_show_values=pc_loadings_show_values,
        taxa_loadings_pcs=taxa_loadings_pcs,
        use_consistent_taxa_order=use_consistent_taxa_order,
        taxa_loadings_figsize=taxa_loadings_figsize,
        ordination_figsize=ordination_figsize,
        zci_figsize=zci_figsize,
        comprehensive_figsize=comprehensive_figsize,
        colors=colors,
        create_taxa_loadings=create_taxa_loadings_plot,
        create_pc_loadings=create_pc_loadings_plot,
        create_ordination=create_ordination_plot,
        create_zci=create_zci_plot,
        create_pc_regression=create_pc_regression_plot,
        create_pollution_vs_species_pc=create_pollution_vs_species_pc_plot,
        create_comprehensive=create_comprehensive_plot,
        verbose=verbose
    )
    
    # =========================================================================
    # BLOCK 5: DATA UPDATING
    # =========================================================================
    updated_raw_data, updated_multiindex_data = update_data_with_zci_and_species_pcs(
        raw_data=raw_data,
        multiindex_data=multiindex_data,
        combined_zci_df=combined_zci_df,
        pca_results=pca_results,
        projected_coords=projected_coords,
        cluster_column=cluster_column,
        verbose=verbose
    )
    
    # =========================================================================
    # GENERATE SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("GENERATING SUMMARY")
        print("="*80)
    
    summary = {
        'n_clusters': len(pca_results['pca_models']),
        'n_training_sites': len(pca_block_data['training_mask'][pca_block_data['training_mask']]),
        'n_non_training_sites': len(pca_block_data['non_training_mask'][pca_block_data['non_training_mask']]),
        'n_total_sites': len(combined_zci_df),
        'n_figures': len([f for f in figures.values() if f is not None]),
        'training_percentile': training_percentile,
        'variance_threshold': variance_threshold,
        'taxa_transformation': taxa_transformation,
        'clusters': {}
    }
    
    # Cluster-level statistics
    for cluster in pca_results['pca_models'].keys():
        cluster_data = combined_zci_df[combined_zci_df['Cluster'] == cluster]
        
        # ZCI-Pollution correlation
        x = cluster_data['Pollution_Score'].values
        y = cluster_data['ZCI'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        
        if valid.sum() > 2:
            from scipy.stats import pearsonr
            r, p = pearsonr(x[valid], y[valid])
        else:
            r, p = np.nan, np.nan
        
        summary['clusters'][cluster] = {
            'n_training': len(pca_results['training_sites'].get(cluster, [])),
            'n_projected': len(projected_coords.get(cluster, pd.DataFrame())),
            'n_pcs': pca_results['n_components'].get(cluster, 0),
            'total_variance': pca_results['total_variance'].get(cluster, 0),
            'zci_pollution_correlation': r,
            'zci_pollution_pvalue': p,
            'mean_zci': cluster_data['ZCI'].mean(),
            'std_zci': cluster_data['ZCI'].std()
        }
    
    # =========================================================================
    # SAVE FIGURES (Optional)
    # =========================================================================
    if save_path and figures:
        import os
        os.makedirs(save_path, exist_ok=True)
        if verbose:
            print("\n--- Saving figures ---")
        
        for i, (name, fig) in enumerate(figures.items(), start=1):
            if fig is not None:
                filepath = os.path.join(save_path, f"figure{i}_{name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"  ✓ Saved: {filepath}")
    
    # =========================================================================
    # Print final summary
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nAnalysis Results:")
        print(f"  Total sites: {summary['n_total_sites']}")
        print(f"    - Training: {summary['n_training_sites']}")
        print(f"    - Non-training: {summary['n_non_training_sites']}")
        print(f"  Clusters: {summary['n_clusters']}")
        print(f"  Figures generated: {summary['n_figures']}")
        print(f"\nData Updated:")
        print(f"  ✓ updated_raw_data with ZCI and Species PCs")
        print(f"  ✓ updated_multiindex_data with ZCI and Species PCs")
        print("\n" + "="*80)
    
    return {
        'pca_results': pca_results,
        'pca_block_data': pca_block_data,
        'combined_zci_df': combined_zci_df,
        'projected_coords': projected_coords,
        'reference_points': reference_points,
        'regression_results': regression_results,
        'figures': figures,
        'updated_raw_data': updated_raw_data,
        'updated_multiindex_data': updated_multiindex_data,
        'summary': summary,
        'pc_regression_table': regression_results['pc_regression_table'],
        'pc_regression_details': regression_results['pc_regression_details'],
        'significant_pc_loadings': regression_results['significant_pc_loadings'],
        'pollution_vs_species_pc_table': regression_results['pollution_vs_species_pc_table'],
        'pollution_vs_species_pc_details': regression_results['pollution_vs_species_pc_details']
    }
