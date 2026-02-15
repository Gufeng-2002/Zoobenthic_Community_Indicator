"""
Reference Sites Taxa Assemblage Pipeline

This module provides a comprehensive pipeline for analyzing taxa assemblages
at reference sites in the St. Clair-Detroit River System.

The pipeline performs the following steps:
1. Imputes missing velocity values
2. Selects reference sites based on pollution scores
3. Performs hierarchical clustering on taxa composition
4. Generates comprehensive visualizations

Author: Developed for St. Clair-Detroit River System Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Any
import warnings

from .velocity_imputation import impute_velocity_for_all_sites
from .reference_site_selection import select_reference_sites, compare_habitat_variables
from .hierarchical_clustering import cluster_species_hierarchical
from .cluster_visualization import visualize_cluster_analysis
from .boxcox_anova import (
    perform_boxcox_anova_analysis,
    create_anova_summary_table,
    create_anova_excel_table
)


def reference_sites_taxa_assemblage_pipeline(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    pollution_column: str = 'Pollution_Score',
    reference_percentile: float = 52,
    designate_ref_sites: Optional[List[int]] = None,
    species_transformation: str = 'hellinger',
    distance_measure: str = 'euclidean',
    n_clusters: int = 3,
    designate_cluster_labels: Optional[Dict[int, int]] = None,
    top_n_taxa: int = 15,
    env_variables: Optional[List[str]] = None,
    create_dendrogram_comparison: bool = False,
    create_comparison_plots: bool = False,
    create_fusion_plots: bool = False,
    create_ward_analysis: bool = False,
    create_cluster_visualization: bool = True,
    standardize_env: bool = False,
    label_positions: Optional[List[int]] = None,
    run_boxcox_anova: bool = True,
    use_transformation: str = 'log',
    anova_taxa_transformation: Optional[str] = None,
    anova_top_n_taxa: Optional[int] = 10,
    save_path: Optional[str] = None,
    save_tables: bool = False,
    table_save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline for reference sites taxa assemblage analysis.
    
    This pipeline performs velocity imputation, reference site selection,
    hierarchical clustering on taxa composition (using Ward's linkage), and 
    generates comprehensive visualizations..
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with single-level columns including coordinates, environmental
        variables, taxa, and pollution scores.
    multiindex_data : pd.DataFrame
        Multi-index DataFrame with hierarchical column structure
        (e.g., level 0: 'env', 'taxa', 'chemical').
    pollution_column : str, default='Pollution_Score'
        Name of the pollution score column in raw_data.
    reference_percentile : float, default=52
        Percentile threshold for selecting reference sites (bottom p% are selected).
        E.g., 52 means sites in the bottom 52% of pollution scores are reference.
    designate_ref_sites : list of int, optional
        List of site indices to forcibly designate as reference sites. If provided,
        these sites will be marked as reference regardless of pollution scores.
    species_transformation : str, default='hellinger'
        Transformation method for taxa data before clustering.
        Options: 'hellinger', 'chord', 'octave', 'none'.
        Note: Clustering always uses Ward's linkage method.
    distance_measure : str, default='euclidean'
        Distance measure to use for clustering.
    n_clusters : int, default=3
        Number of clusters to create.
        Options: 'euclidean', 'braycurtis', 'cityblock', etc.
    designate_cluster_labels: dict, optional
        Dictionary mapping site indices to cluster labels for forced designation.
        E.g., {S1: 0, S2: 1, S3: 2} assigns site S1 to cluster 0, etc.
    top_n_taxa : int, default=15
        Number of most abundant taxa to show in visualization.
    env_variables : list of str, optional
        List of environmental variable names to visualize.
        If None, will auto-select common variables.
    create_dendrogram_comparison : bool, default=False
        Whether to create dendrogram comparison figure (4 linkage methods).
    create_comparison_plots : bool, default=False
        Whether to create linkage comparison plots.
    create_fusion_plots : bool, default=False
        Whether to create fusion level plots.
    create_ward_analysis : bool, default=False
        Whether to create Ward's method analysis figure.
    create_cluster_visualization : bool, default=True
        Whether to create the final 3-panel cluster visualization.
    standardize_env : bool, default=False
        Whether to standardize environmental variables (mean=0, std=1)
        for cluster visualization.
    label_positions : list of int, optional
        Manually set x-positions for cluster labels in Ward analysis.
        E.g., [90, 250, 380] for 3 clusters.
    run_boxcox_anova : bool, default=False
        Whether to perform Box-Cox transformation + ANOVA analysis on
        environmental and taxa variables across clusters.
    use_transformation: str = 'log',
        Transformation to apply before ANOVA ('boxcox', 'log1p', or None).
        If None, performs ANOVA on raw (or taxa-transformed) data directly.
    anova_taxa_transformation : str, optional
        Transformation to apply to taxa before Box-Cox for ANOVA.
        If None, uses the same as species_transformation.
        Options: 'hellinger', 'chord', 'octave', 'none'.
    anova_top_n_taxa : int, optional
        Number of most abundant taxa to include in ANOVA.
        If None, analyzes all taxa.
    save_path : str, optional
        Directory path to save figures. If provided, figures will be saved
        to this directory with numbered filenames.
    save_tables : bool, default=False
        Whether to save ANOVA tables to Excel files.
    table_save_path : str, optional
        Directory path to save tables. If None, defaults to
        '../results/tables/02_taxa_assemblage_in_refs'.
    verbose : bool, default=True
        Whether to print progress messages.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'raw_data' : pd.DataFrame
            Updated raw_data with 'if_ref' and 'clusters' columns
        - 'multiindex_data' : pd.DataFrame
            Updated multiindex_data with cluster labels
        - 'cluster_labels' : pd.Series
            Cluster assignments for reference sites
        - 'reference_sites_count' : int
            Number of reference sites selected
        - 'cluster_distribution' : pd.Series
            Distribution of sites across clusters
        - 'velocity_imputation_fig' : plt.Figure (if created)
            Velocity imputation visualization
        - 'clustering_results' : dict
            Results from hierarchical clustering (includes figures)
        - 'cluster_visualization_fig' : plt.Figure (if created)
            Final 3-panel cluster visualization
        - 'habitat_comparison' : dict (optional)
            Results from habitat variable comparison
        - 'anova_results' : dict (if run_boxcox_anova=True)
            Results from Box-Cox + ANOVA analysis on env and taxa variables
        - 'figures' : dict
            Dictionary of all generated figures for easy saving
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> results = reference_sites_taxa_assemblage_pipeline(
    ...     raw_data=raw_data,
    ...     multiindex_data=data
    ... )
    
    >>> # Custom parameters
    >>> results = reference_sites_taxa_assemblage_pipeline(
    ...     raw_data=raw_data,
    ...     multiindex_data=data,
    ...     reference_percentile=50,
    ...     species_transformation='chord',
    ...     linkage_method='average',
    ...     n_clusters=4,
    ...     env_variables=['Measured Depth (m)', 'Temperature (oC)', 'LOI (%)'],
    ...     create_ward_analysis=True,
    ...     label_positions=[80, 200, 320, 450]
    ... )
    
    >>> # Access results
    >>> updated_raw_data = results['raw_data']
    >>> updated_multiindex = results['multiindex_data']
    >>> cluster_labels = results['cluster_labels']
    >>> viz_figure = results['cluster_visualization_fig']
    """
    
    if verbose:
        print("=" * 80)
        print("REFERENCE SITES TAXA ASSEMBLAGE PIPELINE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  - Reference percentile: {reference_percentile}%")
        print(f"  - Species transformation: {species_transformation}")
        print(f"  - Linkage method: Ward's")
        print(f"  - Number of clusters: {n_clusters}")
        print(f"  - Top taxa to display: {top_n_taxa}")
        print(f"  - Run Box-Cox ANOVA: {run_boxcox_anova}")
        if save_path:
            print(f"  - Save path: {save_path}")
        print()
    
    results = {}
    figures = {}  # Collect all figures for easy saving
    
    # ========================================================================
    # STEP 1: Impute Velocity Values
    # ========================================================================
    if verbose:
        print("STEP 1: Imputing velocity values...")
    
    velocity_results = impute_velocity_for_all_sites(raw_data, multiindex_data)
    raw_data = velocity_results['raw_data']
    multiindex_data = velocity_results['multiindex_data']
    results['velocity_imputation_fig'] = velocity_results['fig']
    if velocity_results['fig'] is not None:
        figures['velocity_imputation'] = velocity_results['fig']
    
    if verbose:
        print(f"  ✓ Velocity imputation complete")
        print()
    
    # ========================================================================
    # STEP 2: Select Reference Sites
    # ========================================================================
    if verbose:
        print(f"STEP 2: Selecting reference sites (bottom {reference_percentile}% by pollution)...")
    
    selection_results = select_reference_sites(
        raw_data, multiindex_data,
        percentile=reference_percentile,
        pollution_col=pollution_column
    )
    raw_data = selection_results['raw_data']
    multiindex_data = selection_results['multiindex_data']
    
    # Collect reference site selection figure
    if selection_results['fig'] is not None:
        results['reference_selection_fig'] = selection_results['fig']
        figures['reference_selection'] = selection_results['fig']
    
    reference_mask = raw_data['if_ref'] == True
    n_reference = reference_mask.sum()
    n_total = len(raw_data)
    
    results['reference_sites_count'] = n_reference
    
    if verbose:
        print(f"  ✓ Selected {n_reference} reference sites out of {n_total} total sites")
        print()
    
    # ========================================================================
    # STEP 3: Extract and Prepare Taxa Data for Clustering
    # ========================================================================
    if verbose:
        print("STEP 3: Extracting taxa data from reference sites...")
    
    # Extract taxa data from multiindex
    taxa_level_mask = multiindex_data.columns.get_level_values(0) == 'taxa'
    taxa_data = multiindex_data.loc[:, taxa_level_mask]
    taxa_ref = taxa_data[reference_mask].copy()
    
    # Simplify column names for clustering
    taxa_ref_simple = taxa_ref.copy()
    taxa_ref_simple.columns = taxa_ref_simple.columns.get_level_values(-1)
    
    # a violent designation way to choose sites as taxa_ref_simple (for testing purpose to match ref-sites with previous results)
    if designate_ref_sites is not None:
        taxa_ref_simple = taxa_data.loc[designate_ref_sites, :]
        taxa_ref_simple.columns = taxa_ref_simple.columns.get_level_values(-1)
        # Update raw_data and multiindex_data to reflect new reference sites
        raw_data['if_ref'] = False
        raw_data.loc[designate_ref_sites, 'if_ref'] = True
        # if ('Reference', 'if_ref') not in multiindex_data.columns:
        #     multiindex_data[('Reference', 'if_ref')] = False
        # multiindex_data.loc[designate_ref_sites, ('Reference', 'if_ref')] = True
        n_reference = len(designate_ref_sites)
        results['reference_sites_count'] = n_reference
        
    if verbose:
        print(f"  ✓ Extracted taxa data: {taxa_ref_simple.shape[0]} sites × {taxa_ref_simple.shape[1]} species")
        print()
    
    # ========================================================================
    # STEP 4: Hierarchical Clustering
    # ========================================================================
    if verbose:
        print(f"STEP 4: Performing hierarchical clustering (Ward's linkage, {species_transformation} transformation)...")
    
    cluster_labels, clustering_results = cluster_species_hierarchical(
        taxa_ref_simple,
        transformation=species_transformation,
        n_clusters=n_clusters,
        distance_measure = distance_measure,
        create_dendrogram_comparison=create_dendrogram_comparison,
        create_comparison_plots=create_comparison_plots,
        create_fusion_plots=create_fusion_plots,
        create_ward_analysis=create_ward_analysis,
        label_positions=label_positions
    )
    
    results['cluster_labels'] = cluster_labels
    
    # violently designate cluster labels for testing purpose to match previous results
    if designate_ref_sites is not None and designate_cluster_labels is not None:
        for site_idx, cluster_id in designate_cluster_labels.items():
            cluster_labels.loc[site_idx] = cluster_id
    
    results['clustering_results'] = clustering_results
    
    # Collect clustering figures
    if 'dendrogram_figure' in clustering_results and clustering_results['dendrogram_figure'] is not None:
        figures['dendrogram_comparison'] = clustering_results['dendrogram_figure']
    if 'comparison_figure' in clustering_results and clustering_results['comparison_figure'] is not None:
        figures['linkage_comparison'] = clustering_results['comparison_figure']
    if 'fusion_figure' in clustering_results and clustering_results['fusion_figure'] is not None:
        figures['fusion_comparison'] = clustering_results['fusion_figure']
    if 'ward_figure' in clustering_results and clustering_results['ward_figure'] is not None:
        figures['ward_analysis'] = clustering_results['ward_figure']
    
    # Get cluster distribution
    cluster_distribution = cluster_labels.value_counts().sort_index()
    results['cluster_distribution'] = cluster_distribution
    
    if verbose:
        print(f"  ✓ Clustering complete!")
        print(f"  ✓ Cluster distribution:")
        for cluster_id, count in cluster_distribution.items():
            print(f"      Cluster {int(cluster_id) + 1}: {count} sites")
        print()
    
    # ========================================================================
    # STEP 5: Add Cluster Labels to Data
    # ========================================================================
    if verbose:
        print("STEP 5: Adding cluster labels to datasets...")
    
    # Add to raw_data (NaN for non-reference sites)
    raw_data['clusters'] = np.nan
    raw_data.loc[taxa_ref_simple.index, 'clusters'] = cluster_labels
    
    # Add to multiindex_data
    if ('Clusters', 'Hierarchical', 'clusters') not in multiindex_data.columns:
        # Create new column in multiindex
        multiindex_data[('Clusters', 'Hierarchical', 'clusters')] = np.nan
    
    multiindex_data.loc[taxa_ref_simple.index, ('Clusters', 'Hierarchical', 'clusters')] = cluster_labels
    
    results['raw_data'] = raw_data
    results['multiindex_data'] = multiindex_data
    
    if verbose:
        print(f"  ✓ Cluster labels added to both datasets")
        print(f"      - Reference sites: labeled with cluster IDs (0 to {n_clusters-1})")
        print(f"      - Non-reference sites: labeled as NaN")
        print()
    
    # ========================================================================
    # STEP 6: Create Cluster Visualization
    # ========================================================================
    if create_cluster_visualization:
        if verbose:
            print("STEP 6: Creating cluster visualization...")
        
        # Use default environmental variables if not specified
        if env_variables is None:
            env_variables = [
                'Measured Depth (m)',
                'Velocity  at bottom (m/sec)',
                'Water DO Bottom (mg/L)',
                'Temperature (oC)',
                'MPS (Phi)',
                'LOI (%)'
            ]
        
        cluster_viz_fig = visualize_cluster_analysis(
            raw_data=raw_data,
            multiindex_data=multiindex_data,
            cluster_column='clusters',
            top_n_taxa=top_n_taxa,
            env_variables=env_variables,
            standardize_env = standardize_env
        )
        
        results['cluster_visualization_fig'] = cluster_viz_fig
        figures['cluster_visualization'] = cluster_viz_fig
        
        if verbose:
            print(f"  ✓ Cluster visualization created")
            print(f"      - Left panel: Geographic distribution")
            print(f"      - Upper right: {len(env_variables)} environmental variables")
            print(f"      - Lower right: Top {top_n_taxa} taxa")
            print()
    
    # ========================================================================
    # STEP 7: ANOVA Analysis (Optional)
    # ========================================================================
    env_anova_table = None
    taxa_anova_table = None
    
    if run_boxcox_anova:
        if verbose:
            transformation_status = f"with {use_transformation}" if use_transformation else "without transformation"
            print(f"STEP 7: Performing ANOVA analysis ({transformation_status})...")
            print()
        
        # Use default env variables if not specified
        if env_variables is None:
            env_variables = [
                'Measured Depth (m)',
                'Velocity  at bottom (m/sec)_Imputed',
                'Water DO Bottom (mg/L)',
                'Temperature (oC)',
                'MPS (Phi)',
                'LOI (%)'
            ]
        
        # Determine taxa transformation for ANOVA
        taxa_transform_for_anova = anova_taxa_transformation if anova_taxa_transformation is not None else species_transformation
        
        # Determine number of taxa for ANOVA
        n_taxa_for_anova = anova_top_n_taxa if anova_top_n_taxa is not None else None
        
        # Get taxa column names
        taxa_level_mask = multiindex_data.columns.get_level_values(0) == 'taxa'
        taxa_multiindex = multiindex_data.loc[:, taxa_level_mask]
        taxa_names = taxa_multiindex.columns.get_level_values(-1).tolist()
        
        # Limit to top N taxa if specified
        if n_taxa_for_anova is not None:
            # Get top N by abundance
            reference_mask = raw_data['if_ref'] == True
            taxa_sums = taxa_multiindex[reference_mask].sum().sort_values(ascending=False)
            taxa_names = taxa_sums.head(n_taxa_for_anova).index.get_level_values(-1).tolist()
        
        # Create environmental ANOVA table (publication-ready format for Excel)
        env_anova_table = create_anova_excel_table(
            raw_data=raw_data,
            multiindex_data=multiindex_data,
            variables=env_variables,
            cluster_column='clusters',
            variable_type='env',
            use_transformation=use_transformation,
            verbose=verbose
        )
        
        # Create taxa ANOVA table (publication-ready format for Excel)
        taxa_anova_table = create_anova_excel_table(
            raw_data=raw_data,
            multiindex_data=multiindex_data,
            variables=taxa_names,
            cluster_column='clusters',
            variable_type='taxa',
            use_transformation=use_transformation,
            taxa_transformation=taxa_transform_for_anova,
            top_n_taxa=n_taxa_for_anova,
            verbose=verbose
        )
        
        results['env_anova_table'] = env_anova_table
        results['taxa_anova_table'] = taxa_anova_table
        
        # Also run the detailed Box-Cox analysis for backward compatibility
        if use_transformation == 'boxcox':
            anova_results = perform_boxcox_anova_analysis(
                raw_data=raw_data,
                multiindex_data=multiindex_data,
                cluster_column='clusters',
                env_variables=env_variables,
                taxa_transformation=taxa_transform_for_anova,
                top_n_taxa=n_taxa_for_anova,
                verbose=verbose
            )
            results['anova_results'] = anova_results
        
        if verbose:
            transformation_status = f"with {use_transformation}" if use_transformation else "without transformation"
            print(f"  ✓ ANOVA analysis complete ({transformation_status})")
            print()
    
    # ========================================================================
    # STEP 8: Save Figures (Optional)
    # ========================================================================
    if save_path and figures:
        import os
        os.makedirs(save_path, exist_ok=True)
        if verbose:
            print("STEP 8: Saving figures...")
        
        for i, (name, fig) in enumerate(figures.items(), start=1):
            if fig is not None:
                filepath = os.path.join(save_path, f"figure{i}_{name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"  ✓ Saved: {filepath}")
        print()
    
    # ========================================================================
    # STEP 9: Save Tables (Optional)
    # ========================================================================
    tables = {}
    if env_anova_table is not None:
        tables['env_cluster_anova'] = env_anova_table
    if taxa_anova_table is not None:
        tables['taxa_cluster_anova'] = taxa_anova_table
    
    results['tables'] = tables
    
    if save_tables and tables:
        import os
        
        t_path = table_save_path if table_save_path else "../results/tables/02_taxa_assemblage_in_refs"
        os.makedirs(t_path, exist_ok=True)
        
        if verbose:
            print(f"STEP 9: Saving tables to {t_path}...")
        
        # Save each table as a separate Excel file
        for i, (table_name, table_df) in enumerate(tables.items(), start=1):
            filepath = os.path.join(t_path, f"table{i}_{table_name}.xlsx")
            table_df.to_excel(filepath, index=False)
            if verbose:
                print(f"  ✓ Saved: {filepath}")
    
    # Add figures dictionary to results
    results['figures'] = figures
    
    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    if verbose:
        print("=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Total sites: {n_total}")
        print(f"  - Reference sites: {n_reference} ({100*n_reference/n_total:.1f}%)")
        print(f"  - Clusters identified: {n_clusters}")
        print(f"  - Transformation used: {species_transformation}")
        print(f"  - Linkage method: Ward's")
        if run_boxcox_anova:
            transformation_status = f"with {use_transformation}" if use_transformation else "without transformation"
            print(f"  - ANOVA: Performed ({transformation_status})")
            # Count significant results from the tables
            if env_anova_table is not None:
                data_rows = env_anova_table.iloc[:-4]  # Exclude footer rows
                n_env_sig = data_rows['p'].apply(lambda x: '*' in str(x)).sum()
                print(f"    • Env variables significant: {n_env_sig}/{len(data_rows)}")
            if taxa_anova_table is not None:
                data_rows = taxa_anova_table.iloc[:-4]  # Exclude footer rows
                n_taxa_sig = data_rows['p'].apply(lambda x: '*' in str(x)).sum()
                print(f"    • Taxa significant: {n_taxa_sig}/{len(data_rows)}")
        if figures:
            print(f"  - Figures generated: {len(figures)}")
        print()
    
    return results


def compare_reference_habitat_variables(
    raw_data: pd.DataFrame,
    compare_with_all: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare habitat variables between reference and non-reference sites.
    
    This is a convenience wrapper around the compare_habitat_variables function
    for use after the pipeline has been run.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with 'if_ref' column indicating reference sites.
    compare_with_all : bool, default=True
        If True, compare reference sites with all sites.
        If False, compare reference sites with non-reference sites only.
    verbose : bool, default=True
        Whether to print results.
    
    Returns
    -------
    results : dict
        Results from habitat comparison including DataFrame and LaTeX table.
    """
    if verbose:
        print("Comparing habitat variables between reference and non-reference sites...")
        print()
    
    results = compare_habitat_variables(raw_data, compare_with_all=compare_with_all)
    
    if verbose:
        print("Results:")
        print("=" * 70)
        from IPython.display import display
        display(results['results_df'])
        print()
    
    return results
