"""
RDA Pipeline for Habitat-Taxa Relationship Analysis

This module provides a complete pipeline for Redundancy Analysis (RDA) to link
habitat variables to community composition, following the workflow from notebook 04.

The pipeline:
1. Takes reference sites data with pollution scores, if_ref indicator, and cluster labels
2. Applies optional transformations (log, z-score) to environmental variables
3. Applies optional transformations (hellinger, chord, octave) to taxa
4. Performs RDA analysis
5. Generates comprehensive summary tables and visualizations

Key Functions:
-------------
- perform_rda_analysis: Complete RDA pipeline with transformations and testing
- create_rda_summary_tables: Generate formatted summary tables
- plot_rda_triplot: Create publication-ready RDA triplot

Author: Developed for St. Clair-Detroit River System Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

from .rda import RDA
from ..taxa_assemblage_in_refs.hierarchical_clustering import (
    hellinger_transform,
    chord_transform,
    octave_transform
)


def perform_rda_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    env_variables: Optional[List[str]] = None,
    log_transform_env: bool = False,
    standardize_env: bool = True,
    taxa_transformation: str = 'hellinger',
    top_n_taxa: Optional[int] = None,
    n_permutations: int = 999,
    random_state: Optional[int] = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform complete RDA analysis on reference sites.
    
    This function analyzes the relationship between habitat variables and
    taxa composition using Redundancy Analysis (RDA). It includes:
    - Optional transformations of environmental and taxa data
    - RDA fitting and extraction of axes/eigenvalues
    - Global permutation test
    - Individual axis tests  
    - Environmental term tests
    - Comprehensive summary tables
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels.
        Must include reference sites (with non-NaN cluster labels).
    multiindex_data : pd.DataFrame
        Multi-index DataFrame containing taxa data.
    cluster_column : str, default='clusters'
        Name of column in raw_data containing cluster labels.
    env_variables : list of str, optional
        List of environmental variable names to use.
        If None, uses default set (Depth, Velocity, DO, Temperature, MPS, LOI).
    log_transform_env : bool, default=False
        Whether to apply log transformation to environmental variables.
    standardize_env : bool, default=True
        Whether to standardize (z-score) environmental variables.
    taxa_transformation : str, default='hellinger'
        Transformation to apply to taxa data.
        Options: 'hellinger', 'chord', 'octave', 'none'.
    top_n_taxa : int, optional
        Number of most abundant taxa to include.
        If None, uses all taxa.
    n_permutations : int, default=999
        Number of permutations for significance testing.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results.
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'rda_model': Fitted RDA object
        - 'env_data': Transformed environmental data
        - 'taxa_data': Transformed taxa data
        - 'global_test': Global RDA test results
        - 'axes_test': Individual axis test results
        - 'terms_test': Environmental term test results
        - 'summary_table_axes': Table 1 - RDA axes summary
        - 'summary_table_terms': Table 2 - Environmental terms summary
        - 'site_scores': RDA site scores
        - 'species_scores': RDA species scores
        - 'biplot_scores': Environmental biplot scores
        - 'transformation_info': Dict with transformation details
        
    Examples
    --------
    >>> results = perform_rda_analysis(
    ...     raw_data=ref_sites_data,
    ...     multiindex_data=multiindex_data,
    ...     cluster_column='clusters',
    ...     env_variables=['Depth', 'Temperature', 'DO', 'MPS', 'LOI', 'Velocity'],
    ...     log_transform_env=False,
    ...     standardize_env=True,
    ...     taxa_transformation='hellinger',
    ...     verbose=True
    ... )
    >>> print(results['summary_table_axes'])
    >>> print(results['summary_table_terms'])
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("REDUNDANCY ANALYSIS (RDA) PIPELINE")
        print("=" * 80)
        print("\nAnalyzing habitat-taxa relationships at reference sites")
        print("=" * 80)
    
    # ========================================================================
    # STEP 1: Filter reference sites
    # ========================================================================
    if verbose:
        print("\nStep 1: Filtering reference sites...")
    
    ref_mask = raw_data[cluster_column].notna()
    n_ref_sites = ref_mask.sum()
    
    if n_ref_sites == 0:
        raise ValueError("No reference sites found (all cluster labels are NaN)")
    
    ref_raw_data = raw_data[ref_mask].copy()
    ref_multiindex = multiindex_data.loc[ref_mask].copy()
    
    if verbose:
        print(f"  - Total sites: {len(raw_data)}")
        print(f"  - Reference sites: {n_ref_sites}")
        print(f"  - Clusters: {int(ref_raw_data[cluster_column].nunique())}")
    
    # ========================================================================
    # STEP 2: Extract and transform environmental data
    # ========================================================================
    if verbose:
        print("\nStep 2: Preparing environmental data...")
    
    # Use default environmental variables if not specified
    if env_variables is None:
        env_variables = [
            'Measured Depth (m)',
            'Velocity  at bottom (m/sec)_Imputed',
            'Water DO Bottom (mg/L)',
            'Temperature (oC)',
            'MPS (Phi)',
            'LOI (%)'
        ]
    
    # Extract environmental data
    env_data = ref_raw_data[env_variables].copy()
    
    # Check for missing values
    if env_data.isna().any().any():
        warnings.warn("Environmental data contains missing values. Dropping rows with NaN.")
        valid_idx = env_data.dropna().index
        env_data = env_data.loc[valid_idx]
        ref_raw_data = ref_raw_data.loc[valid_idx]
        ref_multiindex = ref_multiindex.loc[valid_idx]
    
    # Apply log transformation if requested
    if log_transform_env:
        if verbose:
            print("  - Applying log transformation to environmental variables")
        # Shift negative values before log
        for col in env_data.columns:
            min_val = env_data[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1e-6
                env_data[col] = np.log(env_data[col] + shift)
            else:
                env_data[col] = np.log(env_data[col])
    
    # Apply standardization if requested
    env_scaler = None
    if standardize_env:
        if verbose:
            print("  - Standardizing environmental variables (z-score)")
        env_scaler = StandardScaler()
        env_data_values = env_scaler.fit_transform(env_data)
        env_data = pd.DataFrame(
            env_data_values,
            columns=env_data.columns,
            index=env_data.index
        )
    
    if verbose:
        print(f"  - Environmental variables: {len(env_variables)}")
        print(f"  - Shape: {env_data.shape}")
    
    # ========================================================================
    # STEP 3: Extract and transform taxa data
    # ========================================================================
    if verbose:
        print("\nStep 3: Preparing taxa data...")
    
    # Extract taxa data
    taxa_level_mask = ref_multiindex.columns.get_level_values(0) == 'taxa'
    taxa_data = ref_multiindex.loc[:, taxa_level_mask].copy()
    
    # Simplify column names
    taxa_data.columns = taxa_data.columns.get_level_values(-1)
    
    # Filter to top N taxa if specified
    if top_n_taxa is not None:
        taxa_abundances = taxa_data.sum(axis=0).sort_values(ascending=False)
        top_taxa = taxa_abundances.head(top_n_taxa).index
        taxa_data = taxa_data[top_taxa]
        if verbose:
            print(f"  - Using top {top_n_taxa} most abundant taxa")
    
    # Apply transformation
    if taxa_transformation.lower() == 'hellinger':
        taxa_data = hellinger_transform(taxa_data)
        transformation_name = "Hellinger"
    elif taxa_transformation.lower() == 'chord':
        taxa_data = chord_transform(taxa_data)
        transformation_name = "Chord"
    elif taxa_transformation.lower() == 'octave':
        taxa_data = octave_transform(taxa_data)
        transformation_name = "Octave"
    elif taxa_transformation.lower() == 'none':
        transformation_name = "None (raw)"
    else:
        raise ValueError(f"Unknown transformation: {taxa_transformation}")
    
    if verbose:
        print(f"  - Taxa transformation: {transformation_name}")
        print(f"  - Number of taxa: {taxa_data.shape[1]}")
        print(f"  - Shape: {taxa_data.shape}")
    
    # ========================================================================
    # STEP 4: Fit RDA model
    # ========================================================================
    if verbose:
        print("\nStep 4: Fitting RDA model...")
    
    rda_model = RDA(
        center_X=True,
        center_Y=True,
        scale_X=False,  # Already standardized if requested
        ddof=1
    ).fit(env_data, taxa_data)
    
    if verbose:
        print(f"  ✓ RDA model fitted successfully")
        print(f"  - R²: {rda_model.fit_.r2:.4f}")
        print(f"  - Adjusted R²: {rda_model.fit_.r2_adj:.4f}")
        print(f"  - Constrained inertia: {rda_model.fit_.inertia_constrained:.4f}")
        print(f"  - Residual inertia: {rda_model.fit_.inertia_residual:.4f}")
    
    # ========================================================================
    # STEP 5: Perform permutation tests
    # ========================================================================
    if verbose:
        print(f"\nStep 5: Performing permutation tests ({n_permutations} permutations)...")
    
    # Global test
    global_test = rda_model.test_global(n_permutations=n_permutations, random_state=random_state)
    
    # Axes test
    axes_test = rda_model.test_axes(n_permutations=n_permutations, random_state=random_state)
    
    # Terms test
    terms_test = rda_model.test_terms(n_permutations=n_permutations, random_state=random_state)
    
    if verbose:
        print(f"  ✓ Permutation tests complete")
        print(f"  - Global test p-value: {global_test.p_value:.4f}")
        print(f"  - Significant axes (p < 0.05): {(axes_test['p'] < 0.05).sum()}")
        print(f"  - Significant terms (p < 0.05): {(terms_test['p'] < 0.05).sum()}")
    
    # ========================================================================
    # STEP 6: Create summary tables
    # ========================================================================
    if verbose:
        print("\nStep 6: Creating summary tables...")
    
    # Get RDA scores
    rda_scores = rda_model.scores(n_axes=min(6, len(axes_test)))
    
    # Get biplot scores (environmental variable coefficients)
    biplot_scores = rda_model.get_biplot_scores()
    
    # Table 1: RDA Axes Summary
    table_axes = create_rda_axes_summary_table(
        eigenvalues=rda_model.fit_.constrained_eigenvalues,
        explained_proportion=rda_model.fit_.explained_proportion,
        cumulative_explained=rda_model.fit_.cumulative_explained,
        axes_test=axes_test
    )
    
    # Table 2: Environmental Terms Summary  
    table_terms = create_rda_terms_summary_table(
        terms_test=terms_test,
        biplot_scores=biplot_scores
    )
    
    if verbose:
        print("  ✓ Summary tables created")
        print("\n" + "=" * 80)
        print("TABLE 1: RDA Axes Summary")
        print("=" * 80)
        print(table_axes.to_string(index=False))
        print("\n" + "=" * 80)
        print("TABLE 2: Environmental Terms Summary")
        print("=" * 80)
        print(table_terms.to_string(index=False))
        print("=" * 80)
    
    # ========================================================================
    # Return results
    # ========================================================================
    transformation_info = {
        'env_log_transform': log_transform_env,
        'env_standardize': standardize_env,
        'taxa_transformation': transformation_name,
        'env_scaler': env_scaler,
        'n_sites': len(env_data),
        'n_env_vars': len(env_variables),
        'n_taxa': taxa_data.shape[1]
    }
    
    return {
        'rda_model': rda_model,
        'env_data': env_data,
        'taxa_data': taxa_data,
        'ref_raw_data': ref_raw_data,
        'global_test': global_test,
        'axes_test': axes_test,
        'terms_test': terms_test,
        'summary_table_axes': table_axes,
        'summary_table_terms': table_terms,
        'site_scores': rda_scores.site_scores,
        'species_scores': rda_scores.species_scores,
        'biplot_scores': biplot_scores,
        'transformation_info': transformation_info
    }


def create_rda_axes_summary_table(
    eigenvalues: pd.Series,
    explained_proportion: pd.Series,
    cumulative_explained: pd.Series,
    axes_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Create Table 1: RDA Axes Summary.
    
    Includes eigenvalues, explained variance, F-statistics, and p-values
    for each RDA axis.
    """
    n_axes = len(axes_test)
    
    table = pd.DataFrame({
        'Axis': [f'RDA{i+1}' for i in range(n_axes)],
        'Eigenvalue': eigenvalues.iloc[:n_axes].values,
        'Explained (%)': (explained_proportion.iloc[:n_axes] * 100).values,
        'Cumulative (%)': (cumulative_explained.iloc[:n_axes] * 100).values,
        'F-statistic': axes_test['F'].values,
        'p-value': axes_test['p'].values,
        'Significance': axes_test['p'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ).values
    })
    
    return table


def create_rda_terms_summary_table(
    terms_test: pd.DataFrame,
    biplot_scores: pd.DataFrame
) -> pd.DataFrame:
    """
    Create Table 2: Environmental Terms Summary.
    
    Includes delta-inertia, F-statistics, p-values, and coefficients
    with RDA1 and RDA2 for each environmental variable.
    """
    # Merge test results with biplot scores
    table = terms_test.copy()
    
    # Add RDA1 and RDA2 coefficients
    table['RDA1_coef'] = table['term'].map(lambda t: biplot_scores.loc[t, 'RDA1'] if t in biplot_scores.index else np.nan)
    table['RDA2_coef'] = table['term'].map(lambda t: biplot_scores.loc[t, 'RDA2'] if t in biplot_scores.index else np.nan)
    
    # Add significance symbols
    table['Significance'] = table['p'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    
    # Rename columns for clarity
    table = table.rename(columns={
        'term': 'Environmental Variable',
        'delta_inertia': 'Delta Inertia',
        'F': 'F-statistic',
        'p': 'p-value',
        'RDA1_coef': 'RDA1 Coefficient',
        'RDA2_coef': 'RDA2 Coefficient'
    })
    
    return table[['Environmental Variable', 'Delta Inertia', 'F-statistic', 'p-value', 
                  'Significance', 'RDA1 Coefficient', 'RDA2 Coefficient']]


def plot_rda_triplot(
    rda_results: Dict[str, Any],
    axes: Tuple[int, int] = (1, 2),
    scaling: int = 2,
    show_sites: bool = True,
    show_species: bool = True,
    show_env: bool = True,
    site_groups: Optional[pd.Series] = None,
    arrow_scale: float = 2.0,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
    title: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create publication-ready RDA triplot.
    
    Shows sites, species, and environmental vectors in RDA ordination space.
    
    Parameters
    ----------
    rda_results : dict
        Results dictionary from perform_rda_analysis().
    axes : tuple of int, default (1, 2)
        Which RDA axes to plot (1-indexed).
    scaling : int, default 2
        Biplot scaling (1 = distance, 2 = correlation).
    show_sites : bool, default True
        Whether to show site scores.
    show_species : bool, default True
        Whether to show species scores.
    show_env : bool, default True
        Whether to show environmental vectors.
    site_groups : pd.Series, optional
        Grouping variable for coloring sites (e.g., cluster labels).
    arrow_scale : float, default 2.0
        Scaling factor for environmental arrows.
    figsize : tuple, default (10, 8)
        Figure size in inches.
    dpi : int, default 300
        Figure resolution.
    title : str, optional
        Custom title for the plot.
    **kwargs
        Additional arguments passed to plot_biplot().
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
        
    Examples
    --------
    >>> fig, ax = plot_rda_triplot(
    ...     rda_results=results,
    ...     site_groups=results['ref_raw_data']['clusters'],
    ...     scaling=2,
    ...     arrow_scale=2.5
    ... )
    >>> plt.show()
    """
    rda_model = rda_results['rda_model']
    
    # Use cluster labels as site groups if provided in results
    if site_groups is None and 'ref_raw_data' in rda_results:
        if 'clusters' in rda_results['ref_raw_data'].columns:
            site_groups = rda_results['ref_raw_data']['clusters']
    
    fig, ax = rda_model.plot_biplot(
        axes=axes,
        scaling=scaling,
        show_sites=show_sites,
        show_species=show_species,
        show_env=show_env,
        site_groups=site_groups,
        arrow_scale=arrow_scale,
        figsize=figsize,
        dpi=dpi,
        **kwargs
    )
    
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig, ax
