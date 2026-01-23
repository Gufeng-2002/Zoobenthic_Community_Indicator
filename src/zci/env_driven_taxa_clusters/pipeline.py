"""
Complete Environment-Driven Taxa Clusters Analysis Pipeline

This module provides a comprehensive pipeline that integrates RDA and LDA analyses
to understand how environmental conditions drive community composition and classify
sites into habitat-based clusters.

The pipeline takes data from the taxa assemblage stage (with pollution scores, if_ref,
and cluster labels for reference sites) and produces:
1. RDA ordination analysis with permutation tests
2. LDA classification with Monte Carlo cross-validation
3. LDA-RDA axis comparison
4. Predictions for non-reference sites
5. Updated datasets with complete cluster labels
6. All publication-ready figures and summary tables

Author: Developed for St. Clair-Detroit River System Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import warnings
from scipy import stats

from .rda_pipeline import (
    perform_rda_analysis,
    create_rda_axes_summary_table,
    create_rda_terms_summary_table,
    plot_rda_triplot
)
from .lda_pipeline import (
    perform_lda_analysis,
    perform_monte_carlo_cv,
    plot_lda_cv_results,
    compare_lda_rda_axes,
    predict_nonreference_sites,
    update_data_with_predictions,
    compute_lda_variable_importance
)


def compute_cluster_anova_table(
    raw_data: pd.DataFrame,
    variables: List[str],
    cluster_column: str = 'clusters',
    variable_type: str = 'env'
) -> pd.DataFrame:
    """
    Compute summary statistics and ANOVA F-stat for variables across clusters.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Data containing the variables and cluster labels
    variables : list of str
        Variable names to analyze
    cluster_column : str, default='clusters'
        Column name containing cluster labels
    variable_type : str, default='env'
        Type of variables: 'env' for environmental, 'taxa' for species
    
    Returns
    -------
    pd.DataFrame
        Summary table with columns: Variable, Cluster 0, Cluster 1, ..., F-stat
        Each cluster column contains "mean ± std" format
    """
    clusters = sorted(raw_data[cluster_column].dropna().unique())
    
    results = []
    for var in variables:
        if var not in raw_data.columns:
            continue
            
        row = {'Variable': var}
        
        # Compute mean ± std for each cluster
        groups = []
        for cluster in clusters:
            cluster_data = raw_data[raw_data[cluster_column] == cluster][var].dropna()
            mean_val = cluster_data.mean()
            std_val = cluster_data.std()
            row[f'Cluster {int(cluster)}'] = f"{mean_val:.2f} ± {std_val:.2f}"
            groups.append(cluster_data)
        
        # Compute ANOVA F-statistic
        if all(len(g) > 1 for g in groups) and len(groups) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                row['F-stat'] = f_stat
                row['p-value'] = p_val
            except:
                row['F-stat'] = np.nan
                row['p-value'] = np.nan
        else:
            row['F-stat'] = np.nan
            row['p-value'] = np.nan
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def perform_env_taxa_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    env_variables: List[str],
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    # RDA parameters
    rda_log_transform_env: bool = False,
    rda_standardize_env: bool = True,
    taxa_transformation: str = 'hellinger',
    top_n_taxa: Optional[int] = None,
    rda_n_permutations: int = 999,
    # LDA parameters
    lda_log_transform_env: bool = False,
    lda_standardize_env: bool = True,
    lda_n_iterations: int = 1000,
    lda_test_size: float = 0.2,
    # Visualization parameters
    rda_arrow_scale: float = 0.8,
    rda_species_scale: float = 0.5,
    rda_arrow_head_width: float = 0.03,
    rda_arrow_head_length: float = 0.03,
    rda_env_label_offset: float = 1.05,
    rda_env_fontsize: int = 11,
    rda_figsize: Tuple[float, float] = (14, 10),
    lda_cv_figsize: Tuple[float, float] = (16, 12),
    lda_arrow_scale = 3,
    # Control parameters
    random_state: Optional[int] = 42,
    save_path: Optional[str] = None,
    save_tables: bool = False,
    table_save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform complete environment-driven taxa cluster analysis.
    
    This comprehensive pipeline integrates RDA ordination and LDA classification
    to analyze how environmental conditions drive community composition and
    classify sites into habitat-based clusters.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with site information, taxa abundance, environmental variables,
        pollution scores, if_ref indicator, and cluster labels for reference sites
    multiindex_data : pd.DataFrame
        MultiIndex version with (Site, Variable) structure
    env_variables : list of str
        Environmental variable names to use in analyses
    cluster_column : str, default='clusters'
        Column name containing cluster labels for reference sites
    ref_column : str, default='if_ref'
        Column name indicating reference sites (True/False)
    
    RDA Parameters
    --------------
    rda_log_transform_env : bool, default=False
        Whether to log-transform environmental variables before RDA
    rda_standardize_env : bool, default=True
        Whether to z-score standardize environmental variables
    taxa_transformation : str, default='hellinger'
        Transformation for taxa data: 'hellinger', 'chord', 'octave', or 'none'
    top_n_taxa : int, optional
        Number of most abundant taxa to include (None = all taxa)
    rda_n_permutations : int, default=999
        Number of permutations for RDA significance tests
    
    LDA Parameters
    --------------
    lda_log_transform_env : bool, default=False
        Whether to log-transform environmental variables before LDA
    lda_standardize_env : bool, default=True
        Whether to standardize environmental variables for LDA
    lda_n_iterations : int, default=1000
        Number of Monte Carlo cross-validation iterations
    lda_test_size : float, default=0.2
        Proportion of data for testing in each CV iteration
    
    Visualization Parameters
    ------------------------
    rda_arrow_scale : float, default=0.8
        Scaling factor for environmental arrows in RDA triplot
    rda_species_scale : float, default=0.5
        Scaling factor for species positions in RDA triplot
    rda_arrow_head_width : float, default=0.03
        Width of arrow heads in RDA triplot
    rda_arrow_head_length : float, default=0.03
        Length of arrow heads in RDA triplot
    rda_env_label_offset : float, default=1.05
        Offset for environmental variable labels
    rda_env_fontsize : int, default=11
        Font size for environmental variable labels
    rda_figsize : tuple, default=(14, 10)
        Figure size for RDA triplot
    lda_cv_figsize : tuple, default=(16, 12)
        Figure size for LDA CV results
    lda_arrow_scale : float, default=3
        Scaling factor for environmental arrows in LDA triplot
    
    Control Parameters
    ------------------
    random_state : int, optional, default=42
        Random seed for reproducibility
    save_path : str, optional
        Directory path to save figures. If provided, figures will be saved
        to this directory with numbered filenames.
    verbose : bool, default=True
        Whether to print detailed progress information
    
    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        
        RDA Results:
        - 'rda_results': Complete RDA analysis results
        - 'rda_axes_table': Summary table of RDA axes
        - 'rda_terms_table': Summary table of environmental terms
        - 'rda_triplot': RDA triplot figure
        
        LDA Results:
        - 'lda_results': Basic LDA training results
        - 'lda_cv_results': Monte Carlo cross-validation results
        - 'lda_cv_figure': LDA CV visualization
        
        Comparison and Prediction:
        - 'lda_rda_comparison': LDA-RDA axis comparison
        - 'nonref_predictions': Predictions for non-reference sites
        
        Updated Data:
        - 'updated_raw_data': Raw data with complete cluster labels
        - 'updated_multiindex_data': MultiIndex data with complete cluster labels
        
        Summary:
        - 'n_sites_total': Total number of sites
        - 'n_sites_reference': Number of reference sites
        - 'n_sites_nonreference': Number of non-reference sites
        - 'n_clusters': Number of habitat clusters
        - 'cluster_distribution': Site counts per cluster
    
    Examples
    --------
    >>> # Run complete pipeline
    >>> results = perform_env_taxa_analysis(
    ...     raw_data=raw_data_m_by_p3,
    ...     multiindex_data=multiindex_data_m_by_p3,
    ...     env_variables=[
    ...         'Measured Depth (m)',
    ...         'Velocity  at bottom (m/sec)_Imputed',
    ...         'Water DO Bottom (mg/L)',
    ...         'Temperature (oC)',
    ...         'MPS (Phi)',
    ...         'LOI (%)'
    ...     ],
    ...     taxa_transformation='hellinger',
    ...     lda_n_iterations=1000,
    ...     verbose=True
    ... )
    >>> 
    >>> # Access results
    >>> print(f"RDA R²: {results['rda_results']['rda_model'].fit_.r2:.3f}")
    >>> print(f"LDA Accuracy: {results['lda_results']['accuracy']:.2%}")
    >>> print(f"LDA CV Mean Accuracy: {results['lda_cv_results']['mean_accuracy']:.2%}")
    >>> 
    >>> # Display figures
    >>> results['rda_triplot']
    >>> results['lda_cv_figure']
    """
    if verbose:
        print("\n" + "="*80)
        print("ENVIRONMENT-DRIVEN TAXA CLUSTERS ANALYSIS PIPELINE")
        print("="*80)
        print("\nThis pipeline performs:")
        print("  1. RDA ordination analysis")
        print("  2. LDA classification with Monte Carlo CV")
        print("  3. LDA-RDA axis comparison")
        print("  4. Non-reference site prediction")
        print("  5. Data integration with complete cluster labels")
        print("\n" + "="*80)
    
    # ========================================================================
    # STEP 1: RDA ORDINATION ANALYSIS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: REDUNDANCY ANALYSIS (RDA)")
        print("="*80)
    
    # Filter for reference sites
    ref_mask = raw_data[ref_column] == True
    ref_raw_data = raw_data[ref_mask].copy()
    ref_multiindex_data = multiindex_data.loc[ref_raw_data.index]
    
    rda_results = perform_rda_analysis(
        raw_data=ref_raw_data,
        multiindex_data=ref_multiindex_data,
        cluster_column=cluster_column,
        env_variables=env_variables,
        log_transform_env=rda_log_transform_env,
        standardize_env=rda_standardize_env,
        taxa_transformation=taxa_transformation,
        top_n_taxa=top_n_taxa,
        n_permutations=rda_n_permutations,
        random_state=random_state,
        verbose=verbose
    )
    
    # Create RDA summary tables
    rda_axes_table = rda_results['summary_table_axes']
    rda_terms_table = rda_results['summary_table_terms']
    
    # Create RDA triplot
    rda_triplot = plot_rda_triplot(
        rda_results=rda_results,
        site_groups=rda_results['ref_raw_data'][cluster_column],
        arrow_scale=rda_arrow_scale,
        species_scale=rda_species_scale,
        arrow_head_width=rda_arrow_head_width,
        arrow_head_length=rda_arrow_head_length,
        env_label_offset=rda_env_label_offset,
        env_fontsize=rda_env_fontsize,
        figsize=rda_figsize,
        show_sites=True,
        show_species=True,
        show_env=True
    )
    
    if verbose:
        print("\n✓ RDA analysis completed successfully!")
        print(f"  - R² = {rda_results['rda_model'].fit_.r2:.3f}")
        print(f"  - Adjusted R² = {rda_results['rda_model'].fit_.r2_adj:.3f}")
        print(f"  - Global test p-value = {rda_results['global_test'].p_value:.4f}")
    
    # ========================================================================
    # STEP 2: LDA CLASSIFICATION
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 2: LINEAR DISCRIMINANT ANALYSIS (LDA)")
        print("="*80)
    
    # Basic LDA training (already have ref_raw_data and ref_multiindex_data from Step 1)
    lda_results = perform_lda_analysis(
        raw_data=ref_raw_data,
        multiindex_data=ref_multiindex_data,
        cluster_column=cluster_column,
        env_variables=env_variables,
        log_env=lda_log_transform_env,
        standardize_env=lda_standardize_env,
        verbose=verbose
    )
    
    if verbose:
        print("\n✓ LDA training completed!")
        print(f"  - Overall accuracy: {lda_results['accuracy']:.2%}")
    
    # Compute LDA variable importance using Wilks' Lambda
    if verbose:
        print("\n" + "-"*80)
        print("Computing LDA Variable Importance (Wilks' Lambda)...")
        print("-"*80)
    
    lda_importance = compute_lda_variable_importance(
        lda_results=lda_results,
        verbose=verbose
    )
    
    # Monte Carlo Cross-Validation
    if verbose:
        print("\n" + "-"*80)
        print("Performing Monte Carlo Cross-Validation...")
        print("-"*80)
    
    lda_cv_results = perform_monte_carlo_cv(
        raw_data=ref_raw_data,
        multiindex_data=ref_multiindex_data,
        cluster_column=cluster_column,
        env_variables=env_variables,
        log_env=lda_log_transform_env,
        standardize_env=lda_standardize_env,
        n_iterations=lda_n_iterations,
        test_size=lda_test_size,
        random_state=random_state,
        verbose=verbose
    )
    
    # Create LDA CV visualization
    lda_cv_figure, _ = plot_lda_cv_results(
        cv_results=lda_cv_results,
        figsize=lda_cv_figsize
    )
    
    if verbose:
        print("\n✓ Monte Carlo CV completed!")
        print(f"  - Mean accuracy: {lda_cv_results['mean_accuracy']:.2%} ± {lda_cv_results['std_accuracy']:.2%}")
    
    # Create LDA triplot with significance-based line styles
    if verbose:
        print("\nCreating LDA triplot...")
    
    from .lda_pipeline import plot_lda_triplot
    lda_triplot_figure, _ = plot_lda_triplot(
        lda_results=lda_results,
        raw_data=raw_data,  # Use full dataset, not just reference sites
        multiindex_data=multiindex_data,  # Use full dataset
        cluster_column=cluster_column,
        env_variables=env_variables,
        log_env=lda_log_transform_env,
        arrow_scale=lda_arrow_scale,  # Use same scale as RDA for consistency
        figsize=rda_figsize,
        significance_dict=lda_importance['significance_dict'],  # Pass significance for line styles
        verbose=verbose
    )
    
    if verbose:
        print("✓ LDA triplot created!")
    
    # ========================================================================
    # STEP 3: LDA-RDA AXIS COMPARISON
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 3: COMPARING LDA AND RDA AXES")
        print("="*80)
    
    lda_rda_comparison = compare_lda_rda_axes(
        lda_results=lda_results,
        rda_results=rda_results,
        env_variables=env_variables,
        verbose=verbose
    )
    
    if verbose:
        print("\n✓ LDA-RDA comparison completed!")
    
    # ========================================================================
    # STEP 4: PREDICT NON-REFERENCE SITES
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 4: PREDICTING NON-REFERENCE SITE CLUSTERS")
        print("="*80)
    
    nonref_predictions = predict_nonreference_sites(
        raw_data=raw_data,
        multiindex_data=multiindex_data,
        lda_results=lda_results,
        env_variables=env_variables,
        ref_column=ref_column,
        verbose=verbose
    )
    
    if verbose:
        print("\n✓ Non-reference site predictions completed!")
        print(f"  - Classified {nonref_predictions['n_sites']} non-reference sites")
    
    # ========================================================================
    # STEP 5: UPDATE DATA WITH PREDICTIONS
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 5: UPDATING DATA WITH COMPLETE CLUSTER LABELS")
        print("="*80)
    
    updated_raw_data, updated_multiindex_data = update_data_with_predictions(
        raw_data=raw_data,
        multiindex_data=multiindex_data,
        predictions=nonref_predictions,
        cluster_column=cluster_column,
        ref_column=ref_column,
        verbose=verbose
    )
    
    # ========================================================================
    # STEP 6: CREATE CLUSTER COMPARISON VISUALIZATION
    # ========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STEP 6: CREATING CLUSTER COMPARISON FIGURE")
        print("="*80)
    
    from .lda_pipeline import plot_cluster_comparison
    cluster_comparison_figure, _ = plot_cluster_comparison(
        raw_data=updated_raw_data,
        multiindex_data=updated_multiindex_data,
        cluster_column=cluster_column,
        ref_column=ref_column,
        env_variables=env_variables,
        taxa_transformation=taxa_transformation,
        top_n_taxa=16,  # Show top 16 taxa in comparison figure
        figsize=(18, 12),
        verbose=verbose
    )
    
    if verbose:
        print("✓ Cluster comparison figure created!")
    
    # ========================================================================
    # STEP 7: COMPUTE CLUSTER ANOVA TABLES (this is no need now)
    # ========================================================================
    # if verbose:
    #     print("\n" + "="*80)
    #     print("STEP 7: COMPUTING CLUSTER ANOVA SUMMARIES")
    #     print("="*80)
    
    # # Get taxa columns from multiindex_data
    # taxa_level_mask = multiindex_data.columns.get_level_values(0) == 'taxa'
    # taxa_multiindex = multiindex_data.loc[:, taxa_level_mask]
    # taxa_names = taxa_multiindex.columns.get_level_values(-1).tolist()
    # taxa_cols = [c for c in taxa_names if c in ref_raw_data.columns]

    
    # if verbose:
    #     print(f"  ✓ Environmental ANOVA table: {len(env_anova_table)} variables")
    #     print(f"  ✓ Taxa ANOVA table: {len(taxa_anova_table)} taxa")
    
    # ========================================================================
    # STEP 8: SAVE TABLES (Optional)
    # ========================================================================
    
    # Import LDA table formatting functions
    from .lda_pipeline import (
        create_lda_confusion_matrix_table,
        create_lda_classification_report_table,
        create_mccv_confusion_matrix_table,
        create_mccv_classification_report_table,
        save_lda_tables_to_excel
    )
    
    # Create LDA tables
    lda_cm_table = create_lda_confusion_matrix_table(lda_results)
    lda_report_table = create_lda_classification_report_table(lda_results)
    mccv_cm_table = create_mccv_confusion_matrix_table(lda_cv_results)
    mccv_report_table = create_mccv_classification_report_table(lda_cv_results)
    
    tables = {
        'rda_axes_summary': rda_axes_table,
        'rda_terms_summary': rda_terms_table,
        'lda_axes_summary': lda_importance['axes_summary'],
        'lda_variable_importance': lda_importance['variable_importance'],
        # LDA confusion matrices and classification reports
        'lda_confusion_matrix': lda_cm_table,
        'lda_classification_report': lda_report_table,
        'mccv_confusion_matrix': mccv_cm_table,
        'mccv_classification_report': mccv_report_table,
    }
    
    if save_tables:
        from zci.output_saver import save_tables_dict
        t_path = table_save_path if table_save_path else "../results/tables/03_env_driven_taxa_clusters"
        
        if verbose:
            print("\n" + "="*80)
            print(f"STEP 8: SAVING TABLES TO {t_path}")
            print("="*80)
        
        save_tables_dict(
            tables=tables,
            save_dir=t_path,
            prefix="table",
            formats=['xlsx'],
            verbose=verbose
        )
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    n_sites_total = len(raw_data)
    n_sites_reference = ref_mask.sum()
    n_sites_nonreference = (~ref_mask).sum()
    n_clusters = len(updated_raw_data[cluster_column].dropna().unique())
    cluster_distribution = updated_raw_data[cluster_column].value_counts().sort_index().to_dict()
    
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nData Summary:")
        print(f"  - Total sites: {n_sites_total}")
        print(f"  - Reference sites: {n_sites_reference}")
        print(f"  - Non-reference sites: {n_sites_nonreference}")
        print(f"  - Habitat clusters: {n_clusters}")
        print(f"\nCluster Distribution (all sites):")
        for cluster in sorted(cluster_distribution.keys()):
            count = cluster_distribution[cluster]
            pct = count / n_sites_total * 100
            print(f"  - Cluster {int(cluster)}: {count} sites ({pct:.1f}%)")
        
        print(f"\nKey Results:")
        print(f"  - RDA R²: {rda_results['rda_model'].fit_.r2:.3f}")
        print(f"  - LDA accuracy (full model): {lda_results['accuracy']:.2%}")
        print(f"  - LDA CV accuracy: {lda_cv_results['mean_accuracy']:.2%} ± {lda_cv_results['std_accuracy']:.2%}")
        print(f"  - Best LDA-RDA match: {lda_rda_comparison['best_matches'].iloc[0]['LDA_Axis']} ↔ "
              f"{lda_rda_comparison['best_matches'].iloc[0]['Best_RDA_Match']} "
              f"(r = {lda_rda_comparison['best_matches'].iloc[0]['Correlation']:.3f})")
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    
    # Collect all figures in a dictionary
    figures = {
        'rda_triplot': rda_triplot,
        'lda_cv': lda_cv_figure,
        'lda_triplot': lda_triplot_figure,
        'cluster_comparison': cluster_comparison_figure,
    }
    
    # Save figures if path provided
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        if verbose:
            print("\n" + "="*80)
            print("SAVING FIGURES")
            print("="*80)
        
        for i, (name, fig) in enumerate(figures.items(), start=1):
            try:
                if fig is not None:
                    filepath = os.path.join(save_path, f"figure{i}_{name}.png")
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
            except AttributeError:
                fig[0].savefig(filepath, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"  ✓ Saved: {filepath}")
    
    results = {
        # RDA Results
        'rda_results': rda_results,
        'rda_axes_table': rda_axes_table,
        'rda_terms_table': rda_terms_table,
        'rda_triplot': rda_triplot,
        
        # LDA Results
        'lda_results': lda_results,
        'lda_importance': lda_importance,  # LDA variable importance tables
        'lda_axes_table': lda_importance['axes_summary'],  # LDA axes summary
        'lda_coefficients_table': lda_importance['coefficients_table'],  # LDA coefficients
        'lda_variable_importance_table': lda_importance['variable_importance'],  # Variable importance
        'lda_cv_results': lda_cv_results,
        'lda_cv_figure': lda_cv_figure,
        'lda_triplot': lda_triplot_figure,
        
        # Comparison and Prediction
        'lda_rda_comparison': lda_rda_comparison,
        'nonref_predictions': nonref_predictions,
        
        # Updated Data
        'updated_raw_data': updated_raw_data,
        'updated_multiindex_data': updated_multiindex_data,
        
        # Comprehensive Visualizations
        'cluster_comparison_figure': cluster_comparison_figure,
        
        # All figures in one place
        'figures': figures,
        
        # LDA Confusion Matrix and Classification Report Tables
        'lda_confusion_matrix_table': lda_cm_table,
        'lda_classification_report_table': lda_report_table,
        'mccv_confusion_matrix_table': mccv_cm_table,
        'mccv_classification_report_table': mccv_report_table,
        
        # All tables in one place
        'tables': tables,
        
        # Summary Statistics
        'n_sites_total': n_sites_total,
        'n_sites_reference': n_sites_reference,
        'n_sites_nonreference': n_sites_nonreference,
        'n_clusters': n_clusters,
        'cluster_distribution': cluster_distribution,
    }
    
    return results
