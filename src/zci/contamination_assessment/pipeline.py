"""
Complete contamination assessment pipeline.

This module provides a high-level pipeline function that orchestrates the entire
contamination assessment workflow, from data transformation through pollution scoring.
"""

from .transformations import log_z_score_transform, get_clustered_variable_order
from .pca_analysis import pca_with_PC_loadings
from .visualizations import create_ridge_plot
from .validation import rda_pollution_species_analysis
from .scoring import compute_pollution_scores, merge_pollution_scores_into_data
from sklearn.preprocessing import StandardScaler

def contamination_assessment_pipeline(
    data,
    transform_method='log_z_score',
    standardize_pca_scores=True,
    pc_weights=None,
    multiindex_levels=('pollution', 'weighted', 'SumRel'),
    visualize=True,
    run_rda_validation=False,
    rda_percentile_threshold=50,
    save_figures=False,
    save_individual_pcs=False,
    save_path=None
):
    """
    Complete pipeline for contamination assessment from raw data to pollution scores.
    
    This pipeline orchestrates the following workflow:
    1. Extract chemical/pollution data from multi-index dataframe
    2. Apply transformation and standardization (log + z-score by default)
    3. Perform PCA on transformed pollution variables
    4. Optionally visualize PC loadings with ridge plot
    5. Optionally validate with RDA analysis on reference sites
    6. Compute weighted pollution scores from PC scores
    7. Merge pollution scores back into original data structures
    
    Parameters:
    -----------
    data : pd.DataFrame
        Multi-index dataframe containing at least ('chemical', 'raw') block with pollution variables
    transform_method : str, default='log_z_score'
        Transformation method to apply. Currently supports:
        - 'log_z_score': log transformation followed by z-score standardization
        Future options could include 'robust_scaler', 'min_max', etc.
    standardize_pca_scores : bool, default=True
        Whether to standardize PC scores after PCA (mean=0, std=1)
    pc_weights : dict, list, array, or None, default=None
        Weights for computing pollution score from PCs. If None, uses default weights
        with PC3 weighted higher due to RDA biological relevance
    multiindex_levels : tuple, default=('pollution', 'weighted', 'SumRel')
        Three-level tuple for naming the pollution score column in multi-index data
    visualize : bool, default=True
        Whether to create visualizations (PCA variance, PC loadings ridge plot)
    run_rda_validation : bool, default=False
        Whether to run RDA validation analysis on reference sites
    rda_percentile_threshold : float, default=50
        Percentile threshold for defining reference sites in RDA (50 = median)
    save_figures : bool, default=True
        Whether to save generated figures to disk (currently always saves if visualize=True)
    save_individual_pcs : bool, default=False
        Whether to save individual pollution PC scores in addition to the weighted score.
        If True, adds columns like 'Pollution_PC1', 'Pollution_PC2', etc. to raw_data,
        and ('pollution', 'pc', 'PC1'), etc. to multiindex_data.
    save_path : str, optional
        Directory path to save figures. If None and save_figures=True, uses default path.
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'raw_data': Updated raw dataframe with Pollution_Score column
        - 'multiindex_data': Updated multi-index data with pollution score
        - 'transformed_pollution': Transformed pollution variables (log + z-score)
        - 'PC_loadings': PCA component loadings
        - 'PC_scores': PCA component scores
        - 'pollution_scores': Final weighted pollution scores
        - 'ordered_variables': Variables ordered by hierarchical clustering (if visualize=True)
        - 'rda_results': RDA validation results (if run_rda_validation=True)
        - 'figures': Dictionary of matplotlib figures (if visualize=True)
    
    Example Usage:
    --------------
    # Basic usage with default parameters
    >>> results = contamination_assessment_pipeline(data)
    >>> updated_data = results['multiindex_data']
    >>> pollution_scores = results['pollution_scores']
    
    # Custom weights focusing on specific PCs
    >>> custom_weights = {'PC1': 1.5, 'PC2': 1.0, 'PC3': 2.0, 'PC4': 0.5, 'PC5': 1.0, 'PC6': 0.5}
    >>> results = contamination_assessment_pipeline(
    ...     data,
    ...     pc_weights=custom_weights,
    ...     run_rda_validation=True,
    ...     rda_percentile_threshold=25  # Only bottom quartile as reference
    ... )
    
    # Quick run without visualizations
    >>> results = contamination_assessment_pipeline(
    ...     data,
    ...     visualize=False,
    ...     run_rda_validation=False
    ... )
    """
    
    print("="*70)
    print("CONTAMINATION ASSESSMENT PIPELINE")
    print("="*70)
    
    # =====================================================================
    # Step 1: Data Preparation
    # =====================================================================
    print("\n[Step 1/7] Extracting and preparing pollution data...")
    
    # Create raw data by removing multi-index levels
    raw_data = data.copy()
    raw_data.columns = raw_data.columns.droplevel([0, 1])
    
    # Extract pollution data block
    pollution_data = data[("chemical", "raw")].copy()
    print(f"  ✓ Extracted {pollution_data.shape[1]} pollution variables from {pollution_data.shape[0]} sites")
    
    # =====================================================================
    # Step 2: Transformation and Standardization
    # =====================================================================
    print(f"\n[Step 2/7] Applying transformation (method: {transform_method})...")
    
    if transform_method == 'log_z_score':
        transformed_pollution = log_z_score_transform(pollution_data)
        print("  ✓ Applied log transformation (except As, Bi)")
        print("  ✓ Applied z-score standardization")
        
    elif transform_method == "z_score":
        scaler = StandardScaler()
        transformed_pollution = pd.DataFrame(
            scaler.fit_transform(pollution_data),
            index=pollution_data.index,
            columns=pollution_data.columns
        )
        print("  ✓ Applied z-score standardization")
    else:
        raise ValueError(f"Unsupported transform_method: {transform_method}")
    
    # =====================================================================
    # Step 3: Variable Ordering (optional visualization prep)
    # =====================================================================
    print("\n[Step 3/7] Ordering variables by hierarchical clustering...")
    ordered_variables = None
    if visualize:
        ordered_corr_matrix = get_clustered_variable_order(transformed_pollution)
        ordered_variables = ordered_corr_matrix.columns.tolist()
        print(f"  ✓ Variables reordered by similarity")
    else:
        print("  ✓ Skipping variable ordering (visualize=False)")
    
    # =====================================================================
    # Step 4: PCA Analysis
    # =====================================================================
    print("\n[Step 4/7] Performing PCA on transformed pollution data...")
    
    # Initialize figures dictionary
    figures = {}
    
    PC_loadings, PC_scores, variance_fig = pca_with_PC_loadings(
        transformed_pollution,
        visualize=visualize,
        PC_scores_standardize=standardize_pca_scores,
        save_path=None  # Don't save inline, we'll save all at end
    )
    
    if variance_fig is not None:
        figures['pca_variance'] = variance_fig
    
    print(f"  ✓ Extracted {PC_loadings.shape[1]} principal components")
    print(f"  ✓ PC loadings shape: {PC_loadings.shape}")
    print(f"  ✓ PC scores shape: {PC_scores.shape}")
    if standardize_pca_scores:
        print("  ✓ PC scores standardized (mean=0, std=1)")
    
    # =====================================================================
    # Step 5: Visualization (optional)
    # =====================================================================
    if visualize:
        print("\n[Step 5/7] Creating visualizations...")
        ridge_fig, _ = create_ridge_plot(PC_loadings, save_path=None)  # Don't save inline
        figures['pc_loadings_ridge'] = ridge_fig
        print("  ✓ Ridge plot of PC loadings created")
    else:
        print("\n[Step 5/7] Skipping visualizations (visualize=False)")
    
    # =====================================================================
    # Step 6: RDA Validation (optional)
    # =====================================================================
    rda_results = None
    if run_rda_validation:
        print(f"\n[Step 6/7] Running RDA validation (percentile threshold: {rda_percentile_threshold})...")
        
        # Extract taxa data
        from zci.data_process.dataframe_ops import get_block
        taxa_df = get_block(data, "taxa")
        
        rda_results = rda_pollution_species_analysis(
            taxa_df, 
            PC_scores, 
            percentile_threshold=rda_percentile_threshold
        )
        
        # Collect RDA figure
        if rda_results and 'fig' in rda_results and rda_results['fig'] is not None:
            figures['rda_validation'] = rda_results['fig']
        
        print("  ✓ RDA validation completed")
    else:
        print("\n[Step 6/7] Skipping RDA validation (run_rda_validation=False)")
    
    # =====================================================================
    # Step 7: Compute and Merge Pollution Scores
    # =====================================================================
    print("\n[Step 7/7] Computing weighted pollution scores and merging into data...")
    
    pollution_scores = compute_pollution_scores(PC_scores, weights=pc_weights)
    
    if pc_weights is None:
        print("  ✓ Using default weights (PC3 weighted 2x for biological relevance)")
    else:
        print(f"  ✓ Using custom weights: {pc_weights}")
    
    raw_data, data = merge_pollution_scores_into_data(
        pollution_scores, 
        raw_data, 
        data,
        multiindex_levels=multiindex_levels,
        pc_scores=PC_scores,
        save_individual_pcs=save_individual_pcs
    )
    
    print(f"  ✓ Pollution scores added to data")
    print(f"  ✓ Multi-index column: {multiindex_levels}")
    print(f"  ✓ Raw data column: 'Pollution_Score'")
    if save_individual_pcs:
        pc_cols = [f'Pollution_{col}' for col in PC_scores.columns]
        print(f"  ✓ Individual pollution PCs saved: {pc_cols}")
    
    # =====================================================================
    # Step 8: Save Figures (Optional)
    # =====================================================================
    if save_path and figures:
        import os
        os.makedirs(save_path, exist_ok=True)
        print(f"\n[Step 8/8] Saving figures to {save_path}...")
        
        for i, (name, fig) in enumerate(figures.items(), start=1):
            if fig is not None:
                filepath = os.path.join(save_path, f"figure{i}_{name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved: {filepath}")
    
    # =====================================================================
    # Compile Results
    # =====================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nSummary:")
    print(f"  • Sites analyzed: {pollution_data.shape[0]}")
    print(f"  • Pollution variables: {pollution_data.shape[1]}")
    print(f"  • Principal components extracted: {PC_loadings.shape[1]}")
    print(f"  • Pollution score range: [{pollution_scores.min().values[0]:.2f}, {pollution_scores.max().values[0]:.2f}]")
    print(f"  • Pollution score mean: {pollution_scores.mean().values[0]:.2f}")
    print(f"  • Pollution score std: {pollution_scores.std().values[0]:.2f}")
    
    results = {
        'raw_data': raw_data,
        'multiindex_data': data,
        'transformed_pollution': transformed_pollution,
        'PC_loadings': PC_loadings,
        'PC_scores': PC_scores,
        'pollution_scores': pollution_scores,
        'ordered_variables': ordered_variables,
        'rda_results': rda_results,
        'figures': figures
    }
    
    return results
