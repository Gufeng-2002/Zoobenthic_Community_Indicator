# DEPRECATED MODULE - For Backward Compatibility Only
# =====================================================
# This module is deprecated as of v2.0.0 (January 2026)
#
# The contamination assessment functionality has been refactored into
# focused modules for better maintainability:
#   - transformations.py: Data transformation
#   - pca_analysis.py: PCA computation
#   - visualizations.py: Plotting functions
#   - validation.py: RDA validation
#   - scoring.py: Pollution score computation
#   - pipeline.py: Complete workflow orchestration
#
# MIGRATION:
#   Instead of: from zci.contamination_assessment.pca_pollution_scores import func
#   Use:        from zci.contamination_assessment import func
#
# This file imports and re-exports functions for backward compatibility.
# All functionality remains available but is now better organized.
# =====================================================

import warnings

# Issue deprecation warning on import
warnings.warn(
    "The 'pca_pollution_scores' module is deprecated. "
    "Please use 'from zci.contamination_assessment import ...' instead. "
    "See contamination_assessment/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import all functions from new modular structure for backward compatibility
from .transformations import log_z_score_transform, get_clustered_variable_order
from .pca_analysis import pca_with_PC_loadings
from .visualizations import create_ridge_plot
from .validation import rda_pollution_species_analysis
from .scoring import compute_pollution_scores, merge_pollution_scores_into_data
from .pipeline import contamination_assessment_pipeline

# Re-export everything for backward compatibility
__all__ = [
    'log_z_score_transform',
    'get_clustered_variable_order',
    'pca_with_PC_loadings',
    'create_ridge_plot',
    'rda_pollution_species_analysis',
    'compute_pollution_scores',
    'merge_pollution_scores_into_data',
    'contamination_assessment_pipeline',
]

# Note: FIGURE_SAVE_PATH is now defined in individual modules
# but we re-export it here for compatibility
from .pca_analysis import FIGURE_SAVE_PATH

def log_z_score_transform(df, log_columns = None):
    """Apply log-transformation followed by Z-score standardization on specified columns in the df.

    Args:
        df (_type_): the dataframe to be transformed
        log_columns (_type_, optional): the columns to be log-transformed. 
        Defaults to None, which means all columns.
    """
    

    if log_columns is None:
        log_columns = df.columns
    df_transformed = df.copy()
    
    # Log transformation
    for col in log_columns:
        if col not in ["As", "Bi"]:
            df_transformed[col] = np.log1p(df_transformed[col])
    
    # Z-score standardization on all columns
    columns = df_transformed.columns
    scaler = StandardScaler()
    df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
    
    return df_transformed


# Apply hierarchical clustering to sort pollution variables by similarity
def get_clustered_variable_order(df):
    """
    Sort pollution variables by similarity using hierarchical clustering.
    Variables with high correlation will be placed closer together in the output list.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pollution variables
    
    Returns:
    --------
    ordered_variables : list
        List of variable names sorted by similarity (highly correlated variables are adjacent)
    """
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Convert correlation to distance (1 - |correlation|)
    # Similar variables have low distance
    distance_matrix = 1 - corr_matrix.abs()
    
    # Perform hierarchical clustering using average linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    
    # Extract the ordering from clustering dendrogram
    cluster_order = leaves_list(linkage_matrix)
    ordered_variables = corr_matrix.columns[cluster_order].tolist()
    
    # Reorder the correlation matrix
    clustered_corr = corr_matrix.loc[ordered_variables, ordered_variables]
    
    # Style the reordered correlation matrix
    clustered_styled_corr = clustered_corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1)\
        .format(precision=2)\
        .set_caption("Clustered Correlation Matrix")
        
    return clustered_styled_corr


# Apply PCA on the transformed pollution variables
def pca_with_PC_loadings(df, visualize = True, PC_scores_standardize = True):
    """Apply PCA on the transformed pollution variables and return 
    a dataframe with PCs that account for 80% total variation

    Args:
        df (_type_): the raw pollution dataframe
    """
    
    pca_model = PCA()
    pca_model.fit(df)
    

    # convert the selected PCs back to dataframe, remain the first 6 PCs
    pca_components = pd.DataFrame(np.transpose(pca_model.components_[:6]),
                              index=df.columns,
                              columns=[f"PC{i+1}" for i in range(6)])
    # convert the PCA scores to dataframe
    pca_scores = pd.DataFrame(pca_model.transform(df)[:, :6],
                              index=df.index,
                              columns=[f"PC{i+1}" for i in range(6)])
    # standardize the PC scores as default
    pca_scores = (pca_scores - pca_scores.mean()) / pca_scores.std()
    
    if visualize == True:
        # visualize the explained variance
        plt.figure(figsize=(12, 5))

        # subplot 1: explained variance ratio
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            pca_model.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Component')

        # subplot 2: cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            pca_model.explained_variance_ratio_.cumsum(), 'bo-')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, "PCA_explained_variance.png"), dpi=300)
        plt.show()
        
    return pca_components, pca_scores

# visualize the PCA loadings using ridge plot with clustered variable order
def create_ridge_plot(pca_components, figsize=(16, 10)):
    """
    Create a ridge plot of PC loadings with hierarchically clustered variable order.
    
    Parameters:
    -----------
    pca_components : pd.DataFrame
        PCA loadings matrix (variables × components)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    clustered_variable_names : list
        Variable names in clustered order
    """
    
    # Step 1: Perform hierarchical clustering on variables
    distance_matrix = pdist(pca_components.values, metric='euclidean')
    linkage_matrix = linkage(distance_matrix, method='ward')
    clustered_order = leaves_list(linkage_matrix)
    clustered_variable_names = [pca_components.index[i] for i in clustered_order]
    
    # Step 2: Set up the ridge plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ridge plot parameters
    n_pcs = pca_components.shape[1]
    ridge_height = 0.8
    ridge_spacing = 1.0
    baseline_offset = 0.1
    
    # Color scheme: blue gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_pcs))
    
    # Reorder components according to clustering
    pca_components_clustered = pca_components.reindex(clustered_variable_names)
    
    # Step 3: Create ridge plot for each PC
    for i, pc in enumerate(pca_components_clustered.columns):
        loadings = pca_components_clustered[pc].values
        abs_loadings = np.abs(loadings)
        normalized_loadings = (abs_loadings / abs_loadings.max()) * ridge_height
        
        y_baseline = i * ridge_spacing
        x_positions = np.arange(len(clustered_variable_names))
        
        # Draw bars for each variable
        for j, (x_pos, loading, norm_loading) in enumerate(zip(x_positions, loadings, normalized_loadings)):
            y_bottom = y_baseline + baseline_offset
            y_top = y_bottom + norm_loading
            
            # Style positive and negative loadings differently
            if loading >= 0:
                ax.fill_between([x_pos - 0.4, x_pos + 0.4], [y_bottom, y_bottom], [y_top, y_top], 
                               color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
            else:
                ax.fill_between([x_pos - 0.4, x_pos + 0.4], [y_bottom, y_bottom], [y_top, y_top], 
                               color=colors[i], alpha=0.6, edgecolor='white', linewidth=0.5, hatch='///')
        
        # Add baseline and PC label
        ax.axhline(y=y_baseline + baseline_offset, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.text(-2, y_baseline + baseline_offset + ridge_height/2, pc, 
                fontsize=12, fontweight='bold', ha='right', va='center')
    
    # Step 4: Style the plot
    ax.set_xlim(-3, len(clustered_variable_names))
    ax.set_ylim(-0.2, n_pcs * ridge_spacing + 0.5)
    ax.set_xticks(range(len(clustered_variable_names)))
    ax.set_xticklabels(clustered_variable_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([])
    
    ax.set_title('PC Loadings Ridge Plot\n(Variables Ordered by Hierarchical Clustering)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Chemical Variables (Clustered Order)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Remove spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.8, label='Positive loadings'),
        Patch(facecolor=colors[0], alpha=0.6, hatch='///', label='Negative loadings')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, "PCA_PC_loadings_ridge_plot.png"), dpi=300)
    return fig, ax


def rda_pollution_species_analysis(taxa_df, pc_scores, percentile_threshold=50, n_pcs=6, n_rda_axes=6):
    """
    Perform RDA analysis using pollution PCs to explain species composition at low-pollution sites.
    
    Parameters:
    -----------
    taxa_df : pd.DataFrame
        Taxa abundance data (sites × species)
    pc_scores : pd.DataFrame
        Principal component scores from pollution PCA (sites × PCs)
    percentile_threshold : float, default=50
        Percentile threshold for defining reference sites (e.g., 50 = median)
        Sites with pollution scores <= this percentile are considered reference sites
    n_pcs : int, default=6
        Number of PCs to use as explanatory variables
    n_rda_axes : int, default=6
        Number of RDA axes to extract for visualization
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'rda_model': Fitted RDA model object
        - 'reference_taxa': Hellinger-transformed taxa for reference sites
        - 'reference_pcs': PC scores for reference sites
        - 'rda_scores': RDA scores (site, species, biplot)
        - 'statistics': Dictionary of key statistics
        - 'fig': matplotlib figure object with triplot
    """
    from zci.data_process.transform import hellinger_transform
    from zci.environmental_partition_cluster.rda import RDA
    
    # Step 1: Apply Hellinger transformation to taxa data
    taxa_hellinger_df = hellinger_transform(taxa_df)
    
    print(f"Hellinger transformation completed.")
    print(f"Original taxa data shape: {taxa_df.shape}")
    
    # Step 2: Define reference sites based on pollution scores
    pollution_scores = pc_scores.sum(axis = 1) # Sum of PC scores as pollution score proxy
    threshold_value = np.percentile(pollution_scores, percentile_threshold)
    ref_sites_mask = pollution_scores <= threshold_value
    
    print(f"\nPercentile threshold: {percentile_threshold}th percentile = {threshold_value:.4f}")
    print(f"Number of reference sites: {ref_sites_mask.sum()}")
    print(f"Number of contaminated sites: {(~ref_sites_mask).sum()}")
    
    # Step 3: Extract reference site data
    train_ref_taxa = taxa_hellinger_df[ref_sites_mask]
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    train_ref_habitat = pc_scores[ref_sites_mask][pc_cols]
    
    print(f"\nReference taxa data shape: {train_ref_taxa.shape}")
    print(f"Reference habitat (PC scores) data shape: {train_ref_habitat.shape}")
    
    # Step 4: Fit RDA model
    print("\n" + "="*50)
    print("Fitting RDA model...")
    rda = RDA(center_X=True, center_Y=True, scale_X=False, ddof=1).fit(train_ref_habitat, train_ref_taxa)
    
    # Extract key matrices
    B_hat_df = rda.fit_.coefficients
    hat_train_ref_taxa = rda.fit_.Y_hat
    E_df = rda.fit_.residuals
    cons_eigenvalues = rda.fit_.constrained_eigenvalues
    cons_eigenvectors = rda.fit_.constrained_eigenvectors
    
    # print(f"✓ Coefficients shape: {B_hat_df.shape}")
    # print(f"✓ Constrained eigenvalues shape: {cons_eigenvalues.shape}")
    # print(f"✓ Constrained eigenvectors shape: {cons_eigenvectors.shape}")
    
    # Step 5: Calculate variance partitioning
    total_inertia = rda.fit_.inertia_total
    constrained_inertia = rda.fit_.inertia_constrained
    unconstrained_inertia = rda.fit_.inertia_residual
    
    # print(f"\n📊 Variance Partitioning (Inertia):")
    # print(f"   Total inertia: {total_inertia:.4f}")
    # print(f"   Constrained inertia: {constrained_inertia:.4f} ({100*constrained_inertia/total_inertia:.2f}%)")
    # print(f"   Unconstrained inertia: {unconstrained_inertia:.4f} ({100*unconstrained_inertia/total_inertia:.2f}%)")
    # print(f"   R² = {rda.fit_.r2:.4f}, Adjusted R² = {rda.fit_.r2_adj:.4f}")
    
    # print(f"\n📈 Constrained eigenvalues (first 3):")
    # for i in range(min(3, len(cons_eigenvalues))):
    #     print(f"   RDA{i+1}: {cons_eigenvalues[i]:.4f} ({100*rda.fit_.explained_proportion[i]:.2f}% of constrained variance)")
    
    # Step 6: Get RDA scores
    rda_scores = rda.scores(n_axes=n_rda_axes)
    site_scores = rda_scores.site_scores.values
    species_scores = rda_scores.species_scores.values
    biplot_scores = rda_scores.biplot_scores.values
    
    # Step 7: Create RDA triplot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot sites
    ax.scatter(site_scores[:, 0], site_scores[:, 1], 
               s=80, alpha=0.6, c='steelblue', 
               edgecolors='black', linewidth=0.5,
               label='Reference Sites', zorder=3)
    
    # Label a subset of sites
    n_sites_to_label = min(10, len(site_scores))
    for i in range(n_sites_to_label):
        ax.annotate(train_ref_taxa.index[i], 
                    (site_scores[i, 0], site_scores[i, 1]),
                    fontsize=7, alpha=0.7, xytext=(3, 3),
                    textcoords='offset points')
    
    # Plot species (top 20 by variance)
    species_variance = np.var(species_scores[:, :2], axis=1)
    top_species_idx = np.argsort(species_variance)[-20:]
    
    ax.scatter(species_scores[top_species_idx, 0], 
               species_scores[top_species_idx, 1],
               s=50, alpha=0.5, c='coral', marker='^',
               edgecolors='darkred', linewidth=0.5,
               label='Species (top 20)', zorder=2)
    
    # Add species labels
    for idx in top_species_idx:
        species_name = train_ref_taxa.columns[idx]
        if len(species_name) > 15:
            species_name = species_name[:12] + '...'
        
        scale_factor = 30
        x_pos = species_scores[idx, 0] * scale_factor
        y_pos = species_scores[idx, 1] * scale_factor
        
        ax.annotate(species_name, 
                    (x_pos, y_pos),
                    fontsize=8, alpha=0.8, color='darkred', fontweight='bold',
                    xytext=(3, 3), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', 
                             facecolor='white', alpha=0.7, edgecolor='coral'))
    
    # Plot PC vectors (biplot scores)
    pc_names = train_ref_habitat.columns
    colors_pc = plt.cm.Set2(np.linspace(0, 1, len(pc_names)))
    
    for i, pc_name in enumerate(pc_names):
        scale_factor = 3.0
        x_end = biplot_scores[i, 0] * scale_factor
        y_end = biplot_scores[i, 1] * scale_factor
        
        ax.arrow(0, 0, x_end, y_end,
                 head_width=0.1, head_length=0.15,
                 fc=colors_pc[i], ec=colors_pc[i],
                 alpha=0.8, linewidth=2.5, zorder=4,
                 length_includes_head=True)
        
        ax.text(x_end * 1.1, y_end * 1.1, pc_name,
                fontsize=15, fontweight='bold',
                color=colors_pc[i], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', alpha=0.7, edgecolor=colors_pc[i]))
    
    # Add grid and reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.3, zorder=1)
    
    # Labels and title
    var_explained_rda1 = 100 * rda.fit_.explained_proportion.iloc[0]
    var_explained_rda2 = 100 * rda.fit_.explained_proportion.iloc[1]
    
    ax.set_xlabel(f'RDA1 ({var_explained_rda1:.2f}% of constrained variance)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(f'RDA2 ({var_explained_rda2:.2f}% of constrained variance)', 
                  fontsize=13, fontweight='bold')
    
    title = f'RDA Triplot: Taxa Composition vs. {n_pcs} Pollution PCs\n'
    title += f'(Reference Sites ≤ {percentile_threshold}th percentile, Hellinger-transformed Taxa)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add statistics text box
    stats_text = f"Statistics:\n"
    stats_text += f"Coefficients: {B_hat_df.shape}\n"
    stats_text += f"Eigenvalues: {cons_eigenvalues.shape}\n"
    stats_text += f"Eigenvectors: {cons_eigenvectors.shape}\n"
    stats_text += f"Constrained inertia: {constrained_inertia:.4f}\n"
    stats_text += f"Unconstrained inertia: {unconstrained_inertia:.4f}\n"
    stats_text += f"R² = {rda.fit_.r2:.4f} | Adj. R² = {rda.fit_.r2_adj:.4f}"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=15, framealpha=0.9)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, "RDA_triplot_pollution_species.png"), dpi=300)
    
    print("\n" + "="*50)
    print("✓ RDA triplot created successfully!")
    print(f"✓ RDA1 + RDA2 explain {var_explained_rda1 + var_explained_rda2:.2f}% of constrained variance")
    
    # Compile statistics
    statistics = {
        'coefficients_shape': B_hat_df.shape,
        'constrained_eigenvalues_shape': cons_eigenvalues.shape,
        'constrained_eigenvectors_shape': cons_eigenvectors.shape,
        'total_inertia': total_inertia,
        'constrained_inertia': constrained_inertia,
        'unconstrained_inertia': unconstrained_inertia,
        'r2': rda.fit_.r2,
        'r2_adj': rda.fit_.r2_adj,
        'n_reference_sites': ref_sites_mask.sum(),
        'n_contaminated_sites': (~ref_sites_mask).sum(),
        'pollution_threshold': threshold_value,
        'percentile': percentile_threshold
    }

def compute_pollution_scores(pca_scores, weights=None):
    """
    Compute pollution scores by weighted sum of PC scores.
    
    Parameters:
    -----------
    pca_scores : pd.DataFrame
        DataFrame of PCA scores (sites × PCs), e.g., columns=['PC1', 'PC2', ...]
    weights : dict, list, array, or None, default=None
        Weights for each PC. Can be:
        - dict: {'PC1': 1.0, 'PC2': 1.5, ...} - maps PC names to weights
        - list/array: [1.0, 1.5, ...] - weights in same order as pca_scores columns
        - None: equal weights (1.0) for all PCs
    
    Returns:
    --------
    pd.DataFrame
        Single-column DataFrame with pollution scores (index preserved from pca_scores)
    
    Examples:
    ---------
    # Equal weights
    scores = compute_pollution_scores(pca_scores)
    
    # Custom weights as dict
    scores = compute_pollution_scores(pca_scores, {'PC1': 1, 'PC2': 1, 'PC3': 2})
    
    # Custom weights as list
    scores = compute_pollution_scores(pca_scores, [1, 1, 2, 0, 1, 1])
    """
    
    # Handle different weight input types
    if weights is None:
        # give more weight to PC3 based on RDA analysis
        pc_weights = {
            'PC1': 1,     # Highest variance, likely general contamination
            'PC2': 1,     # Second highest variance
            'PC3': 2,     # High biological impact based on RDA results
            'PC5': 1,     # Lower variance but still relevant
            'PC6': 1      # Lowest variance among selected PCs
        }
        weight_array = np.array([pc_weights.get(pc, 0) for pc in pca_scores.columns])
        
    elif isinstance(weights, dict):
        # Convert dictionary to array matching pca_scores column order
        weight_array = np.zeros(pca_scores.shape[1])
        for i, pc_name in enumerate(pca_scores.columns):
            weight_array[i] = weights.get(pc_name, 0)  # Default to 0 if PC not in dict
            
    else:
        # Assume it's array-like (list, tuple, or numpy array)
        weight_array = np.array(weights)
        if len(weight_array) != pca_scores.shape[1]:
            raise ValueError(
                f"Length of weights ({len(weight_array)}) must match "
                f"number of PCs in pca_scores ({pca_scores.shape[1]})"
            )
    
    # Compute weighted sum: each site's pollution score = sum(PC_i * weight_i)
    pollution_scores = pca_scores.values @ weight_array
    
    # Return as DataFrame with preserved index
    pollution_scores_df = pd.DataFrame(
        pollution_scores, 
        index=pca_scores.index, 
        columns=["Pollution_Score"]
    )
    
    return pollution_scores_df


def merge_pollution_scores_into_data(pollution_scores, raw_data, multiindex_data, 
                                     multiindex_levels=('pollution', 'weighted', 'SumRel')):
    """
    Merge pollution scores into both raw (flat) and multi-index dataframes in-place.
    
    Parameters:
    -----------
    pollution_scores : pd.DataFrame
        Pollution scores DataFrame (single column, typically from compute_pollution_scores)
    raw_data : pd.DataFrame
        Flat DataFrame without multi-index columns (modified in-place)
    multiindex_data : pd.DataFrame
        DataFrame with multi-index columns (modified in-place)
    multiindex_levels : tuple, default=('pollution', 'weighted', 'SumRel')
        Three-level tuple for creating multi-index column name
    
    Returns:
    --------
    raw_data, multiindex_data : tuple of pd.DataFrame
        Both dataframes with pollution scores added (same objects as inputs, modified in-place)
    
    Example:
    --------
    pollution_scores = compute_pollution_scores(PC_scores)
    raw_data, data = merge_pollution_scores_into_data(pollution_scores, raw_data, data)
    """
    from zci.data_process.dataframe_ops import concat_blocks
    
    # Add to raw_data (simple concatenation)
    raw_data_merged = pd.concat([raw_data, pollution_scores], axis=1)
    
    # Add to multi-index data with proper column structure
    multiindex_col = pd.MultiIndex.from_tuples([multiindex_levels])
    pollution_multiindex_df = pd.DataFrame(
        pollution_scores.values,
        index=pollution_scores.index,
        columns=multiindex_col
    )
    multiindex_data_merged = concat_blocks([multiindex_data, pollution_multiindex_df])
    
    return raw_data_merged, multiindex_data_merged


def contamination_assessment_pipeline(
    data,
    transform_method='log_z_score',
    standardize_pca_scores=True,
    pc_weights=None,
    multiindex_levels=('pollution', 'weighted', 'SumRel'),
    visualize=True,
    run_rda_validation=False,
    rda_percentile_threshold=50,
    save_figures=True
):
    """
    Complete pipeline for contamination assessment from raw data to pollution scores.
    
    This pipeline performs the following steps:
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
        Whether to save generated figures to disk
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'raw_data': Updated raw dataframe with Pollution_Score column
        - 'multiindex_data': Updated multi-index data with pollution score
        - 'transformed_pollution': Transformed pollution variables (log + z-score)
        - 'PC_loadings': PCA component loadings
        - 'PC_scores': PCA component scores
        - 'pollution_scores': Final weighted pollution scores
        - 'ordered_variables': Variables ordered by hierarchical clustering (optional)
        - 'rda_results': RDA validation results (if run_rda_validation=True)
    
    Example Usage:
    --------------
    # Basic usage with default parameters
    results = contamination_assessment_pipeline(data)
    updated_data = results['multiindex_data']
    pollution_scores = results['pollution_scores']
    
    # Custom weights focusing on specific PCs
    custom_weights = {'PC1': 1.5, 'PC2': 1.0, 'PC3': 2.0, 'PC4': 0.5, 'PC5': 1.0, 'PC6': 0.5}
    results = contamination_assessment_pipeline(
        data,
        pc_weights=custom_weights,
        run_rda_validation=True,
        rda_percentile_threshold=25  # Only bottom quartile as reference
    )
    
    # Quick run without visualizations
    results = contamination_assessment_pipeline(
        data,
        visualize=False,
        run_rda_validation=False
    )
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
    
    # =====================================================================
    # Step 4: PCA Analysis
    # =====================================================================
    print("\n[Step 4/7] Performing PCA on transformed pollution data...")
    
    PC_loadings, PC_scores = pca_with_PC_loadings(
        transformed_pollution,
        visualize=visualize,
        PC_scores_standardize=standardize_pca_scores
    )
    
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
        create_ridge_plot(PC_loadings)
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
        multiindex_levels=multiindex_levels
    )
    
    print(f"  ✓ Pollution scores added to data")
    print(f"  ✓ Multi-index column: {multiindex_levels}")
    print(f"  ✓ Raw data column: 'Pollution_Score'")
    
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
        'rda_results': rda_results
    }
    
    return results
    


