"""
Validation utilities for contamination assessment.

This module provides RDA (Redundancy Analysis) validation to assess the biological
relevance of pollution principal components by examining their influence on species
composition at reference sites.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from zci.env_driven_taxa_clusters.rda import RDA

# Set the figure save path
FIGURE_SAVE_PATH = "../results/figures/01_Contamination_assessment/"


def rda_pollution_species_analysis(taxa_df, pc_scores, percentile_threshold=50, 
                                   n_pcs=6, n_rda_axes=6):
    """
    Perform RDA analysis using pollution PCs to explain species composition.
    
    This validation approach helps identify which pollution PCs have the strongest
    biological impact on benthic communities by analyzing reference (low-pollution) sites.
    
    Parameters:
    -----------
    taxa_df : pd.DataFrame
        Taxa abundance data (sites × species)
    pc_scores : pd.DataFrame
        Principal component scores from pollution PCA (sites × PCs)
    percentile_threshold : float, default=50
        Percentile threshold for defining reference sites (e.g., 50 = median).
        Sites with pollution scores ≤ this percentile are considered reference sites.
    n_pcs : int, default=6
        Number of PCs to use as explanatory variables
    n_rda_axes : int, default=6
        Number of RDA axes to extract for visualization
    
    Returns:
    --------
    dict or None
        Dictionary containing RDA results including statistics, scores, and figure.
        Returns None if RDA fails.
    
    Notes:
    ------
    - Uses Hellinger transformation on taxa data to handle compositional nature
    - Focuses on reference sites to avoid confounding effects of severe contamination
    - Creates a triplot showing sites, species, and pollution PC vectors
    - High R² indicates strong pollution influence on community composition
    
    Example:
    --------
    >>> from zci.data_process.dataframe_ops import get_block
    >>> taxa_df = get_block(data, "taxa")
    >>> rda_results = rda_pollution_species_analysis(taxa_df, PC_scores, percentile_threshold=50)
    >>> print(f"R² = {rda_results['statistics']['r2']:.4f}")
    """
    from zci.data_process.transform import hellinger_transform
    from zci.env_driven_taxa_clusters.rda import RDA
    
    # Step 1: Apply Hellinger transformation to taxa data
    taxa_hellinger_df = hellinger_transform(taxa_df)
    
    print(f"Hellinger transformation completed.")
    print(f"Original taxa data shape: {taxa_df.shape}")
    
    # Step 2: Define reference sites based on pollution scores
    pollution_scores = pc_scores.sum(axis=1)  # Sum of PC scores as pollution proxy
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
    rda = RDA(center_X=True, center_Y=True, scale_X=False, ddof=1).fit(
        train_ref_habitat, train_ref_taxa
    )
    
    # Extract key matrices
    B_hat_df = rda.fit_.coefficients
    cons_eigenvalues = rda.fit_.constrained_eigenvalues
    cons_eigenvectors = rda.fit_.constrained_eigenvectors
    
    # Step 5: Calculate variance partitioning
    total_inertia = rda.fit_.inertia_total
    constrained_inertia = rda.fit_.inertia_constrained
    unconstrained_inertia = rda.fit_.inertia_residual
    
    # Step 6: Get RDA scores
    rda_scores = rda.scores(n_axes=n_rda_axes)
    site_scores = rda_scores.site_scores.values
    species_scores = rda_scores.species_scores.values
    biplot_scores = rda_scores.biplot_scores.values
    
    # Step 7: Create RDA triplot
    fig = _create_rda_triplot(
        site_scores, species_scores, biplot_scores,
        train_ref_taxa, train_ref_habitat,
        rda.fit_, percentile_threshold, n_pcs
    )
    
    var_explained_rda1 = 100 * rda.fit_.explained_proportion.iloc[0]
    var_explained_rda2 = 100 * rda.fit_.explained_proportion.iloc[1]
    
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
    
    return {
        'rda_model': rda,
        'reference_taxa': train_ref_taxa,
        'reference_pcs': train_ref_habitat,
        'rda_scores': rda_scores,
        'statistics': statistics,
        'fig': fig
    }


def _create_rda_triplot(site_scores, species_scores, biplot_scores,
                        train_ref_taxa, train_ref_habitat,
                        rda_fit, percentile_threshold, n_pcs):
    """
    Create RDA triplot visualization.
    
    Internal function to generate the RDA triplot showing sites, species,
    and pollution PC vectors.
    """
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
    var_explained_rda1 = 100 * rda_fit.explained_proportion.iloc[0]
    var_explained_rda2 = 100 * rda_fit.explained_proportion.iloc[1]
    
    ax.set_xlabel(f'RDA1 ({var_explained_rda1:.2f}% of constrained variance)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(f'RDA2 ({var_explained_rda2:.2f}% of constrained variance)', 
                  fontsize=13, fontweight='bold')
    
    title = f'RDA Triplot: Taxa Composition vs. {n_pcs} Pollution PCs\n'
    title += f'(Reference Sites ≤ {percentile_threshold}th percentile, Hellinger-transformed Taxa)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add statistics text box
    stats_text = f"Statistics:\n"
    stats_text += f"R² = {rda_fit.r2:.4f} | Adj. R² = {rda_fit.r2_adj:.4f}\n"
    stats_text += f"Constrained inertia: {rda_fit.inertia_constrained:.4f}\n"
    stats_text += f"Unconstrained inertia: {rda_fit.inertia_residual:.4f}"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=15, framealpha=0.9)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_SAVE_PATH, "RDA_triplot_pollution_species.png"), dpi=300)
    
    return fig
