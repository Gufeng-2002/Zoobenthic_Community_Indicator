"""
Principal Component Analysis (PCA) for pollution variables.

This module provides PCA computation on transformed pollution data, extracting
principal components that capture the major patterns of contamination.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Set the figure save path
FIGURE_SAVE_PATH = "../results/figures/01_Contamination_assessment/"


def pca_with_PC_loadings(df, visualize=True, PC_scores_standardize=True, n_components=6, save_path=None):
    """
    Apply PCA on transformed pollution variables.
    
    Extracts principal components that explain the major variance in pollution data.
    By default, returns the first 6 PCs which typically explain >80% of total variance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transformed pollution variables (should already be log-transformed and standardized)
    visualize : bool, default=True
        Whether to create and display variance explanation plots
    PC_scores_standardize : bool, default=True
        Whether to standardize PC scores (mean=0, std=1) after extraction
    n_components : int, default=6
        Number of principal components to retain
    save_path : str, optional
        If provided, save figures to this directory
    
    Returns:
    --------
    PC_loadings : pd.DataFrame
        PCA component loadings (variables × components)
        Shows the contribution of each pollution variable to each PC
    PC_scores : pd.DataFrame
        PCA component scores (sites × components)
        Shows the position of each site in the PC space
    variance_fig : plt.Figure or None
        Figure showing variance explained (if visualize=True)
    
    Notes:
    ------
    - The first PC typically captures general contamination levels
    - Subsequent PCs capture specific contamination patterns
    - PC scores are standardized by default for easier interpretation
    
    Example:
    --------
    >>> transformed_pollution = log_z_score_transform(pollution_data)
    >>> PC_loadings, PC_scores, fig = pca_with_PC_loadings(transformed_pollution)
    >>> print(f"First PC explains {PC_loadings.iloc[:, 0].abs().sum():.2f} total loading")
    """
    # Fit PCA model
    pca_model = PCA()
    pca_model.fit(df)
    
    # Extract loadings (variables × components)
    PC_loadings = pd.DataFrame(
        np.transpose(pca_model.components_[:n_components]),
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Extract scores (sites × components)
    PC_scores = pd.DataFrame(
        pca_model.transform(df)[:, :n_components],
        index=df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Optionally standardize PC scores
    if PC_scores_standardize:
        PC_scores = (PC_scores - PC_scores.mean()) / PC_scores.std()
    
    # Visualization
    variance_fig = None
    if visualize:
        variance_fig = _plot_pca_variance(pca_model, save_path=save_path)
    
    return PC_loadings, PC_scores, variance_fig


def _plot_pca_variance(pca_model, save_path=None):
    """
    Create variance explanation plots for PCA results.
    
    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    save_path : str, optional
        If provided, save the figure to this directory
    
    Returns:
    --------
    fig : plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n_components = len(pca_model.explained_variance_ratio_)
    
    # Subplot 1: Individual variance explained
    ax1.bar(range(1, n_components + 1), pca_model.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax1.set_title('Explained Variance Ratio by Component', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative variance explained
    cumulative_var = pca_model.explained_variance_ratio_.cumsum()
    ax2.plot(range(1, n_components + 1), cumulative_var, 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, label='80% variance', alpha=0.7)
    ax2.axhline(y=0.9, color='g', linestyle='--', linewidth=1.5, label='90% variance', alpha=0.7)
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=11)
    ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "PCA_explained_variance.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig
