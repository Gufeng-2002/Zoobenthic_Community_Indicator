"""
PCA Analysis Module for Community Composition

This module provides functions for fitting PCA models on transformed 
species data within each habitat cluster.

Supported transformations:
- Hellinger: sqrt(abundance / row_total)
- Chord: abundance / sqrt(sum of squared abundances)
- Octave: log2(abundance + 1)
- Bray-Curtis: (abundance - row_mean) / (row_max - row_min), then standardized
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
from zci.taxa_assemblage_in_refs.hierarchical_clustering import (
    hellinger_transform,
    chord_transform,
    octave_transform
)



# =============================================================================
# TAXA TRANSFORMATION FUNCTIONS
# =============================================================================

def bray_curtis_transform(df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
    """
    Apply Bray-Curtis inspired transformation to taxa abundance data.
    
    This transformation prepares data for PCA while maintaining some properties
    relevant to Bray-Curtis dissimilarity. It involves:
    1. Converting abundances to proportions (row-wise) - skipped if already proportions
    2. Optional standardization (column-wise z-score)
    
    Note: True Bray-Curtis is a dissimilarity metric, not a transformation.
    This transformation creates proportional data suitable for PCA that 
    considers relative abundances similar to Bray-Curtis concepts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Taxa abundance data (sites x species). Can be raw counts or
        relative abundances (proportions).
    standardize : bool
        Whether to standardize columns after proportional transformation
        
    Returns:
    --------
    pd.DataFrame
        Transformed data with same shape and indices
    """
    if is_relative_abundance:
        # Data is already proportions
        proportions = df.copy()
    else:
        # Convert to proportions (row-wise)
        row_sums = df.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        proportions = df.div(row_sums, axis=0)
        proportions = proportions.fillna(0)
    
    if standardize:
        # Standardize each column (species) to mean=0, std=1
        scaler = StandardScaler()
        transformed_values = scaler.fit_transform(proportions.values)
        transformed = pd.DataFrame(
            transformed_values,
            index=df.index,
            columns=df.columns
        )
    else:
        transformed = proportions
    
    return transformed


def transform_taxa_data(
    df: pd.DataFrame, 
    method: str = 'hellinger',
    is_relative_abundance: bool = False
) -> pd.DataFrame:
    """
    Apply a specified transformation to taxa abundance data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Taxa abundance data (sites x species). Can be raw counts or
        relative abundances (proportions).
    method : str
        Transformation method. Options:
        - 'hellinger': Hellinger transformation (default)
        - 'chord': Chord (L2 normalization) transformation
        - 'octave': Log2 transformation
        - 'bray-curtis' or 'bray_curtis': Proportional + standardization
        - 'none' or None: No transformation (raw data)
    is_relative_abundance : bool, default=False
        If True, the input data is already in relative abundance format
        (proportions that sum to 1 for each row/site). This skips the
        proportion/relative abundance calculation step in transformations
        that would otherwise compute it.
        
    Returns:
    --------
    pd.DataFrame
        Transformed data with same shape and indices
        
    Raises:
    -------
    ValueError
        If unknown transformation method is specified
    """
    if method is None or method.lower() == 'none':
        return df.copy()
    
    method = method.lower().replace('-', '_')
    
    if method == 'hellinger':
        return hellinger_transform(df)
    elif method == 'chord':
        return chord_transform(df)
    elif method == 'octave':
        return octave_transform(df)
    elif method in ['bray_curtis', 'braycurtis']:
        return bray_curtis_transform(df)
    else:
        raise ValueError(
            f"Unknown transformation method: '{method}'. "
            f"Supported methods: 'hellinger', 'chord', 'octave', 'bray-curtis', 'none'"
        )


# =============================================================================
# PCA UTILITY FUNCTIONS
# =============================================================================


def get_n_components_for_variance(
    pca_model: PCA,
    variance_threshold: float = 0.70
) -> int:
    """
    Determine the number of components needed to explain a given variance threshold.
    
    Parameters:
    -----------
    pca_model : PCA
        Fitted PCA model (should be fitted with all possible components)
    variance_threshold : float
        Cumulative variance threshold (default: 0.70 for 70%)
        
    Returns:
    --------
    int
        Number of components needed to explain >= variance_threshold
    """
    cumsum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    return max(n_components, 1)  # At least 1 component


def fit_cluster_pcas(
    taxa_data: pd.DataFrame,
    cluster_labels: pd.Series,
    site_mask: Optional[pd.Series] = None,
    variance_threshold: float = 0.70,
    taxa_transformation: str = 'hellinger',
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Fit separate PCA models for each cluster on transformed species data.
    
    Parameters:
    -----------
    taxa_data : pd.DataFrame
        Raw taxa abundance data (sites x species). Will be transformed.
    cluster_labels : pd.Series
        Cluster assignment for each site (index should match taxa_data index)
    site_mask : pd.Series, optional
        Boolean mask indicating which sites to use for PCA fitting.
        If None, all sites are used. Use this to select training sites
        based on pollution percentile.
    variance_threshold : float
        Cumulative variance threshold for selecting number of PCs (default: 0.70)
    taxa_transformation : str
        Transformation method for taxa data. Options:
        - 'hellinger': Hellinger transformation (default)
        - 'chord': Chord (L2 normalization) transformation
        - 'octave': Log2 transformation
        - 'bray-curtis': Proportional + standardization
        - 'none': No transformation
    random_state : int
        Random state for reproducibility
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'pca_models': {cluster_id: fitted PCA model}
        - 'taxa_transformed': Transformed taxa data
        - 'transformation_method': Name of transformation used
        - 'training_sites': {cluster_id: list of training site indices}
        - 'n_components': {cluster_id: number of PCs used}
        - 'variance_explained': {cluster_id: array of variance explained per PC}
        - 'total_variance': {cluster_id: total variance explained}
        - 'pca_coordinates': {cluster_id: DataFrame with PC coordinates}
        - 'loadings': {cluster_id: DataFrame with species loadings}
    """
    # Apply transformation
    taxa_transformed = transform_taxa_data(taxa_data, method=taxa_transformation)
    
    # Get unique clusters
    unique_clusters = sorted(cluster_labels.dropna().unique())
    
    if verbose:
        print("=" * 80)
        print("FITTING PCA MODELS FOR EACH CLUSTER")
        print("=" * 80)
        print(f"Taxa transformation: {taxa_transformation}")
        print(f"Variance threshold: {variance_threshold * 100:.0f}%")
        print(f"Number of clusters: {len(unique_clusters)}")
    
    # Initialize result containers
    pca_models = {}
    training_sites = {}
    n_components_dict = {}
    variance_explained = {}
    total_variance = {}
    pca_coordinates = {}
    loadings_dict = {}
    
    for cluster in unique_clusters:
        if verbose:
            print(f"\n--- Cluster {int(cluster)} ---")
        
        # Get sites in this cluster
        cluster_mask = cluster_labels == cluster
        
        # Apply site mask if provided (e.g., pollution-based selection)
        if site_mask is not None:
            training_mask = cluster_mask & site_mask
        else:
            training_mask = cluster_mask
        
        # Get training sites
        train_sites = taxa_transformed.index[training_mask].tolist()
        training_sites[cluster] = train_sites
        
        if len(train_sites) < 3:
            if verbose:
                print(f"  WARNING: Only {len(train_sites)} sites - skipping PCA")
            continue
        
        # Get transformed data for training sites
        cluster_taxa = taxa_transformed.loc[train_sites]
        
        if verbose:
            print(f"  Training sites: {len(train_sites)}")
            print(f"  Number of species: {cluster_taxa.shape[1]}")
        
        # Fit PCA with maximum possible components first
        max_components = min(cluster_taxa.shape[0] - 1, cluster_taxa.shape[1])
        pca_full = PCA(n_components=max_components, random_state=random_state)
        pca_full.fit(cluster_taxa.values)
        
        # Determine number of components for variance threshold
        n_components = get_n_components_for_variance(pca_full, variance_threshold)
        n_components_dict[cluster] = n_components
        
        # Fit final PCA with optimal number of components
        pca = PCA(n_components=n_components, random_state=random_state)
        pca_coords = pca.fit_transform(cluster_taxa.values)
        
        # Store results
        pca_models[cluster] = pca
        variance_explained[cluster] = pca.explained_variance_ratio_ * 100
        total_variance[cluster] = variance_explained[cluster].sum()
        
        # Create coordinates DataFrame
        coord_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_coordinates[cluster] = pd.DataFrame(
            pca_coords,
            index=train_sites,
            columns=coord_columns
        )
        
        # Create loadings DataFrame
        loadings_dict[cluster] = pd.DataFrame(
            pca.components_.T,
            index=cluster_taxa.columns,
            columns=coord_columns
        )
        
        if verbose:
            print(f"  Number of PCs: {n_components}")
            print(f"  Variance explained:")
            for i in range(min(3, n_components)):
                print(f"    PC{i+1}: {variance_explained[cluster][i]:.2f}%")
            print(f"  Total variance: {total_variance[cluster]:.2f}%")
    
    return {
        'pca_models': pca_models,
        'taxa_transformed': taxa_transformed,
        'taxa_hellinger': taxa_transformed,  # Backward compatibility alias
        'transformation_method': taxa_transformation,
        'training_sites': training_sites,
        'n_components': n_components_dict,
        'variance_explained': variance_explained,
        'total_variance': total_variance,
        'pca_coordinates': pca_coordinates,
        'loadings': loadings_dict,
        'variance_threshold': variance_threshold
    }


def transform_sites_with_pca(
    taxa_transformed: pd.DataFrame,
    pca_model: PCA,
    sites: List
) -> pd.DataFrame:
    """
    Transform sites using a fitted PCA model.
    
    Parameters:
    -----------
    taxa_transformed : pd.DataFrame
        Transformed taxa data (e.g., Hellinger, chord, etc.)
    pca_model : PCA
        Fitted PCA model
    sites : list
        List of site indices to transform
        
    Returns:
    --------
    pd.DataFrame
        PCA coordinates for the transformed sites
    """
    # Get available sites
    available_sites = [s for s in sites if s in taxa_transformed.index]
    
    if not available_sites:
        return pd.DataFrame()
    
    # Get data for sites
    site_data = taxa_transformed.loc[available_sites]
    
    # Transform using PCA
    coords = pca_model.transform(site_data.values)
    
    # Create DataFrame
    n_components = coords.shape[1]
    coord_columns = [f'PC{i+1}' for i in range(n_components)]
    
    return pd.DataFrame(
        coords,
        index=available_sites,
        columns=coord_columns
    )
