"""
ZCI (Zoobenthic Community Indicator) Calculation Module

This module provides functions for calculating ZCI - the compositional distance
from reference conditions in PC space - and for projecting non-training sites
onto fitted PC space.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union


def calculate_zci_for_cluster(
    pca_coordinates: pd.DataFrame,
    reference_sites: Optional[List] = None,
    use_centroid: bool = True,
    normalize_pcs: bool = True
) -> Tuple[pd.Series, np.ndarray]:
    """
    Calculate ZCI (compositional distance from reference) for a single cluster.
    
    ZCI is the Euclidean distance in multidimensional PC space from the
    reference centroid (or a specific reference point).
    
    Parameters:
    -----------
    pca_coordinates : pd.DataFrame
        PCA coordinates for sites (sites x PCs)
    reference_sites : list, optional
        List of site indices to use as reference. If None, uses all sites.
    use_centroid : bool
        If True, use centroid of reference sites as reference point.
        If False, use the first reference site.
    normalize_pcs : bool
        If True (default), z-score normalize PC coordinates before calculating
        distances. This ensures all PCs contribute equally regardless of their
        scale/variance.
        
    Returns:
    --------
    tuple
        (ZCI Series indexed by site, reference_point array)
    """
    if reference_sites is None:
        reference_sites = pca_coordinates.index.tolist()
    
    # Get available reference sites
    available_refs = [s for s in reference_sites if s in pca_coordinates.index]
    
    if not available_refs:
        raise ValueError("No reference sites found in PCA coordinates")
    
    # Apply z-score normalization if requested
    if normalize_pcs:
        # Z-score normalize each PC column
        coords_normalized = pca_coordinates.copy()
        for col in coords_normalized.columns:
            mean_val = coords_normalized[col].mean()
            std_val = coords_normalized[col].std()
            if std_val > 0:
                coords_normalized[col] = (coords_normalized[col] - mean_val) / std_val
            else:
                coords_normalized[col] = 0  # Constant column
        working_coords = coords_normalized
    else:
        working_coords = pca_coordinates
    
    # Calculate reference point
    if use_centroid:
        ref_coords = working_coords.loc[available_refs]
        reference_point = ref_coords.mean().values
    else:
        reference_point = working_coords.loc[available_refs[0]].values
    
    # Calculate Euclidean distance from reference point for all sites
    coords = working_coords.values
    distances = np.sqrt(((coords - reference_point) ** 2).sum(axis=1))
    
    zci = pd.Series(distances, index=pca_coordinates.index, name='ZCI')
    
    return zci, reference_point


def calculate_zci_all_clusters(
    pca_results: Dict,
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    pollution_column: str = 'Pollution_Score',
    use_ref_centroid: bool = True,
    normalize_pcs: bool = True
) -> pd.DataFrame:
    """
    Calculate ZCI for all sites across all clusters.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas() containing:
        - 'pca_coordinates': {cluster_id: DataFrame}
        - 'n_components': {cluster_id: int}
    raw_data : pd.DataFrame
        Raw data with cluster labels, reference indicator, and pollution scores
    cluster_column : str
        Column name for cluster labels
    ref_column : str
        Column name for reference site indicator (boolean or 0/1)
    pollution_column : str
        Column name for pollution scores
    use_ref_centroid : bool
        Use centroid of reference sites as reference point
    normalize_pcs : bool
        If True (default), z-score normalize PC coordinates before calculating
        distances. This ensures all PCs contribute equally.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Site, Cluster, Site_Type, ZCI, Pollution_Score,
        PC1, PC2, ..., N_PCs_Used, Reference_Point
    """
    pca_coordinates = pca_results['pca_coordinates']
    n_components = pca_results['n_components']
    
    results = []
    reference_points = {}
    
    for cluster in sorted(pca_coordinates.keys()):
        coords = pca_coordinates[cluster]
        n_pcs = n_components[cluster]
        
        # Get reference sites for this cluster
        cluster_mask = raw_data[cluster_column] == cluster
        ref_mask = raw_data[ref_column].isin([True, 1])
        ref_sites = raw_data[cluster_mask & ref_mask].index.tolist()
        
        # Calculate ZCI
        zci, ref_point = calculate_zci_for_cluster(
            coords, 
            reference_sites=ref_sites if ref_sites else None,
            use_centroid=use_ref_centroid,
            normalize_pcs=normalize_pcs
        )
        reference_points[cluster] = ref_point
        
        # Build results for each site
        for site in coords.index:
            site_type = 'Reference' if site in ref_sites else 'Training'
            pollution = raw_data.loc[site, pollution_column] if site in raw_data.index else np.nan
            
            result = {
                'Site': site,
                'Cluster': cluster,
                'Site_Type': site_type,
                'ZCI': zci[site],
                'Pollution_Score': pollution,
                'N_PCs_Used': n_pcs
            }
            
            # Add PC coordinates
            for i in range(n_pcs):
                result[f'PC{i+1}'] = coords.loc[site, f'PC{i+1}']
            
            results.append(result)
    
    zci_df = pd.DataFrame(results)
    zci_df['Reference_Point'] = zci_df['Cluster'].map(
        lambda c: str(reference_points.get(c, []))
    )
    
    return zci_df


def project_sites_to_pc_space(
    taxa_hellinger: pd.DataFrame,
    pca_models: Dict,
    sites_to_project: pd.Series,
    cluster_labels: pd.Series,
    training_sites: Dict
) -> Dict[int, pd.DataFrame]:
    """
    Project non-training sites onto fitted PC space for each cluster.
    
    Parameters:
    -----------
    taxa_hellinger : pd.DataFrame
        Hellinger-transformed taxa data (sites x species)
    pca_models : dict
        {cluster_id: fitted PCA model}
    sites_to_project : pd.Series
        Series indicating which sites to project (boolean mask)
    cluster_labels : pd.Series
        Cluster assignment for each site
    training_sites : dict
        {cluster_id: list of training site indices}
        
    Returns:
    --------
    dict
        {cluster_id: DataFrame with PC coordinates for projected sites}
    """
    projected_coords = {}
    
    for cluster, pca_model in pca_models.items():
        # Get sites in this cluster that need projection
        cluster_mask = cluster_labels == cluster
        projection_mask = cluster_mask & sites_to_project
        
        # Exclude training sites
        train_set = set(training_sites.get(cluster, []))
        sites = [s for s in taxa_hellinger.index[projection_mask] 
                 if s not in train_set]
        
        if not sites:
            projected_coords[cluster] = pd.DataFrame()
            continue
        
        # Get available sites
        available_sites = [s for s in sites if s in taxa_hellinger.index]
        
        if not available_sites:
            projected_coords[cluster] = pd.DataFrame()
            continue
        
        # Get data and project
        site_data = taxa_hellinger.loc[available_sites]
        coords = pca_model.transform(site_data.values)
        
        # Create DataFrame
        n_components = coords.shape[1]
        coord_columns = [f'PC{i+1}' for i in range(n_components)]
        
        projected_coords[cluster] = pd.DataFrame(
            coords,
            index=available_sites,
            columns=coord_columns
        )
    
    return projected_coords


def calculate_zci_for_projected_sites(
    projected_coordinates: Dict[int, pd.DataFrame],
    reference_points: Dict[int, np.ndarray],
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    pollution_column: str = 'Pollution_Score',
    normalize_pcs: bool = True,
    training_coords_stats: Optional[Dict[int, Dict[str, pd.Series]]] = None
) -> pd.DataFrame:
    """
    Calculate ZCI for projected (non-training) sites.
    
    Parameters:
    -----------
    projected_coordinates : dict
        {cluster_id: DataFrame with PC coordinates}
    reference_points : dict
        {cluster_id: reference point array}
    raw_data : pd.DataFrame
        Raw data with pollution scores
    cluster_column : str
        Column name for cluster labels
    pollution_column : str
        Column name for pollution scores
    normalize_pcs : bool
        If True (default), z-score normalize PC coordinates before calculating
        distances. Uses training set statistics if provided.
    training_coords_stats : dict, optional
        {cluster_id: {'mean': pd.Series, 'std': pd.Series}} containing
        mean and std for each PC from training data. Required if normalize_pcs
        is True for consistent normalization with training data.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ZCI values for projected sites
    """
    results = []
    
    for cluster, coords in projected_coordinates.items():
        if coords.empty:
            continue
        
        ref_point = reference_points.get(cluster)
        if ref_point is None:
            continue
        
        # Apply z-score normalization if requested
        if normalize_pcs:
            working_coords = coords.copy()
            if training_coords_stats and cluster in training_coords_stats:
                # Use training set statistics for consistent normalization
                stats = training_coords_stats[cluster]
                for col in working_coords.columns:
                    if col in stats['mean'].index:
                        mean_val = stats['mean'][col]
                        std_val = stats['std'][col]
                        if std_val > 0:
                            working_coords[col] = (working_coords[col] - mean_val) / std_val
                        else:
                            working_coords[col] = 0
            else:
                # Normalize using projected coords only (less ideal)
                for col in working_coords.columns:
                    mean_val = working_coords[col].mean()
                    std_val = working_coords[col].std()
                    if std_val > 0:
                        working_coords[col] = (working_coords[col] - mean_val) / std_val
                    else:
                        working_coords[col] = 0
        else:
            working_coords = coords
        
        # Calculate distances
        distances = np.sqrt(((working_coords.values - ref_point) ** 2).sum(axis=1))
        
        for i, site in enumerate(coords.index):
            pollution = raw_data.loc[site, pollution_column] if site in raw_data.index else np.nan
            
            result = {
                'Site': site,
                'Cluster': cluster,
                'Site_Type': 'Projected',
                'ZCI': distances[i],
                'Pollution_Score': pollution,
                'N_PCs_Used': coords.shape[1]
            }
            
            # Add PC coordinates
            for j, col in enumerate(coords.columns):
                result[col] = coords.iloc[i, j]
            
            results.append(result)
    
    return pd.DataFrame(results)


def combine_training_and_projected_zci(
    training_zci_df: pd.DataFrame,
    projected_zci_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine ZCI DataFrames for training and projected sites.
    
    Parameters:
    -----------
    training_zci_df : pd.DataFrame
        ZCI for training sites
    projected_zci_df : pd.DataFrame
        ZCI for projected sites
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all sites
    """
    # Handle empty DataFrames
    if training_zci_df.empty and projected_zci_df.empty:
        return pd.DataFrame()
    
    if training_zci_df.empty:
        return projected_zci_df.copy()
    
    if projected_zci_df.empty:
        return training_zci_df.copy()
    
    # Get common columns, ensuring 'Site' is included if present
    common_cols = list(set(training_zci_df.columns) & set(projected_zci_df.columns))
    
    # Ensure critical columns are included
    essential_cols = ['Site', 'Cluster', 'Site_Type', 'ZCI', 'Pollution_Score']
    for col in essential_cols:
        if col in training_zci_df.columns and col not in common_cols:
            common_cols.append(col)
    
    combined = pd.concat([
        training_zci_df[common_cols],
        projected_zci_df[[c for c in common_cols if c in projected_zci_df.columns]]
    ], ignore_index=True)
    
    return combined


def calculate_zci_with_least_pollution_threshold(
    pca_coordinates: pd.DataFrame,
    raw_data: pd.DataFrame,
    cluster_id: int,
    cluster_column: str = 'clusters',
    pollution_column: str = 'Pollution_Score',
    least_pollution_threshold: float = 25.0,
    pc_columns: Optional[List[str]] = None,
    normalize_pcs: bool = True
) -> Tuple[pd.Series, np.ndarray]:
    """
    Calculate ZCI using least-polluted sites as reference within a cluster.
    
    Instead of using pre-defined reference sites, this method selects the
    least-polluted sites in the cluster (bottom X percentile) and uses their
    centroid as the reference point for ZCI calculation.
    
    Parameters:
    -----------
    pca_coordinates : pd.DataFrame
        PCA coordinates for sites in this cluster (sites x PCs)
    raw_data : pd.DataFrame
        Raw data with pollution scores for each site
    cluster_id : int
        Current cluster ID (for identifying sites in this cluster)
    cluster_column : str
        Column name for cluster labels
    pollution_column : str
        Column name for pollution scores
    least_pollution_threshold : float
        Percentile threshold for least-polluted sites (default: 25.0).
        E.g., 25.0 means use the bottom 25% least-polluted sites.
    pc_columns : list, optional
        Which PC columns to use for distance calculation. If None, uses all.
    normalize_pcs : bool
        If True (default), z-score normalize PC coordinates before calculating
        distances. This ensures all PCs contribute equally.
        
    Returns:
    --------
    tuple
        (ZCI Series indexed by site, reference_point array)
    """
    # Identify sites in this cluster that are in both pca_coordinates and raw_data
    common_sites = pca_coordinates.index.intersection(raw_data.index)
    
    if len(common_sites) == 0:
        raise ValueError(f"No common sites found between PCA coordinates and raw_data")
    
    # Get pollution scores for sites in this cluster
    site_pollution = raw_data.loc[common_sites, pollution_column].copy()
    
    # Remove NaN pollution scores
    site_pollution = site_pollution.dropna()
    
    if len(site_pollution) == 0:
        raise ValueError(f"No valid pollution scores found for cluster {cluster_id}")
    
    # Find the threshold pollution score (bottom X percentile)
    threshold_score = np.percentile(site_pollution, least_pollution_threshold)
    
    # Get sites below this threshold
    least_polluted_sites = site_pollution[site_pollution <= threshold_score].index.tolist()
    
    if not least_polluted_sites:
        raise ValueError(f"No sites found below {least_pollution_threshold}% threshold")
    
    # Select PC columns to use
    if pc_columns is None:
        pc_columns = [col for col in pca_coordinates.columns if col.startswith('PC')]
    
    # Validate that selected PCs exist
    available_pcs = [col for col in pc_columns if col in pca_coordinates.columns]
    if not available_pcs:
        raise ValueError(f"No PC columns found in coordinates")
    
    # Get working coordinates for selected PCs
    working_coords = pca_coordinates.loc[common_sites, available_pcs].copy()
    
    # Apply z-score normalization if requested
    if normalize_pcs:
        for col in working_coords.columns:
            mean_val = working_coords[col].mean()
            std_val = working_coords[col].std()
            if std_val > 0:
                working_coords[col] = (working_coords[col] - mean_val) / std_val
            else:
                working_coords[col] = 0  # Constant column
    
    # Calculate centroid of least-polluted sites
    least_polluted_coords = working_coords.loc[least_polluted_sites]
    reference_point = least_polluted_coords.mean().values
    
    # Calculate Euclidean distance from reference point for all sites
    all_coords = working_coords.values
    distances = np.sqrt(((all_coords - reference_point) ** 2).sum(axis=1))
    
    zci = pd.Series(distances, index=common_sites, name='ZCI')
    
    return zci, reference_point


def calculate_zci_all_clusters_with_threshold(
    pca_results: Dict,
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    pollution_column: str = 'Pollution_Score',
    least_pollution_threshold: float = 25.0,
    pc_planes: Optional[Dict[int, Tuple[int, int]]] = None,
    normalize_pcs: bool = True
) -> pd.DataFrame:
    """
    Calculate ZCI for all sites using least-pollution-based reference points.
    
    This method allows flexible definition of reference sites per cluster,
    and flexible selection of which PCs to use for distance calculation.
    
    Parameters:
    -----------
    pca_results : dict
        Output from fit_cluster_pcas()
    raw_data : pd.DataFrame
        Raw data with cluster labels and pollution scores
    cluster_column : str
        Column name for cluster labels
    pollution_column : str
        Column name for pollution scores
    least_pollution_threshold : float
        Percentile threshold (0-100) for least-polluted sites.
        Default 25.0 uses bottom 25% of sites by pollution score.
    pc_planes : dict, optional
        {cluster_id: (pc_x, pc_y, ...)} specifying which PCs to use for
        distance calculation. If None, uses all available PCs.
    normalize_pcs : bool
        If True (default), z-score normalize PC coordinates before calculating
        distances. This ensures all PCs contribute equally.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Site, Cluster, ZCI, Pollution_Score, and PC columns
    """
    pca_coordinates = pca_results['pca_coordinates']
    n_components = pca_results['n_components']
    
    results = []
    reference_points = {}
    
    # Get reference sites from raw_data if available
    ref_column = 'if_ref'  # Default reference column
    has_ref_column = ref_column in raw_data.columns
    
    for cluster in sorted(pca_coordinates.keys()):
        coords = pca_coordinates[cluster]
        n_pcs = n_components[cluster]
        
        # Determine which PCs to use for this cluster
        if pc_planes and cluster in pc_planes:
            pc_indices = pc_planes[cluster]
            pc_cols = [f'PC{i}' for i in pc_indices]
        else:
            pc_cols = [f'PC{i+1}' for i in range(n_pcs)]
        
        # Calculate ZCI with least-pollution threshold
        zci, ref_point = calculate_zci_with_least_pollution_threshold(
            pca_coordinates=coords,
            raw_data=raw_data,
            cluster_id=cluster,
            cluster_column=cluster_column,
            pollution_column=pollution_column,
            least_pollution_threshold=least_pollution_threshold,
            pc_columns=pc_cols,
            normalize_pcs=normalize_pcs
        )
        reference_points[cluster] = ref_point
        
        # Get reference sites for Site_Type
        if has_ref_column:
            ref_mask = raw_data[ref_column].isin([True, 1])
            cluster_mask = raw_data[cluster_column] == cluster
            ref_sites = set(raw_data[cluster_mask & ref_mask].index.tolist())
        else:
            ref_sites = set()
        
        # Build results for each site
        for site in coords.index:
            if site not in raw_data.index:
                continue
                
            pollution = raw_data.loc[site, pollution_column] if pollution_column in raw_data.columns else np.nan
            
            # Determine site type
            if site in ref_sites:
                site_type = 'Reference'
            else:
                site_type = 'Training'
            
            result = {
                'Site': site,
                'Cluster': cluster,
                'Site_Type': site_type,
                'ZCI': zci[site],
                'Pollution_Score': pollution,
                'N_PCs_Used': len(pc_cols)
            }
            
            # Add PC coordinates for selected PCs
            for pc_col in pc_cols:
                if pc_col in coords.columns:
                    result[pc_col] = coords.loc[site, pc_col]
            
            results.append(result)
    
    zci_df = pd.DataFrame(results)
    
    return zci_df
