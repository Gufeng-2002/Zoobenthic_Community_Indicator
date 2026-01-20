"""
Data transformation utilities for contamination assessment.

This module provides functions for transforming and standardizing pollution variables
before PCA analysis, including log transformations and hierarchical clustering for
variable ordering.
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def log_z_score_transform(df, log_columns=None):
    """
    Apply log-transformation followed by Z-score standardization.
    
    This transformation is commonly used for pollution variables to handle:
    - Skewed distributions (via log transformation)
    - Different scales across variables (via z-score standardization)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing pollution variables to transform
    log_columns : list or None, default=None
        Columns to apply log transformation to. If None, applies to all columns.
        Note: As and Bi are excluded from log transformation by default.
    
    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame with same shape and index as input
    
    Notes:
    ------
    - Uses np.log1p() which computes log(1 + x) to handle zero values
    - Excludes 'As' and 'Bi' from log transformation due to their distribution characteristics
    - Z-score standardization is applied to all columns after log transformation
    
    Example:
    --------
    >>> pollution_data = df[("chemical", "raw")]
    >>> transformed = log_z_score_transform(pollution_data)
    """
    if log_columns is None:
        log_columns = df.columns
    
    df_transformed = df.copy()
    
    # Apply log transformation (excluding As and Bi)
    for col in log_columns:
        if col not in ["As", "Bi"]:
            df_transformed[col] = np.log1p(df_transformed[col])
    
    # Apply Z-score standardization to all columns
    scaler = StandardScaler()
    df_transformed[df_transformed.columns] = scaler.fit_transform(df_transformed)
    
    return df_transformed


def get_clustered_variable_order(df):
    """
    Order pollution variables by similarity using hierarchical clustering.
    
    Variables with high correlation are placed closer together, which is useful
    for visualizations like heatmaps and ridge plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing pollution variables
    
    Returns:
    --------
    pd.io.formats.style.Styler
        Styled correlation matrix with variables reordered by similarity.
        The underlying DataFrame can be accessed via .data attribute.
    
    Notes:
    ------
    - Uses correlation distance: distance = 1 - |correlation|
    - Performs hierarchical clustering with average linkage
    - Returns a styled correlation matrix for visualization
    
    Example:
    --------
    >>> ordered_corr = get_clustered_variable_order(transformed_pollution)
    >>> ordered_variables = ordered_corr.columns.tolist()
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
    
    # Reorder the correlation matrix
    ordered_variables = corr_matrix.columns[cluster_order].tolist()
    clustered_corr = corr_matrix.loc[ordered_variables, ordered_variables]
    
    # Style the reordered correlation matrix for display
    clustered_styled_corr = (
        clustered_corr.style
        .background_gradient(cmap='RdBu_r', vmin=-1, vmax=1)
        .format(precision=2)
        .set_caption("Clustered Correlation Matrix")
    )
    
    return clustered_styled_corr
