"""
Pollution scoring utilities for contamination assessment.

This module provides functions for computing pollution scores from principal
components and merging them into data structures.
"""

import numpy as np
import pandas as pd


def compute_pollution_scores(pca_scores, weights=None):
    """
    Compute pollution scores by weighted sum of PC scores.
    
    Each site receives a single pollution score calculated as a weighted sum
    of its PC scores. Weights can be customized to emphasize PCs with greater
    biological or interpretive relevance.
    
    Parameters:
    -----------
    pca_scores : pd.DataFrame
        DataFrame of PCA scores (sites × PCs), e.g., columns=['PC1', 'PC2', ...]
    weights : dict, list, array, or None, default=None
        Weights for each PC. Can be:
        - dict: {'PC1': 1.0, 'PC2': 1.5, ...} - maps PC names to weights
        - list/array: [1.0, 1.5, ...] - weights in same order as pca_scores columns
        - None: uses default weights with PC3 weighted 2x for biological relevance
    
    Returns:
    --------
    pd.DataFrame
        Single-column DataFrame with pollution scores (index preserved from pca_scores)
    
    Notes:
    ------
    Default weights are based on RDA validation results:
    - PC1, PC2: weight = 1 (general contamination patterns)
    - PC3: weight = 2 (high biological impact on species composition)
    - PC4: weight = 0 (low relevance)
    - PC5, PC6: weight = 1 (moderate relevance)
    
    Examples:
    ---------
    # Use default weights
    >>> scores = compute_pollution_scores(PC_scores)
    
    # Custom weights as dict
    >>> custom_weights = {'PC1': 1.5, 'PC2': 1.0, 'PC3': 2.0}
    >>> scores = compute_pollution_scores(PC_scores, custom_weights)
    
    # Equal weights as list
    >>> equal_weights = [1.0] * 6
    >>> scores = compute_pollution_scores(PC_scores, equal_weights)
    """
    # Handle different weight input types
    if weights is None:
        # Default weights based on RDA analysis
        pc_weights = {
            'PC1': 1,     # Highest variance, general contamination
            'PC2': 1,     # Second highest variance
            'PC3': 2,     # High biological impact (from RDA results)
            'PC4': 0,     # Low relevance
            'PC5': 1,     # Moderate relevance
            'PC6': 1      # Moderate relevance
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
    
    # Compute weighted sum: pollution_score = sum(PC_i * weight_i)
    pollution_scores = pca_scores.values @ weight_array
    
    # Return as DataFrame with preserved index
    pollution_scores_df = pd.DataFrame(
        pollution_scores, 
        index=pca_scores.index, 
        columns=["Pollution_Score"]
    )
    
    return pollution_scores_df


def merge_pollution_scores_into_data(pollution_scores, raw_data, multiindex_data, 
                                     multiindex_levels=('pollution', 'weighted', 'SumRel'),
                                     pc_scores=None,
                                     save_individual_pcs=False):
    """
    Merge pollution scores into both raw (flat) and multi-index dataframes.
    
    Adds the pollution scores as a new column to both data structures, maintaining
    proper formatting for each. Optionally also saves individual pollution PCs.
    
    Parameters:
    -----------
    pollution_scores : pd.DataFrame
        Pollution scores DataFrame (single column, typically from compute_pollution_scores)
    raw_data : pd.DataFrame
        Flat DataFrame without multi-index columns
    multiindex_data : pd.DataFrame
        DataFrame with multi-index columns (e.g., 3-level column index)
    multiindex_levels : tuple, default=('pollution', 'weighted', 'SumRel')
        Three-level tuple for creating multi-index column name in multiindex_data
    pc_scores : pd.DataFrame, optional
        Individual PC scores (sites x PCs) to save if save_individual_pcs=True
    save_individual_pcs : bool, default=False
        If True, also save individual PC scores as separate columns
    
    Returns:
    --------
    raw_data_merged : pd.DataFrame
        Raw dataframe with pollution scores added as 'Pollution_Score' column,
        and optionally individual PCs as 'Pollution_PC1', 'Pollution_PC2', etc.
    multiindex_data_merged : pd.DataFrame
        Multi-index dataframe with pollution scores added under specified levels,
        and optionally individual PCs under ('pollution', 'pc', 'PC1'), etc.
    
    Notes:
    ------
    - Creates new dataframes rather than modifying in-place
    - Preserves all existing columns in both data structures
    - Multi-index column structure is maintained in multiindex_data
    
    Example:
    --------
    >>> pollution_scores = compute_pollution_scores(PC_scores)
    >>> raw_data, data = merge_pollution_scores_into_data(
    ...     pollution_scores, raw_data, data,
    ...     multiindex_levels=('pollution', 'weighted', 'SumRel'),
    ...     pc_scores=PC_scores,
    ...     save_individual_pcs=True
    ... )
    >>> print(data[('pollution', 'weighted', 'SumRel')].head())
    >>> print(raw_data[['Pollution_PC1', 'Pollution_PC2']].head())
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
    
    # Optionally save individual PC scores
    if save_individual_pcs and pc_scores is not None:
        # Add individual PCs to raw_data with Pollution_PC prefix
        pc_cols_renamed = {col: f'Pollution_{col}' for col in pc_scores.columns}
        pc_scores_renamed = pc_scores.rename(columns=pc_cols_renamed)
        raw_data_merged = pd.concat([raw_data_merged, pc_scores_renamed], axis=1)
        
        # Add individual PCs to multiindex_data under ('pollution', 'pc', 'PCx')
        for pc_name in pc_scores.columns:
            pc_multiindex_col = pd.MultiIndex.from_tuples([('pollution', 'pc', pc_name)])
            pc_df = pd.DataFrame(
                pc_scores[pc_name].values,
                index=pc_scores.index,
                columns=pc_multiindex_col
            )
            multiindex_data_merged = concat_blocks([multiindex_data_merged, pc_df])
    
    return raw_data_merged, multiindex_data_merged
