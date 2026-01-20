"""
Box-Cox Transformation and ANOVA Analysis Module

This module provides utilities to perform Box-Cox transformations followed by
one-way ANOVA to test for significant differences across clusters/groups.

The Box-Cox transformation is used to stabilize variance and normalize the
distribution of variables before performing ANOVA, improving the validity
of the statistical test.

Key Functions:
-------------
- boxcox_transform_and_anova_env: Box-Cox + ANOVA for environmental variables
- boxcox_transform_and_anova_taxa: Box-Cox + ANOVA for taxa variables
- perform_boxcox_anova_analysis: Combined analysis for both env and taxa

References:
----------
Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations.
Journal of the Royal Statistical Society: Series B, 26(2), 211-252.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import boxcox, f_oneway, shapiro, levene
from scipy.special import inv_boxcox
import warnings


def _apply_boxcox_transformation(
    data: pd.Series,
    feature_name: str
) -> Dict[str, Any]:
    """
    Apply Box-Cox transformation to a single feature.
    
    Parameters
    ----------
    data : pd.Series
        Feature data to transform
    feature_name : str
        Name of the feature for logging
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'transformed': Transformed data (pd.Series)
        - 'lambda': Optimal lambda parameter
        - 'shift': Shift value applied before transformation
        - 'interpretation': Interpretation of lambda value
        - 'success': Boolean indicating if transformation succeeded
    """
    try:
        # Box-Cox requires all positive values
        min_val = data.min()
        
        if min_val <= 0:
            shift_value = abs(min_val) + 1e-6  # Small epsilon for positivity
            shifted_data = data + shift_value
        else:
            shifted_data = data
            shift_value = 0.0
        
        # Apply Box-Cox transformation
        transformed_values, optimal_lambda = boxcox(shifted_data)
        
        # Interpret lambda value
        if abs(optimal_lambda - 1.0) < 0.1:
            interpretation = "≈ No transformation needed"
        elif abs(optimal_lambda - 0.5) < 0.1:
            interpretation = "≈ Square root transformation"
        elif abs(optimal_lambda) < 0.1:
            interpretation = "≈ Log transformation"
        elif abs(optimal_lambda - (-1.0)) < 0.1:
            interpretation = "≈ Reciprocal transformation"
        else:
            interpretation = "Custom power transformation"
        
        # Create Series with same index
        transformed_series = pd.Series(transformed_values, index=data.index)
        
        return {
            'transformed': transformed_series,
            'lambda': optimal_lambda,
            'shift': shift_value,
            'interpretation': interpretation,
            'success': True
        }
        
    except Exception as e:
        warnings.warn(f"Box-Cox failed for {feature_name}: {str(e)}. Using log transformation.")
        
        # Fallback to log transformation
        if shift_value > 0:
            transformed_values = np.log(shifted_data)
        else:
            transformed_values = np.log1p(data)
        
        transformed_series = pd.Series(transformed_values, index=data.index)
        
        return {
            'transformed': transformed_series,
            'lambda': 0.0,  # log transformation
            'shift': shift_value,
            'interpretation': "Log transformation (fallback)",
            'success': False
        }


def boxcox_transform_and_anova_env(
    raw_data: pd.DataFrame,
    cluster_column: str,
    env_variables: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform Box-Cox transformation and one-way ANOVA on environmental variables
    across clusters.
    
    This function:
    1. Applies Box-Cox transformation to each environmental variable
    2. Tests ANOVA assumptions (normality and homogeneity of variance)
    3. Performs one-way ANOVA to test for differences across clusters
    4. Returns detailed results and transformed data
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels.
        Should only include sites with cluster assignments (reference sites).
    cluster_column : str
        Name of the column containing cluster labels
    env_variables : list of str
        List of environmental variable column names to analyze
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'anova_results': pd.DataFrame with ANOVA results for each variable
        - 'transformation_info': dict with Box-Cox parameters for each variable
        - 'transformed_data': dict with Box-Cox transformed Series for each variable
        - 'n_significant': int, number of variables with significant differences
        
    Examples
    --------
    >>> results = boxcox_transform_and_anova_env(
    ...     raw_data=ref_sites_data,
    ...     cluster_column='clusters',
    ...     env_variables=['Depth', 'Temperature', 'DO'],
    ...     verbose=True
    ... )
    >>> print(results['anova_results'])
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("BOX-COX TRANSFORMATION + ANOVA: Environmental Variables")
        print("=" * 80)
    
    # Filter to sites with cluster labels (non-NaN)
    clustered_data = raw_data[raw_data[cluster_column].notna()].copy()
    n_clusters = int(clustered_data[cluster_column].nunique())
    
    if verbose:
        print(f"\nAnalyzing {len(env_variables)} environmental variables")
        print(f"Sites: {len(clustered_data)} (across {n_clusters} clusters)")
        print(f"Cluster distribution: {clustered_data[cluster_column].value_counts().sort_index().to_dict()}")
    
    # Store results
    transformation_info = {}
    transformed_data = {}
    anova_results = []
    
    for var in env_variables:
        if var not in clustered_data.columns:
            warnings.warn(f"Variable '{var}' not found in data. Skipping.")
            continue
        
        var_data = clustered_data[var].dropna()
        
        if len(var_data) == 0:
            warnings.warn(f"Variable '{var}' has no valid data. Skipping.")
            continue
        
        # Apply Box-Cox transformation
        transform_result = _apply_boxcox_transformation(var_data, var)
        transformation_info[var] = transform_result
        transformed_data[var] = transform_result['transformed']
        
        # Get transformed data for each cluster
        groups_data = []
        for cluster_id in sorted(clustered_data[cluster_column].unique()):
            cluster_mask = clustered_data[cluster_column] == cluster_id
            cluster_indices = clustered_data[cluster_mask].index
            # Get transformed values for this cluster
            group_values = transform_result['transformed'].loc[
                transform_result['transformed'].index.isin(cluster_indices)
            ]
            groups_data.append(group_values.values)
        
        # Test ANOVA assumptions
        # 1. Normality (Shapiro-Wilk on each group)
        normality_pass = True
        for i, group_data in enumerate(groups_data):
            if len(group_data) >= 3:
                _, shapiro_p = shapiro(group_data)
                if shapiro_p < 0.05:
                    normality_pass = False
        
        # 2. Homogeneity of variance (Levene's test)
        levene_stat, levene_p = levene(*groups_data)
        variance_pass = levene_p >= 0.05
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups_data)
        
        # Determine significance
        is_significant = p_value < 0.05
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # Store results
        anova_results.append({
            'Variable': var,
            'Lambda': transform_result['lambda'],
            'F-statistic': f_stat,
            'p-value': p_value,
            'Significance': significance,
            'Significant': is_significant,
            'Normality': '✓' if normality_pass else '✗',
            'Homogeneity': '✓' if variance_pass else '✗',
            'Interpretation': transform_result['interpretation']
        })
    
    # Create DataFrame
    anova_df = pd.DataFrame(anova_results).sort_values('p-value')
    n_significant = anova_df['Significant'].sum()
    
    if verbose:
        print("\n" + "-" * 80)
        print("ANOVA Results (sorted by p-value):")
        print("-" * 80)
        print(anova_df[['Variable', 'F-statistic', 'p-value', 'Significance', 
                       'Lambda', 'Normality', 'Homogeneity']].to_string(index=False))
        print("-" * 80)
        print(f"\nSummary:")
        print(f"  - Total variables: {len(anova_df)}")
        print(f"  - Significant differences (p < 0.05): {n_significant}")
        print(f"  - Non-significant: {len(anova_df) - n_significant}")
        print("=" * 80)
    
    return {
        'anova_results': anova_df,
        'transformation_info': transformation_info,
        'transformed_data': transformed_data,
        'n_significant': n_significant
    }


def boxcox_transform_and_anova_taxa(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str,
    taxa_transformation: str = 'hellinger',
    top_n_taxa: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform Box-Cox transformation and one-way ANOVA on taxa variables
    across clusters.
    
    This function:
    1. Extracts taxa data and applies user-specified transformation (hellinger/chord/octave)
    2. Applies Box-Cox transformation to each taxon
    3. Tests ANOVA assumptions
    4. Performs one-way ANOVA to identify taxa with significant differences across clusters
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing cluster labels
    multiindex_data : pd.DataFrame
        Multi-index DataFrame containing taxa data
    cluster_column : str
        Name of the column in raw_data containing cluster labels
    taxa_transformation : str, default='hellinger'
        Initial transformation to apply to taxa data before Box-Cox.
        Options: 'hellinger', 'chord', 'octave', 'none'
    top_n_taxa : int, optional
        If specified, only analyze the top N most abundant taxa
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'anova_results': pd.DataFrame with ANOVA results for each taxon
        - 'transformation_info': dict with Box-Cox parameters for each taxon
        - 'transformed_data': dict with Box-Cox transformed Series for each taxon
        - 'n_significant': int, number of taxa with significant differences
        - 'significant_taxa': list of taxa names that are significant
        
    Examples
    --------
    >>> results = boxcox_transform_and_anova_taxa(
    ...     raw_data=ref_sites_data,
    ...     multiindex_data=data,
    ...     cluster_column='clusters',
    ...     taxa_transformation='hellinger',
    ...     verbose=True
    ... )
    >>> print(results['significant_taxa'])
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("BOX-COX TRANSFORMATION + ANOVA: Taxa Variables")
        print("=" * 80)
    
    # Import transformation functions
    from .hierarchical_clustering import hellinger_transform, chord_transform, octave_transform
    
    # Filter to clustered sites
    clustered_mask = raw_data[cluster_column].notna()
    clustered_indices = raw_data[clustered_mask].index
    n_clusters = int(raw_data.loc[clustered_mask, cluster_column].nunique())
    
    # Extract taxa data
    taxa_level_mask = multiindex_data.columns.get_level_values(0) == 'taxa'
    taxa_data = multiindex_data.loc[clustered_indices, taxa_level_mask].copy()
    
    # Simplify column names
    taxa_data.columns = taxa_data.columns.get_level_values(-1)
    
    # Filter to top N taxa if specified
    if top_n_taxa is not None:
        taxa_abundances = taxa_data.sum(axis=0).sort_values(ascending=False)
        top_taxa = taxa_abundances.head(top_n_taxa).index
        taxa_data = taxa_data[top_taxa]
        if verbose:
            print(f"\nAnalyzing top {top_n_taxa} most abundant taxa (out of {len(taxa_abundances)} total)")
    
    # Apply initial transformation
    if taxa_transformation.lower() == 'hellinger':
        taxa_transformed = hellinger_transform(taxa_data)
        transformation_name = "Hellinger"
    elif taxa_transformation.lower() == 'chord':
        taxa_transformed = chord_transform(taxa_data)
        transformation_name = "Chord"
    elif taxa_transformation.lower() == 'octave':
        taxa_transformed = octave_transform(taxa_data)
        transformation_name = "Octave"
    elif taxa_transformation.lower() == 'none':
        taxa_transformed = taxa_data
        transformation_name = "None (raw)"
    else:
        raise ValueError(f"Unknown transformation: {taxa_transformation}")
    
    if verbose:
        print(f"\nInitial transformation: {transformation_name}")
        print(f"Sites: {len(taxa_transformed)} (across {n_clusters} clusters)")
        print(f"Taxa analyzed: {taxa_transformed.shape[1]}")
        cluster_dist = raw_data.loc[clustered_indices, cluster_column].value_counts().sort_index()
        print(f"Cluster distribution: {cluster_dist.to_dict()}")
    
    # Get cluster labels
    cluster_labels = raw_data.loc[taxa_transformed.index, cluster_column]
    
    # Store results
    transformation_info = {}
    transformed_data = {}
    anova_results = []
    
    for taxon in taxa_transformed.columns:
        taxon_data = taxa_transformed[taxon]
        
        # Apply Box-Cox transformation
        transform_result = _apply_boxcox_transformation(taxon_data, taxon)
        transformation_info[taxon] = transform_result
        transformed_data[taxon] = transform_result['transformed']
        
        # Get transformed data for each cluster
        groups_data = []
        for cluster_id in sorted(cluster_labels.unique()):
            cluster_mask = cluster_labels == cluster_id
            group_values = transform_result['transformed'][cluster_mask].values
            groups_data.append(group_values)
        
        # Test ANOVA assumptions
        # 1. Normality (Shapiro-Wilk on each group)
        normality_pass = True
        for group_data in groups_data:
            if len(group_data) >= 3:
                _, shapiro_p = shapiro(group_data)
                if shapiro_p < 0.05:
                    normality_pass = False
        
        # 2. Homogeneity of variance (Levene's test)
        levene_stat, levene_p = levene(*groups_data)
        variance_pass = levene_p >= 0.05
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups_data)
        
        # Determine significance
        is_significant = p_value < 0.05
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # Store results
        anova_results.append({
            'Taxon': taxon,
            'Lambda': transform_result['lambda'],
            'F-statistic': f_stat,
            'p-value': p_value,
            'Significance': significance,
            'Significant': is_significant,
            'Normality': '✓' if normality_pass else '✗',
            'Homogeneity': '✓' if variance_pass else '✗'
        })
    
    # Create DataFrame
    anova_df = pd.DataFrame(anova_results).sort_values('p-value')
    n_significant = anova_df['Significant'].sum()
    significant_taxa = anova_df[anova_df['Significant']]['Taxon'].tolist()
    
    if verbose:
        print("\n" + "-" * 80)
        print("ANOVA Results (sorted by p-value):")
        print("-" * 80)
        print(anova_df[['Taxon', 'F-statistic', 'p-value', 'Significance', 
                       'Lambda', 'Normality', 'Homogeneity']].to_string(index=False))
        print("-" * 80)
        print(f"\nSummary:")
        print(f"  - Total taxa analyzed: {len(anova_df)}")
        print(f"  - Significant differences (p < 0.05): {n_significant} ({100*n_significant/len(anova_df):.1f}%)")
        print(f"  - Non-significant: {len(anova_df) - n_significant} ({100*(len(anova_df)-n_significant)/len(anova_df):.1f}%)")
        if n_significant > 0:
            print(f"\nSignificant taxa: {', '.join(significant_taxa[:10])}")
            if n_significant > 10:
                print(f"  ... and {n_significant - 10} more")
        print("=" * 80)
    
    return {
        'anova_results': anova_df,
        'transformation_info': transformation_info,
        'transformed_data': transformed_data,
        'n_significant': n_significant,
        'significant_taxa': significant_taxa,
        'taxa_transformation_used': transformation_name
    }


def perform_boxcox_anova_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str,
    env_variables: List[str],
    taxa_transformation: str = 'hellinger',
    top_n_taxa: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive Box-Cox + ANOVA analysis on both environmental
    and taxa variables across clusters.
    
    This is a convenience function that combines environmental and taxa
    ANOVA analyses in one call.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels
    multiindex_data : pd.DataFrame
        Multi-index DataFrame containing taxa data
    cluster_column : str
        Name of the column containing cluster labels
    env_variables : list of str
        List of environmental variable names to analyze
    taxa_transformation : str, default='hellinger'
        Initial transformation for taxa data ('hellinger', 'chord', 'octave', 'none')
    top_n_taxa : int, optional
        Number of most abundant taxa to analyze
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'env_results': Results from environmental variable ANOVA
        - 'taxa_results': Results from taxa variable ANOVA
        - 'summary': dict with overall summary statistics
        
    Examples
    --------
    >>> results = perform_boxcox_anova_analysis(
    ...     raw_data=data,
    ...     multiindex_data=multiindex_data,
    ...     cluster_column='clusters',
    ...     env_variables=['Depth', 'Temperature', 'DO', 'MPS', 'LOI', 'Velocity'],
    ...     taxa_transformation='hellinger',
    ...     top_n_taxa=16,
    ...     verbose=True
    ... )
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BOX-COX + ANOVA ANALYSIS")
        print("=" * 80)
        print("\nThis analysis performs Box-Cox transformation followed by one-way ANOVA")
        print("to identify environmental variables and taxa that differ significantly")
        print("across clusters.")
        print("=" * 80)
    
    # Analyze environmental variables
    env_results = boxcox_transform_and_anova_env(
        raw_data=raw_data,
        cluster_column=cluster_column,
        env_variables=env_variables,
        verbose=verbose
    )
    
    # Analyze taxa variables
    taxa_results = boxcox_transform_and_anova_taxa(
        raw_data=raw_data,
        multiindex_data=multiindex_data,
        cluster_column=cluster_column,
        taxa_transformation=taxa_transformation,
        top_n_taxa=top_n_taxa,
        verbose=verbose
    )
    
    # Overall summary
    summary = {
        'env_variables_analyzed': len(env_results['anova_results']),
        'env_significant': env_results['n_significant'],
        'taxa_analyzed': len(taxa_results['anova_results']),
        'taxa_significant': taxa_results['n_significant'],
        'taxa_transformation': taxa_results['taxa_transformation_used']
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"\nEnvironmental Variables:")
        print(f"  - Analyzed: {summary['env_variables_analyzed']}")
        print(f"  - Significant: {summary['env_significant']}")
        print(f"\nTaxa Variables:")
        print(f"  - Transformation: {summary['taxa_transformation']}")
        print(f"  - Analyzed: {summary['taxa_analyzed']}")
        print(f"  - Significant: {summary['taxa_significant']}")
        print("=" * 80)
    
    return {
        'env_results': env_results,
        'taxa_results': taxa_results,
        'summary': summary
    }
