"""
LDA Pipeline for Cluster Classification

This module provides a comprehensive pipeline for Linear Discriminant Analysis (LDA)
to classify reference sites into habitat-based clusters, following the workflow from
notebook 04.

The pipeline includes:
1. Basic LDA training with full model evaluation
2. Monte Carlo Cross-Validation for robustness assessment
3. Variable importance using Wilks' Lambda

Key Functions:
-------------
- perform_lda_analysis: Complete LDA training and evaluation
- perform_monte_carlo_cv: Monte Carlo cross-validation with comprehensive metrics
- plot_lda_cv_results: Visualize cross-validation results
- compute_lda_variable_importance: Calculate variable importance using Wilks' Lambda

Author: Developed for St. Clair-Detroit River System Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
from zci.taxa_assemblage_in_refs.hierarchical_clustering import (
    hellinger_transform,
    chord_transform,
    octave_transform,
    octave_to_relative_abundance,
)


def perform_lda_analysis(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    env_variables: Optional[List[str]] = None,
    log_env: bool = False,
    standardize_env: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform complete LDA analysis on reference sites.
    
    Trains a Linear Discriminant Analysis model to classify sites into clusters
    based on habitat characteristics. Provides comprehensive evaluation including
    confusion matrix, classification report, and discriminant function coefficients.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels.
        Must include reference sites (with non-NaN cluster labels).
    multiindex_data : pd.DataFrame
        Multi-index DataFrame (not used directly, but kept for consistency).
    cluster_column : str, default='clusters'
        Name of column in raw_data containing cluster labels.
    env_variables : list of str, optional
        List of environmental variable names to use.
        If None, uses default set.
    log_env: bool, default=False.
        Whether to log-transform environmental variables before standardization.
    standardize_env : bool, default=True
        Whether to standardize (z-score) environmental variables.
    verbose : bool, default=True
        Whether to print progress and results.
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'lda_model': Fitted LDA object
        - 'env_data': Environmental data used (standardized if requested)
        - 'cluster_labels': True cluster labels
        - 'predictions': Predicted cluster labels
        - 'accuracy': Overall classification accuracy
        - 'confusion_matrix': Confusion matrix
        - 'classification_report': Classification report (dict)
        - 'discriminant_coefficients': LDA discriminant function coefficients
        - 'explained_variance_ratio': Variance explained by each discriminant function
        - 'scaler': StandardScaler object (if standardize_env=True, else None)
        
    Examples
    --------
    >>> results = perform_lda_analysis(
    ...     raw_data=ref_sites_data,
    ...     multiindex_data=multiindex_data,
    ...     cluster_column='clusters',
    ...     standardize_env=True,
    ...     verbose=True
    ... )
    >>> print(f"Accuracy: {results['accuracy']:.2%}")
    >>> print(results['discriminant_coefficients'])
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("LINEAR DISCRIMINANT ANALYSIS (LDA) PIPELINE")
        print("=" * 80)
        print("\nClassifying reference sites into habitat-based clusters")
        print("=" * 80)
    
    # ========================================================================
    # STEP 1: Filter reference sites
    # ========================================================================
    if verbose:
        print("\nStep 1: Filtering reference sites...")
    
    ref_mask = raw_data[cluster_column].notna()
    n_ref_sites = ref_mask.sum()
    
    if n_ref_sites == 0:
        raise ValueError("No reference sites found (all cluster labels are NaN)")
    
    ref_raw_data = raw_data[ref_mask].copy()
    
    if verbose:
        print(f"  - Total sites: {len(raw_data)}")
        print(f"  - Reference sites: {n_ref_sites}")
        print(f"  - Clusters: {int(ref_raw_data[cluster_column].nunique())}")
    
    # ========================================================================
    # STEP 2: Prepare environmental data
    # ========================================================================
    if verbose:
        print("\nStep 2: Preparing environmental data...")
    
    # Use default environmental variables if not specified
    if env_variables is None:
        env_variables = [
            'Measured Depth (m)',
            'Velocity  at bottom (m/sec)_Imputed',
            'Water DO Bottom (mg/L)',
            'Temperature (oC)',
            'MPS (Phi)',
            'LOI (%)'
        ]
    
    # Extract environmental data
    env_data = ref_raw_data[env_variables].copy()
    
    # Check for missing values
    if env_data.isna().any().any():
        warnings.warn("Environmental data contains missing values. Dropping rows with NaN.")
        valid_idx = env_data.dropna().index
        env_data = env_data.loc[valid_idx]
        ref_raw_data = ref_raw_data.loc[valid_idx]
    
    # Log-transformation if needed
    env_data = env_data.copy()
    # check the minimum values in each column
    for col in env_data.columns:
        min_val = env_data[col].min()
        if min_val < 0:
            env_data[col] = env_data[col] + abs(min_val) + 1e-6  # shift to make all values non-negative
    if log_env:
        if verbose:
            print("  - Log-transforming environmental variables")
        env_data = np.log1p(env_data + 1e-6)  # small constant to avoid log(0)
    
    # Standardize if requested
    scaler = None
    if standardize_env:
        if verbose:
            print("  - Standardizing environmental variables (z-score)")
        scaler = StandardScaler()
        env_data_values = scaler.fit_transform(env_data)
        env_data = pd.DataFrame(
            env_data_values,
            columns=env_data.columns,
            index=env_data.index
        )
    
    if verbose:
        print(f"  - Environmental variables: {len(env_variables)}")
        print(f"  - Shape: {env_data.shape}")
    
    # ========================================================================
    # STEP 3: Extract cluster labels
    # ========================================================================
    cluster_labels = ref_raw_data[cluster_column].values
    
    if verbose:
        print("\nStep 3: Cluster distribution...")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"  - Cluster {int(cluster)}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # ========================================================================
    # STEP 4: Train LDA model
    # ========================================================================
    if verbose:
        print("\nStep 4: Training LDA model...")
    
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(env_data.values, cluster_labels)
    
    if verbose:
        print(f"  ✓ LDA model fitted successfully")
        print(f"  - Number of discriminant functions: {lda_model.coef_.shape[0]}")
    
    # ========================================================================
    # STEP 5: Make predictions and evaluate
    # ========================================================================
    if verbose:
        print("\nStep 5: Evaluating model performance...")
    
    predictions = lda_model.predict(env_data.values)
    accuracy = accuracy_score(cluster_labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(cluster_labels, predictions)
    
    # Classification report
    cluster_names = [f"Cluster {int(i)}" for i in sorted(np.unique(cluster_labels))]
    class_report_dict = classification_report(
        cluster_labels, 
        predictions,
        target_names=cluster_names,
        output_dict=True,
        zero_division=0
    )
    
    if verbose:
        print(f"  - Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n{'='*80}")
        print("CONFUSION MATRIX")
        print(f"{'='*80}")
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {name}" for name in cluster_names],
            columns=[f"Pred {name}" for name in cluster_names]
        )
        print(cm_df)
        
        print(f"\n{'='*80}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*80}")
        print(classification_report(
            cluster_labels,
            predictions,
            target_names=cluster_names,
            zero_division=0
        ))
    
    # ========================================================================
    # STEP 6: Extract discriminant function coefficients
    # ========================================================================
    if verbose:
        print(f"{'='*80}")
        print("DISCRIMINANT FUNCTION COEFFICIENTS")
        print(f"{'='*80}")
    
    discriminant_coefs = pd.DataFrame(
        lda_model.coef_,
        columns=env_variables,
        index=[f'LD{i+1}' for i in range(lda_model.coef_.shape[0])]
    )
    
    explained_var_ratio = lda_model.explained_variance_ratio_
    
    if verbose:
        print("\nLinear Discriminant Function Coefficients:")
        print(discriminant_coefs.round(4))
        
        print("\nExplained Variance Ratio:")
        for i, var_ratio in enumerate(explained_var_ratio):
            print(f"  LD{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
        print(f"  Total: {explained_var_ratio.sum():.4f} ({explained_var_ratio.sum()*100:.2f}%)")
    
    # ========================================================================
    # Return results
    # ========================================================================
    return {
        'lda_model': lda_model,
        'env_data': env_data,
        'cluster_labels': cluster_labels,
        'predictions': predictions,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'confusion_matrix_df': cm_df,
        'classification_report': class_report_dict,
        'classification_report_text': classification_report(
            cluster_labels, predictions, target_names=cluster_names, zero_division=0
        ),
        'discriminant_coefficients': discriminant_coefs,
        'explained_variance_ratio': explained_var_ratio,
        'scaler': scaler,
        'env_variables': env_variables,
        'cluster_names': cluster_names,
        'ref_raw_data': ref_raw_data
    }


def compute_lda_variable_importance(
    lda_results: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute variable importance for LDA using Wilks' Lambda stepwise approach.
    
    Calculates:
    - Wilks' Lambda for the full model
    - Delta Wilks' Lambda for each variable (importance when removed)
    - F-statistics and p-values for each variable
    - Overall model significance (Bartlett's approximation)
    
    Parameters
    ----------
    lda_results : Dict[str, Any]
        Results from perform_lda_analysis() containing:
        - env_data: Environmental data
        - cluster_labels: Cluster labels
        - env_variables: Variable names
        - discriminant_coefficients: LDA coefficients
    verbose : bool, default=True
        Whether to print results tables
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'axes_summary': DataFrame with LDA axes explained variance
        - 'coefficients_table': DataFrame with transposed discriminant coefficients
        - 'variable_importance': DataFrame with variable importance statistics
        - 'wilks_lambda_full': Full model Wilks' Lambda
        - 'overall_significance': Dict with chi-square, df, and p-value
        - 'significance_dict': Dict mapping variable names to significance levels
    
    Examples
    --------
    >>> importance = compute_lda_variable_importance(lda_results, verbose=True)
    >>> print(importance['variable_importance'])
    >>> print(f"Full model Wilks' Lambda: {importance['wilks_lambda_full']:.4f}")
    """
    
    # Extract data from lda_results
    env_data = lda_results['env_data']
    cluster_labels = lda_results['cluster_labels']
    env_vars = lda_results['env_variables']
    discriminant_coefs = lda_results['discriminant_coefficients']
    explained_var = lda_results['explained_variance_ratio']
    
    # Convert to numpy arrays if needed
    env_X_array = env_data.values if hasattr(env_data, 'values') else np.array(env_data)
    cluster_y = cluster_labels.values if hasattr(cluster_labels, 'values') else np.array(cluster_labels)
    
    n_samples = len(cluster_y)
    n_groups = len(np.unique(cluster_y))
    n_vars = len(env_vars)
    
    # ========================================================================
    # TABLE 1: LDA Axes Summary
    # ========================================================================
    cumulative_var = np.cumsum(explained_var)
    axes_summary = pd.DataFrame({
        'Axis': [f'LD{i+1}' for i in range(len(explained_var))],
        'Explained (%)': explained_var * 100,
        'Cumulative (%)': cumulative_var * 100
    })
    axes_summary = axes_summary.set_index('Axis')
    
    # ========================================================================
    # TABLE 2: Coefficients (transposed)
    # ========================================================================
    coef_table = discriminant_coefs.T.copy()
    coef_table.index.name = 'Environmental Variable'
    
    # ========================================================================
    # TABLE 3: Variable Importance using Wilks' Lambda
    # ========================================================================
    
    def calc_wilks_lambda(env_X, y):
        """Calculate Wilks' Lambda for LDA."""
        groups = np.unique(y)
        n_features = env_X.shape[1]
        
        # Within-group scatter matrix
        W = np.zeros((n_features, n_features))
        for g in groups:
            X_g = env_X[y == g]
            mean_g = X_g.mean(axis=0)
            W += (X_g - mean_g).T @ (X_g - mean_g)
        
        # Total scatter matrix
        mean_total = env_X.mean(axis=0)
        T = (env_X - mean_total).T @ (env_X - mean_total)
        
        # Wilks' Lambda = |W| / |T|
        det_W = np.linalg.det(W)
        det_T = np.linalg.det(T)
        
        if det_T == 0:
            return 1.0
        return det_W / det_T
    
    # Calculate full model Wilks' Lambda
    wilks_full = calc_wilks_lambda(env_X_array, cluster_y)
    
    # Calculate importance by removing each variable
    importance_data = []
    significance_dict = {}
    
    for i, var in enumerate(env_vars):
        # Create reduced dataset without this variable
        mask = np.ones(n_vars, dtype=bool)
        mask[i] = False
        env_X_reduced = env_X_array[:, mask]
        
        # Calculate Wilks' Lambda without this variable
        wilks_reduced = calc_wilks_lambda(env_X_reduced, cluster_y)
        
        # Delta Lambda = Reduced - Full (larger delta means variable is more important)
        delta_lambda = wilks_reduced - wilks_full
        
        # F-statistic approximation
        df1 = n_groups - 1
        df2 = n_samples - n_groups - n_vars + 1
        
        if wilks_full > 0 and delta_lambda > 0:
            f_stat = (delta_lambda / wilks_full) * (df2 / df1)
        else:
            f_stat = 0
        
        p_value = 1 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 and df2 > 0 else 1.0
        
        # Get coefficients
        ld1_coef = discriminant_coefs.iloc[0, i] if len(discriminant_coefs) > 0 else 0
        ld2_coef = discriminant_coefs.iloc[1, i] if len(discriminant_coefs) > 1 else 0
        
        # Significance stars
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        significance_dict[var] = {'p_value': p_value, 'significance': sig}
        
        importance_data.append({
            'Environmental Variable': var,
            "Delta Wilks' Lambda": delta_lambda,
            'F-statistic': f_stat,
            'p-value': p_value,
            'Significance': sig,
            'LD1 Coefficient': ld1_coef,
            'LD2 Coefficient': ld2_coef
        })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('F-statistic', ascending=False)
    importance_df = importance_df.set_index('Environmental Variable')
    
    # Overall model significance (Bartlett's approximation)
    n = n_samples
    p = n_vars
    g = n_groups
    chi_sq = -(n - 1 - (p + g)/2) * np.log(wilks_full)
    df_chi = p * (g - 1)
    overall_p = 1 - stats.chi2.cdf(chi_sq, df_chi)
    
    overall_significance = {
        'chi_square': chi_sq,
        'df': df_chi,
        'p_value': overall_p
    }
    
    # Print tables if verbose
    if verbose:
        print("=" * 80)
        print("TABLE 1: LDA Discriminant Axes Summary")
        print("=" * 80)
        print(axes_summary.round(4).to_string())
        
        print("\n" + "=" * 80)
        print("TABLE 2: LDA Discriminant Function Coefficients (Standardized)")
        print("=" * 80)
        print(coef_table.round(4).to_string())
        
        print("\n" + "=" * 80)
        print("TABLE 3: Environmental Variable Importance in LDA")
        print("=" * 80)
        print(importance_df.round(4).to_string())
        
        print("\nSignificance codes: '***' p<0.001, '**' p<0.01, '*' p<0.05, 'ns' not significant")
        print(f"\nFull model Wilks' Lambda: {wilks_full:.4f}")
        
        print(f"\nOverall model significance:")
        print(f"  Chi-square: {chi_sq:.4f}")
        print(f"  df: {df_chi}")
        print(f"  p-value: {overall_p:.6f}")
    
    return {
        'axes_summary': axes_summary,
        'coefficients_table': coef_table,
        'variable_importance': importance_df,
        'wilks_lambda_full': wilks_full,
        'overall_significance': overall_significance,
        'significance_dict': significance_dict
    }


def create_lda_excel_table(
    lda_results: Dict[str, Any],
    lda_importance: Dict[str, Any],
    raw_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    env_variables: Optional[List[str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Create publication-ready LDA variable importance table for Excel export.
    
    This function creates a clean table format suitable for saving to Excel,
    matching the publication style with:
    - Habitat variable names in first column
    - Significance level (p < 0.001***, p < 0.01**, etc.)
    - Mean (± 1SE) for each cluster
    - Rows sorted by p-value
    
    Parameters
    ----------
    lda_results : Dict[str, Any]
        Results from perform_lda_analysis() containing env_data, cluster_labels
    lda_importance : Dict[str, Any]
        Results from compute_lda_variable_importance() containing significance_dict
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels
    cluster_column : str, default='clusters'
        Column name containing cluster labels
    env_variables : list of str, optional
        List of environmental variable names. If None, uses all from lda_results.
    verbose : bool, default=False
        Whether to print progress
    
    Returns
    -------
    pd.DataFrame
        Clean table ready for Excel export with columns:
        Habitat variables | Significance level | Cluster C1 | Cluster C2 | ...
    """
    # Get data from lda_results
    cluster_labels = lda_results['cluster_labels']
    
    if env_variables is None:
        env_variables = lda_results['env_variables']
    
    # Filter to sites with cluster labels
    clustered_mask = raw_data[cluster_column].notna()
    clustered_data = raw_data[clustered_mask].copy()
    clusters = sorted(clustered_data[cluster_column].unique())
    
    # Calculate sample size per cluster
    sample_sizes = {}
    for cluster in clusters:
        n = (clustered_data[cluster_column] == cluster).sum()
        sample_sizes[int(cluster)] = n
    
    # Get significance info from lda_importance
    significance_dict = lda_importance['significance_dict']
    
    results = []
    for var in env_variables:
        if var not in clustered_data.columns:
            continue
        
        row = {'Habitat variables': var}
        
        # Get p-value and significance from lda_importance
        if var in significance_dict:
            p_val = significance_dict[var]['p_value']
            sig = significance_dict[var]['significance']
            
            # Format significance level like image
            if p_val < 0.001:
                row['Significance level'] = 'p < 0.001***'
            elif p_val < 0.01:
                row['Significance level'] = 'p < 0.01**'
            elif p_val < 0.05:
                row['Significance level'] = 'p < 0.05*'
            else:
                row['Significance level'] = 'p > 0.05'
            
            row['p_value_numeric'] = p_val  # For sorting
        else:
            row['Significance level'] = 'p > 0.05'
            row['p_value_numeric'] = 1.0
        
        # Calculate Mean ± 1SE for each cluster
        for cluster in clusters:
            cluster_int = int(cluster)
            cluster_mask = clustered_data[cluster_column] == cluster
            cluster_values = clustered_data.loc[cluster_mask, var].dropna()
            
            if len(cluster_values) > 0:
                mean_val = cluster_values.mean()
                n_cluster = len(cluster_values)
                se_val = cluster_values.std() / np.sqrt(n_cluster) if n_cluster > 1 else 0
                row[f'Cluster C{cluster_int + 1}'] = f"{mean_val:.2f} ± {se_val:.2f}"
            else:
                row[f'Cluster C{cluster_int + 1}'] = ''
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Sort by p-value (ascending, most significant first)
    df = df.sort_values('p_value_numeric', ascending=True)
    df = df.drop(columns=['p_value_numeric'])
    df = df.reset_index(drop=True)
    
    # Reorder columns
    cluster_cols = [f'Cluster C{int(c) + 1}' for c in clusters]
    col_order = ['Habitat variables', 'Significance level'] + cluster_cols
    df = df[col_order]
    
    # Add empty row as separator
    empty_row = {col: '' for col in df.columns}
    
    # Add sample size row - put sample sizes in the cluster columns
    sample_row = {col: '' for col in df.columns}
    sample_row['Habitat variables'] = 'Sample size (n)'
    for cluster in clusters:
        cluster_int = int(cluster)
        sample_row[f'Cluster C{cluster_int + 1}'] = sample_sizes[cluster_int]
    
    # Add significance legend row
    legend_row = {col: '' for col in df.columns}
    legend_row['Habitat variables'] = 'Significance: *** p<0.001, ** p<0.01, * p<0.05'
    
    # Append footer
    df = pd.concat([
        df,
        pd.DataFrame([empty_row]),
        pd.DataFrame([sample_row]),
        pd.DataFrame([empty_row]),
        pd.DataFrame([legend_row])
    ], ignore_index=True)
    
    if verbose:
        data_rows = df.iloc[:-4]
        n_sig = data_rows['Significance level'].apply(lambda x: '*' in str(x)).sum()
        print(f"  - LDA Variable Importance table: {len(data_rows)} variables, {n_sig} significant")
    
    return df


def perform_monte_carlo_cv(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    env_variables: Optional[List[str]] = None,
    log_env: bool = False,
    standardize_env: bool = True,
    n_iterations: int = 1000,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform Monte Carlo Cross-Validation for LDA.
    
    Uses repeated random subsampling to assess model robustness and generalization.
    Each iteration randomly splits data into train/test sets while maintaining
    cluster proportions (stratification).
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing environmental variables and cluster labels.
    multiindex_data : pd.DataFrame
        Multi-index DataFrame (not used directly).
    cluster_column : str, default='clusters'
        Name of column containing cluster labels.
    env_variables : list of str, optional
        List of environmental variable names to use.
    log_env: bool, default=False.
        Whether to log-transform environmental variables before standardization.
    standardize_env : bool, default=True
        Whether to standardize environmental variables.
    n_iterations : int, default=1000
        Number of random train/test splits.
    test_size : float, default=0.2
        Proportion of data for test set (e.g., 0.2 = 20%).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results.
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'iteration_accuracies': List of accuracies for each iteration
        - 'mean_accuracy': Mean accuracy across iterations
        - 'std_accuracy': Standard deviation of accuracies
        - 'median_accuracy': Median accuracy
        - 'min_accuracy': Minimum accuracy
        - 'max_accuracy': Maximum accuracy
        - 'aggregate_confusion_matrix': Combined confusion matrix from all iterations
        - 'avg_confusion_matrix_normalized': Average normalized confusion matrix
        - 'avg_classification_report': Average metrics with std dev
        - 'iteration_cluster_distributions': Cluster distributions in test sets
        - 'all_predictions': All predictions from all iterations
        - 'all_true_labels': All true labels from all iterations
        
    Examples
    --------
    >>> cv_results = perform_monte_carlo_cv(
    ...     raw_data=ref_sites_data,
    ...     multiindex_data=multiindex_data,
    ...     n_iterations=1000,
    ...     test_size=0.2,
    ...     verbose=True
    ... )
    >>> print(f"Mean accuracy: {cv_results['mean_accuracy']:.2%} ± {cv_results['std_accuracy']:.2%}")
    """
    
    if verbose:
        print("\n" + "=" * 80)
        print("MONTE CARLO CROSS-VALIDATION (Repeated Random Subsampling)")
        print("=" * 80)
    
    # ========================================================================
    # STEP 1: Prepare data
    # ========================================================================
    ref_mask = raw_data[cluster_column].notna()
    ref_raw_data = raw_data[ref_mask].copy()
    
    if env_variables is None:
        env_variables = [
            'Measured Depth (m)',
            'Velocity  at bottom (m/sec)_Imputed',
            'Water DO Bottom (mg/L)',
            'Temperature (oC)',
            'MPS (Phi)',
            'LOI (%)'
        ]
    
    env_data = ref_raw_data[env_variables].copy()
    
    if env_data.isna().any().any():
        valid_idx = env_data.dropna().index
        env_data = env_data.loc[valid_idx]
        ref_raw_data = ref_raw_data.loc[valid_idx]
    
    if log_env:
        # check the minimum values in each column
        for col in env_data.columns:
            min_val = env_data[col].min()
            if min_val < 0:
                env_data[col] = env_data[col] + abs(min_val) + 1e-6  # shift to make all values non-negative
        env_data = np.log1p(env_data + 1e-6)  # small constant to avoid log(0)

    if standardize_env:
        scaler = StandardScaler()
        env_data_values = scaler.fit_transform(env_data)
        X = env_data_values
    else:
        X = env_data.values
    
    y = ref_raw_data[cluster_column].values
    cluster_names = [f"Cluster {int(i)}" for i in sorted(np.unique(y))]
    
    if verbose:
        print(f"\nTotal samples: {len(y)}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of iterations: {n_iterations}")
        print(f"Test size per iteration: {test_size*100:.0f}% (~{int(len(y)*test_size)} samples)")
        print(f"Train size per iteration: {(1-test_size)*100:.0f}% (~{int(len(y)*(1-test_size))} samples)")
        
        cluster_counts = pd.Series(y).value_counts().sort_index()
        print("\nCluster distribution:")
        for cluster, count in cluster_counts.items():
            test_samples = int(count * test_size)
            print(f"  Cluster {int(cluster)}: {count} total → ~{test_samples} in test set per iteration")
    
    # ========================================================================
    # STEP 2: Perform Monte Carlo CV
    # ========================================================================
    sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size, random_state=random_state)
    
    iteration_accuracies = []
    iteration_predictions = []
    iteration_true_labels = []
    iteration_confusion_matrices = []
    iteration_classification_reports = []
    iteration_cluster_distributions = []
    
    if verbose:
        print("\n" + "=" * 80)
        print("Running Monte Carlo Cross-Validation...")
        print("=" * 80)
    
    for iter_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train LDA
        lda_mc = LinearDiscriminantAnalysis()
        lda_mc.fit(X_train, y_train)
        
        # Predict
        y_pred = lda_mc.predict(X_test)
        
        # Calculate accuracy
        iter_accuracy = accuracy_score(y_test, y_pred)
        iteration_accuracies.append(iter_accuracy)
        
        # Store predictions and labels
        iteration_predictions.extend(y_pred)
        iteration_true_labels.extend(y_test)
        
        # Confusion matrix
        iter_cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
        iteration_confusion_matrices.append(iter_cm)
        
        # Classification report
        iter_report = classification_report(
            y_test, y_pred,
            target_names=cluster_names,
            output_dict=True,
            zero_division=0
        )
        iteration_classification_reports.append(iter_report)
        
        # Track cluster distribution in test set
        test_cluster_dist = pd.Series(y_test).value_counts().sort_index().to_dict()
        iteration_cluster_distributions.append(test_cluster_dist)
        
        # Progress update
        if verbose and (iter_idx % 100 == 0 or iter_idx == 1):
            print(f"Iteration {iter_idx:4d}/{n_iterations}: Accuracy = {iter_accuracy:.4f}, "
                  f"Test clusters: {test_cluster_dist}")
    
    # ========================================================================
    # STEP 3: Calculate statistics
    # ========================================================================
    mean_accuracy = np.mean(iteration_accuracies)
    std_accuracy = np.std(iteration_accuracies)
    median_accuracy = np.median(iteration_accuracies)
    min_accuracy = np.min(iteration_accuracies)
    max_accuracy = np.max(iteration_accuracies)
    
    if verbose:
        print("\n" + "=" * 80)
        print("MONTE CARLO CV RESULTS")
        print("=" * 80)
        print(f"Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
        print(f"Std Dev: {std_accuracy:.4f}")
        print(f"Median Accuracy: {median_accuracy:.4f} ({median_accuracy*100:.2f}%)")
        print(f"Min Accuracy: {min_accuracy:.4f} ({min_accuracy*100:.2f}%)")
        print(f"Max Accuracy: {max_accuracy:.4f} ({max_accuracy*100:.2f}%)")
        print(f"95% CI: [{mean_accuracy - 1.96*std_accuracy:.4f}, {mean_accuracy + 1.96*std_accuracy:.4f}]")
    
    # Aggregate confusion matrix
    aggregate_cm = confusion_matrix(
        iteration_true_labels, 
        iteration_predictions,
        labels=sorted(np.unique(y))
    )
    
    # Average normalized confusion matrix
    avg_cm_normalized = np.mean(
        [cm / cm.sum() for cm in iteration_confusion_matrices if cm.sum() > 0],
        axis=0
    )
    
    # Average classification metrics
    avg_metrics = {}
    for cluster in cluster_names + ['weighted avg']:
        avg_metrics[cluster] = {'precision': [], 'recall': [], 'f1-score': []}
    
    for report in iteration_classification_reports:
        for cluster in cluster_names:
            if cluster in report:
                avg_metrics[cluster]['precision'].append(report[cluster]['precision'])
                avg_metrics[cluster]['recall'].append(report[cluster]['recall'])
                avg_metrics[cluster]['f1-score'].append(report[cluster]['f1-score'])
        
        if 'weighted avg' in report:
            avg_metrics['weighted avg']['precision'].append(report['weighted avg']['precision'])
            avg_metrics['weighted avg']['recall'].append(report['weighted avg']['recall'])
            avg_metrics['weighted avg']['f1-score'].append(report['weighted avg']['f1-score'])
    
    avg_classification_report = {}
    for cluster, metrics in avg_metrics.items():
        avg_classification_report[cluster] = {
            'precision_mean': np.mean(metrics['precision']) if metrics['precision'] else 0,
            'precision_std': np.std(metrics['precision']) if metrics['precision'] else 0,
            'recall_mean': np.mean(metrics['recall']) if metrics['recall'] else 0,
            'recall_std': np.std(metrics['recall']) if metrics['recall'] else 0,
            'f1-score_mean': np.mean(metrics['f1-score']) if metrics['f1-score'] else 0,
            'f1-score_std': np.std(metrics['f1-score']) if metrics['f1-score'] else 0,
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("AVERAGE CLASSIFICATION METRICS ACROSS ITERATIONS")
        print("=" * 80)
        print(f"{'Class':<15} {'Precision':>15} {'Recall':>15} {'F1-Score':>15}")
        print("-" * 80)
        for cluster in cluster_names:
            prec = avg_classification_report[cluster]
            print(f"{cluster:<15} {prec['precision_mean']:.3f}±{prec['precision_std']:.3f}  "
                  f"{prec['recall_mean']:.3f}±{prec['recall_std']:.3f}  "
                  f"{prec['f1-score_mean']:.3f}±{prec['f1-score_std']:.3f}")
        print("-" * 80)
        wtavg = avg_classification_report['weighted avg']
        print(f"{'Weighted Avg':<15} {wtavg['precision_mean']:.3f}±{wtavg['precision_std']:.3f}  "
              f"{wtavg['recall_mean']:.3f}±{wtavg['recall_std']:.3f}  "
              f"{wtavg['f1-score_mean']:.3f}±{wtavg['f1-score_std']:.3f}")
    
    # ========================================================================
    # Return results
    # ========================================================================
    return {
        'iteration_accuracies': iteration_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'median_accuracy': median_accuracy,
        'min_accuracy': min_accuracy,
        'max_accuracy': max_accuracy,
        'aggregate_confusion_matrix': aggregate_cm,
        'avg_confusion_matrix_normalized': avg_cm_normalized,
        'avg_classification_report': avg_classification_report,
        'iteration_cluster_distributions': iteration_cluster_distributions,
        'all_predictions': iteration_predictions,
        'all_true_labels': iteration_true_labels,
        'cluster_names': cluster_names,
        'n_iterations': n_iterations,
        'test_size': test_size
    }


def plot_lda_cv_results(
    cv_results: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 12),
    dpi: int = 150
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive visualization of Monte Carlo CV results.
    
    Generates a 4-panel figure showing:
    1. Distribution of accuracies across iterations
    2. Accuracy trend across iterations
    3. Aggregate confusion matrix (counts)
    4. Average normalized confusion matrix (percentages)
    
    Parameters
    ----------
    cv_results : dict
        Results from perform_monte_carlo_cv().
    figsize : tuple, default (14, 12)
        Figure size (width, height) in inches.
    dpi : int, default 150
        Figure resolution.
        
    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    
    iteration_accuracies = cv_results['iteration_accuracies']
    mean_accuracy = cv_results['mean_accuracy']
    std_accuracy = cv_results['std_accuracy']
    median_accuracy = cv_results['median_accuracy']
    aggregate_cm = cv_results['aggregate_confusion_matrix']
    avg_cm_normalized = cv_results['avg_confusion_matrix_normalized']
    cluster_names = cv_results['cluster_names']
    n_iterations = cv_results['n_iterations']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    # Panel 1: Distribution of accuracies
    axes[0, 0].hist(iteration_accuracies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mean_accuracy, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_accuracy:.3f}')
    axes[0, 0].axvline(median_accuracy, color='green', linestyle=':', linewidth=2,
                       label=f'Median: {median_accuracy:.3f}')
    axes[0, 0].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Distribution of Accuracies\n({n_iterations} iterations)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Panel 2: Accuracy across iterations
    axes[0, 1].plot(range(1, n_iterations + 1), iteration_accuracies,
                    alpha=0.4, color='steelblue', linewidth=0.8)
    axes[0, 1].axhline(mean_accuracy, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_accuracy:.3f}')
    axes[0, 1].fill_between(range(1, n_iterations + 1),
                            mean_accuracy - std_accuracy,
                            mean_accuracy + std_accuracy,
                            alpha=0.2, color='red', label=f'±1 Std Dev')
    axes[0, 1].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Accuracy Across Iterations', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3)
    
    # Panel 3: Aggregate confusion matrix
    cm_df = pd.DataFrame(
        aggregate_cm,
        index=[f"True {name}" for name in cluster_names],
        columns=[f"Pred {name}" for name in cluster_names]
    )
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, ax=axes[1, 0])
    axes[1, 0].set_title(f'Aggregate Confusion Matrix\n({n_iterations} iterations combined)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Panel 4: Average normalized confusion matrix
    avg_cm_norm_df = pd.DataFrame(
        avg_cm_normalized * 100,
        index=[f"True {name}" for name in cluster_names],
        columns=[f"Pred {name}" for name in cluster_names]
    )
    sns.heatmap(avg_cm_norm_df, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Percentage (%)'}, ax=axes[1, 1])
    axes[1, 1].set_title('Average Normalized Confusion Matrix\n(% of total predictions)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


def compare_lda_rda_axes(
    lda_results: Dict[str, Any],
    rda_results: Dict[str, Any],
    env_variables: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare LDA discriminant axes with RDA ordination axes.
    
    Computes correlations between LDA discriminant function coefficients and
    RDA environmental variable loadings to assess how well LDA classification
    axes match the gradients identified by RDA ordination.
    
    Parameters
    ----------
    lda_results : dict
        Results from perform_lda_analysis() containing 'discriminant_coefficients'
    rda_results : dict
        Results from perform_rda_analysis() containing 'rda_model'
    env_variables : list of str
        List of environmental variable names used in both analyses
    verbose : bool, default=True
        Whether to print detailed comparison results
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'lda_coefficients': DataFrame of LDA discriminant coefficients
        - 'rda_loadings': DataFrame of RDA environmental variable loadings
        - 'correlation_matrix': Correlation matrix between LDA and RDA axes
        - 'best_matches': DataFrame showing best RDA match for each LDA axis
        - 'cosine_similarity': Cosine similarity matrix between axes
    
    Examples
    --------
    >>> comparison = compare_lda_rda_axes(lda_results, rda_results, env_vars)
    >>> print(comparison['best_matches'])
    """
    if verbose:
        print("\n" + "="*80)
        print("COMPARING LDA DISCRIMINANT AXES WITH RDA ORDINATION AXES")
        print("="*80)
    
    # Extract LDA discriminant coefficients
    lda_coef = lda_results['discriminant_coefficients'].copy()
    if verbose:
        print(f"\nLDA Discriminant Functions: {lda_coef.shape[0]}")
        print(f"Environmental Variables: {lda_coef.shape[1]}")
    
    # Extract RDA environmental variable loadings
    rda_model = rda_results['rda_model']
    # Get biplot scores for environmental variables (these are the loadings)
    rda_biplot = rda_model.get_biplot_scores()
    
    # Create DataFrame with RDA loadings (select first few axes)
    n_rda_axes = min(rda_biplot.shape[1], 4)  # Use first 4 RDA axes
    rda_loadings = rda_biplot.iloc[:, :n_rda_axes].T  # Transpose to match LDA format (axes × variables)
    
    if verbose:
        print(f"RDA Axes: {rda_loadings.shape[0]}")
        print(f"\nLDA Explained Variance:")
        for i, var in enumerate(lda_results['explained_variance_ratio'], 1):
            print(f"  LD{i}: {var:.2%}")
        print(f"\nRDA Explained Variance:")
        # Get explained variance from RDA model
        explained_prop = rda_model.fit_.explained_proportion
        for i in range(n_rda_axes):
            var_explained = explained_prop.iloc[i]
            print(f"  RDA{i+1}: {var_explained:.2%}")
    
    # Compute Pearson correlation between LDA and RDA axes
    # Each row is an axis, each column is an environmental variable
    correlation_matrix = pd.DataFrame(
        index=lda_coef.index,
        columns=rda_loadings.index
    )
    
    for lda_axis in lda_coef.index:
        for rda_axis in rda_loadings.index:
            corr = np.corrcoef(
                lda_coef.loc[lda_axis].values,
                rda_loadings.loc[rda_axis].values
            )[0, 1]
            correlation_matrix.loc[lda_axis, rda_axis] = corr
    
    correlation_matrix = correlation_matrix.astype(float)
    
    # Compute cosine similarity (normalized dot product)
    cosine_similarity = pd.DataFrame(
        index=lda_coef.index,
        columns=rda_loadings.index
    )
    
    for lda_axis in lda_coef.index:
        lda_vec = lda_coef.loc[lda_axis].values
        lda_norm = np.linalg.norm(lda_vec)
        
        for rda_axis in rda_loadings.index:
            rda_vec = rda_loadings.loc[rda_axis].values
            rda_norm = np.linalg.norm(rda_vec)
            
            cos_sim = np.dot(lda_vec, rda_vec) / (lda_norm * rda_norm)
            cosine_similarity.loc[lda_axis, rda_axis] = cos_sim
    
    cosine_similarity = cosine_similarity.astype(float)
    
    # Find best matches
    best_matches = pd.DataFrame({
        'LDA_Axis': lda_coef.index,
        'Best_RDA_Match': correlation_matrix.abs().idxmax(axis=1).values,
        'Correlation': correlation_matrix.abs().max(axis=1).values,
        'Cosine_Similarity': [
            cosine_similarity.loc[lda_axis, correlation_matrix.abs().loc[lda_axis].idxmax()]
            for lda_axis in lda_coef.index
        ]
    })
    
    if verbose:
        print("\n" + "="*80)
        print("CORRELATION MATRIX (LDA vs RDA Axes)")
        print("="*80)
        print(correlation_matrix.round(3))
        
        print("\n" + "="*80)
        print("COSINE SIMILARITY MATRIX (LDA vs RDA Axes)")
        print("="*80)
        print(cosine_similarity.round(3))
        
        print("\n" + "="*80)
        print("BEST MATCHES")
        print("="*80)
        for idx, row in best_matches.iterrows():
            print(f"\n{row['LDA_Axis']} ↔ {row['Best_RDA_Match']}")
            print(f"  Correlation: {row['Correlation']:.3f}")
            print(f"  Cosine Similarity: {row['Cosine_Similarity']:.3f}")
            
            # Interpretation
            if abs(row['Correlation']) > 0.9:
                print(f"  → Very strong match (|r| > 0.9)")
            elif abs(row['Correlation']) > 0.7:
                print(f"  → Strong match (|r| > 0.7)")
            elif abs(row['Correlation']) > 0.5:
                print(f"  → Moderate match (|r| > 0.5)")
            else:
                print(f"  → Weak match (|r| < 0.5)")
    
    return {
        'lda_coefficients': lda_coef,
        'rda_loadings': rda_loadings,
        'correlation_matrix': correlation_matrix,
        'cosine_similarity': cosine_similarity,
        'best_matches': best_matches
    }


def predict_nonreference_sites(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    lda_results: Dict[str, Any],
    env_variables: List[str],
    ref_column: str = 'if_ref',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Predict cluster labels for non-reference sites using trained LDA model.
    
    Applies the trained LDA model from reference sites to classify non-reference
    (potentially polluted) sites into habitat-based clusters.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data containing both reference and non-reference sites
    multiindex_data : pd.DataFrame
        MultiIndex version with environmental variables
    lda_results : dict
        Results from perform_lda_analysis() containing trained model and scaler
    env_variables : list of str
        Environmental variable names used for prediction
    ref_column : str, default='if_ref'
        Column name indicating reference sites (True/False)
    verbose : bool, default=True
        Whether to print detailed prediction results
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': Array of predicted cluster labels for non-ref sites
        - 'probabilities': Prediction probabilities for each class
        - 'site_names': Index of non-reference sites
        - 'env_data': Environmental data used for prediction
        - 'n_sites': Number of non-reference sites classified
    
    Examples
    --------
    >>> predictions = predict_nonreference_sites(
    ...     raw_data, multiindex_data, lda_results, env_vars
    ... )
    >>> print(predictions['predictions'])
    """
    if verbose:
        print("\n" + "="*80)
        print("PREDICTING NON-REFERENCE SITE CLUSTERS")
        print("="*80)
    
    # Filter for non-reference sites
    if ref_column in raw_data.columns:
        non_ref_mask = raw_data[ref_column] == False
    else:
        raise ValueError(f"Column '{ref_column}' not found in raw_data")
    
    non_ref_sites = raw_data[non_ref_mask].copy()
    n_non_ref = len(non_ref_sites)
    
    if verbose:
        print(f"\nTotal sites: {len(raw_data)}")
        print(f"Reference sites: {(~non_ref_mask).sum()}")
        print(f"Non-reference sites: {n_non_ref}")
    
    if n_non_ref == 0:
        print("\n⚠ No non-reference sites found!")
        return {
            'predictions': np.array([]),
            'probabilities': np.array([]),
            'site_names': [],
            'env_data': pd.DataFrame(),
            'n_sites': 0
        }
    
    # Extract environmental data for non-reference sites
    non_ref_multiindex = multiindex_data.loc[non_ref_sites.index]
    
    # Get environmental variables from multiindex
    # Check if multiindex has the expected structure
    if isinstance(non_ref_multiindex.index, pd.MultiIndex):
        # MultiIndex format: extract from level 1
        env_data = []
        for var in env_variables:
            if var in non_ref_multiindex.index.get_level_values(1):
                var_data = non_ref_multiindex.xs(var, level=1).iloc[:, 0]
                env_data.append(var_data)
            else:
                raise ValueError(f"Environmental variable '{var}' not found in multiindex data")
        X_nonref = pd.DataFrame(env_data, index=env_variables).T
    else:
        # Single index format: environmental variables should be in columns
        X_nonref = non_ref_sites[env_variables].copy()

    
    # Check for missing values
    if X_nonref.isnull().any().any():
        print(f"\n⚠ Warning: {X_nonref.isnull().sum().sum()} missing values found in environmental data")
        print("Sites with missing values will be excluded from prediction")
        valid_mask = ~X_nonref.isnull().any(axis=1)
        X_nonref = X_nonref[valid_mask]
        non_ref_sites = non_ref_sites.loc[valid_mask]
    
    # Standardize using the same scaler from training
    if 'scaler' in lda_results and lda_results['scaler'] is not None:
        X_nonref_scaled = lda_results['scaler'].transform(X_nonref)
        if verbose:
            print(f"\n✓ Environmental data standardized using training scaler")
    else:
        X_nonref_scaled = X_nonref.values
        if verbose:
            print(f"\n⚠ No scaler found, using raw environmental data")
    
    # Predict using trained LDA model
    lda_model = lda_results['lda_model']
    predictions = lda_model.predict(X_nonref_scaled)
    probabilities = lda_model.predict_proba(X_nonref_scaled)
    
    if verbose:
        print(f"\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        print(f"\nPredicted cluster distribution:")
        unique, counts = np.unique(predictions, return_counts=True)
        for cluster, count in zip(unique, counts):
            percentage = count / len(predictions) * 100
            print(f"  Cluster {int(cluster)}: {count} sites ({percentage:.1f}%)")
        
        print(f"\nPrediction confidence:")
        max_probs = probabilities.max(axis=1)
        print(f"  Mean: {max_probs.mean():.2%}")
        print(f"  Median: {np.median(max_probs):.2%}")
        print(f"  Min: {max_probs.min():.2%}")
        print(f"  Max: {max_probs.max():.2%}")
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'site_names': X_nonref.index.tolist(),
        'env_data': X_nonref,
        'n_sites': len(predictions)
    }


def update_data_with_predictions(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    predictions: Dict[str, Any],
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Update raw_data and multiindex_data with predicted cluster labels for non-reference sites.
    
    Adds predicted cluster labels to the same cluster column that contains reference site
    clusters, creating a complete classification for all sites.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data to update with predictions
    multiindex_data : pd.DataFrame
        MultiIndex data to update with predictions
    predictions : dict
        Prediction results from predict_nonreference_sites()
    cluster_column : str, default='clusters'
        Name of the column to store cluster labels
    ref_column : str, default='if_ref'
        Column indicating reference sites
    verbose : bool, default=True
        Whether to print update information
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Updated raw_data and multiindex_data with predictions added
    
    Examples
    --------
    >>> updated_raw, updated_multi = update_data_with_predictions(
    ...     raw_data, multiindex_data, predictions
    ... )
    """
    if verbose:
        print("\n" + "="*80)
        print("UPDATING DATA WITH PREDICTIONS")
        print("="*80)
    
    # Create copies to avoid modifying originals
    raw_data_updated = raw_data.copy()
    multiindex_data_updated = multiindex_data.copy()
    
    # Initialize cluster column if it doesn't exist
    if cluster_column not in raw_data_updated.columns:
        raw_data_updated[cluster_column] = np.nan
        if verbose:
            print(f"\n✓ Created new column '{cluster_column}' in raw_data")
    
    # Get prediction site names and labels
    site_names = predictions['site_names']
    pred_labels = predictions['predictions']
    
    if len(site_names) == 0:
        print("\n⚠ No predictions to add (no non-reference sites)")
        return raw_data_updated, multiindex_data_updated
    
    # Update raw_data
    for site, label in zip(site_names, pred_labels):
        if site in raw_data_updated.index:
            raw_data_updated.loc[site, cluster_column] = label
    
    # Update multiindex_data
    # Add cluster information to the multiindex
    multi_index_cluster_column = ("Clusters", "Hierarchical", "clusters")
    for site in multiindex_data_updated.index.get_level_values(0).unique():
        if site in site_names:
            label = pred_labels[site_names.index(site)]
            # Update all rows for this site
            site_mask = multiindex_data_updated.index.get_level_values(0) == site
            if multi_index_cluster_column in multiindex_data_updated.columns:
                multiindex_data_updated.loc[site_mask, multi_index_cluster_column] = label
            else:
                # If column doesn't exist, add it
                multiindex_data_updated[multi_index_cluster_column] = np.nan
                multiindex_data_updated.loc[site_mask, multi_index_cluster_column] = label
    
    if verbose:
        print(f"\n✓ Updated {len(site_names)} non-reference sites with predicted clusters")
        
        # Show summary
        ref_clusters = raw_data_updated[raw_data_updated[ref_column] == True][cluster_column]
        nonref_clusters = raw_data_updated[raw_data_updated[ref_column] == False][cluster_column]
        
        print(f"\n" + "="*80)
        print("UPDATED CLUSTER DISTRIBUTION")
        print("="*80)
        
        print(f"\nReference sites ({len(ref_clusters)} total):")
        ref_dist = ref_clusters.value_counts().sort_index()
        for cluster, count in ref_dist.items():
            if not np.isnan(cluster):
                print(f"  Cluster {int(cluster)}: {count} sites")
        
        print(f"\nNon-reference sites ({len(nonref_clusters)} total):")
        nonref_dist = nonref_clusters.value_counts().sort_index()
        for cluster, count in nonref_dist.items():
            if not np.isnan(cluster):
                print(f"  Cluster {int(cluster)}: {count} sites")
        
        print(f"\nAll sites ({len(raw_data_updated)} total):")
        all_dist = raw_data_updated[cluster_column].value_counts().sort_index()
        for cluster, count in all_dist.items():
            if not np.isnan(cluster):
                percentage = count / len(raw_data_updated) * 100
                print(f"  Cluster {int(cluster)}: {count} sites ({percentage:.1f}%)")
    
    return raw_data_updated, multiindex_data_updated


def plot_lda_triplot(
    lda_results: Dict[str, Any],
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    env_variables: Optional[List[str]] = None,
    log_env: bool = False,
    standardize_env: bool = True,
    arrow_scale: Optional[float] = None,
    site_scale: Optional[float] = None,
    site_label_offset: float = 0.02,
    env_label_offset: float = 0.1,
    env_fontsize: int = 10,
    site_fontsize: int = 8,
    figsize: Tuple[float, float] = (14, 10),
    show_site_labels: bool = False,
    show_significance: bool = True,
    significance_dict: Optional[Dict[str, Dict]] = None,
    verbose: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create an LDA triplot showing site scores, cluster centroids, and habitat vectors.
    
    Similar to RDA triplot but for LDA space, visualizing:
    - All sites' LDA axis scores (colored by cluster)
    - Habitat variable vectors in LDA space
    - Cluster separation in discriminant space
    
    Parameters
    ----------
    lda_results : Dict[str, Any]
        Results from perform_lda_analysis() containing the LDA model
    raw_data : pd.DataFrame
        Raw data with cluster labels
    multiindex_data : pd.DataFrame
        MultiIndex data for extracting environmental variables
    cluster_column : str, default='clusters'
        Column name containing cluster labels
    env_variables : Optional[List[str]], default=None
        List of environmental variables. If None, uses all from multiindex_data
    log_env : bool, default=False
        Whether to log-transform environmental variables before standardization.
    standardize_env : bool, default=True
        Whether to standardize environmental variables
    arrow_scale : Optional[float], default=None
        Scaling factor for habitat vector arrows. If None, auto-computed.
    site_scale : Optional[float], default=None
        Scaling factor for site positions. If None, no scaling applied.
    site_label_offset : float, default=0.02
        Offset for site labels
    env_label_offset : float, default=0.1
        Offset for environmental variable labels (fraction of arrow length)
    env_fontsize : int, default=10
        Font size for environmental variable labels
    site_fontsize : int, default=8
        Font size for site labels
    figsize : Tuple[float, float], default=(14, 10)
        Figure size in inches
    show_site_labels : bool, default=False
        Whether to show site ID labels
    show_significance : bool, default=True
        Whether to show significance of environmental variables in LDA
    significance_dict : Optional[Dict[str, Dict]], default=None
        Dictionary mapping variable names to {'p_value': float, 'significance': str}.
        If provided, uses these values for line style (solid if p<0.05, dashed otherwise)
        and significance indicators. If None, computes pseudo-significance from loadings.
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
        
    Notes
    -----
    The LDA triplot visualizes:
    - Site scores: Transformed coordinates of all sites in LDA space
    - Habitat vectors: Direction and magnitude of environmental gradients
    - Cluster separation: Visual assessment of discriminant function performance
    
    Uses the same color scheme as RDA triplot:
    - Cluster 0: Red (#d62728)
    - Cluster 1: Blue (#1f77b4)
    - Cluster 2: Green (#2ca02c)
    """
    from zci.data_process import get_block
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING LDA TRIPLOT")
        print("="*80)
    
    # Extract LDA model and parameters
    lda_model = lda_results['lda_model']
    n_components = lda_model.n_components
    
    # Handle case where n_components is None (determined automatically)
    if n_components is None:
        # Get actual number of components from the scalings shape
        n_components = lda_model.scalings_.shape[1]
    
    if verbose:
        print(f"\nLDA components: {n_components}")
    
    # Prepare environmental data for all sites
    # Try to use get_block if multiindex_data has the proper structure,
    # otherwise extract from raw_data directly
    try:
        env_data = get_block(multiindex_data, block='environmental', subblock='raw')
        if env_variables is not None:
            env_data = env_data[env_variables]
    except (KeyError, AttributeError):
        # Fallback: extract from lda_results if available, or use provided env_variables
        if 'env_data' in lda_results:
            env_data = lda_results['env_data'].copy()
        elif env_variables is not None:
            # Extract directly from raw_data using env_variables
            env_data = raw_data[env_variables].copy()
        else:
            raise ValueError("Cannot extract environmental data. Please provide env_variables or use multiindex_data with proper structure.")
    
    # Log-transform if requested
    if log_env:
        for col in env_data.columns:
            min_val = env_data[col].min()
            if min_val <= 0:
                env_data[col] = env_data[col] + abs(min_val) + 1e-6 # shift values to non-negative range before log
        env_data = np.log1p(env_data)
        
    # Standardize if requested
    if standardize_env:
        scaler = StandardScaler()
        env_data_array = scaler.fit_transform(env_data)
        env_data_scaled = pd.DataFrame(
            env_data_array,
            columns=env_data.columns,
            index=env_data.index
        )
    else:
        env_data_scaled = env_data.copy()
    
    # Get cluster labels (handle NaN for non-reference sites)
    clusters = raw_data[cluster_column].copy()
    
    # Transform all sites to LDA space
    lda_scores = lda_model.transform(env_data_scaled)
    lda_scores_df = pd.DataFrame(
        lda_scores,
        index=env_data_scaled.index,
        columns=[f'LD{i+1}' for i in range(n_components)]
    )
    
    # Apply site scaling if provided
    if site_scale is not None:
        lda_scores_df = lda_scores_df * site_scale
    
    if verbose:
        print(f"Sites transformed to LDA space: {lda_scores_df.shape}")
    
    # Get habitat vectors (LDA scalings/coefficients)
    # These show how each environmental variable contributes to each discriminant function
    lda_scalings = lda_model.scalings_  # Shape: (n_features, n_components)
    
    # Auto-compute arrow_scale if not provided
    if arrow_scale is None:
        # Scale arrows to be visible relative to site scatter
        site_range = max(
            lda_scores_df['LD1'].max() - lda_scores_df['LD1'].min(),
            (lda_scores_df['LD2'].max() - lda_scores_df['LD2'].min()) if n_components > 1 else 1
        )
        max_scaling = np.abs(lda_scalings).max()
        arrow_scale = site_range * 0.3 / max_scaling if max_scaling > 0 else 1.0
    
    habitat_vectors = pd.DataFrame(
        lda_scalings * arrow_scale,
        index=env_data.columns,
        columns=[f'LD{i+1}' for i in range(n_components)]
    )
    
    if verbose:
        print(f"Habitat vectors computed: {habitat_vectors.shape}")
        print(f"Arrow scale: {arrow_scale:.3f}")
        print("\nTop habitat loadings on LD1:")
        print(habitat_vectors['LD1'].abs().sort_values(ascending=False).head())
    
    # Use provided significance_dict if available, otherwise compute pseudo-significance
    if significance_dict is not None:
        # Use actual significance from Wilks' Lambda tests
        env_significance_info = significance_dict
        if verbose:
            print("\nUsing provided significance values from Wilks' Lambda tests")
    else:
        # Compute pseudo-significance from loadings (fallback)
        env_importance = {}
        for i, var in enumerate(env_data.columns):
            # Use sum of squared loadings across all LDs as importance measure
            importance = np.sum(lda_scalings[i, :]**2)
            env_importance[var] = importance
        
        # Normalize to create pseudo p-values (lower = more important)
        max_importance = max(env_importance.values())
        env_significance_info = {}
        for k, v in env_importance.items():
            pseudo_p = 1 - (v / max_importance) if max_importance > 0 else 0.5
            if pseudo_p < 0.001:
                sig = '***'
            elif pseudo_p < 0.01:
                sig = '**'
            elif pseudo_p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            env_significance_info[k] = {'p_value': pseudo_p, 'significance': sig}
        
        if verbose:
            print("\nUsing pseudo-significance computed from LDA loadings")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # First 5 colors are specified, then use colormap for remaining
    base_colors = ['blue', 'red', 'green', 'orange', 'black']
    
    # Plot sites by cluster
    unique_clusters = sorted([c for c in clusters.unique() if not pd.isna(c)])
    # create a color map for clusters
    color_map = dict(zip(unique_clusters, base_colors))
    
    for cluster in unique_clusters:
        cluster_mask = clusters == cluster
        cluster_sites = lda_scores_df.loc[cluster_mask]
        
        ax.scatter(
            cluster_sites['LD1'],
            cluster_sites['LD2'] if n_components > 1 else np.zeros(len(cluster_sites)),
            c=color_map.get(cluster, 'gray'),
            label=f'Cluster {int(cluster)}',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Optionally add site labels
        if show_site_labels:
            for idx, row in cluster_sites.iterrows():
                ax.text(
                    row['LD1'] + site_label_offset,
                    row['LD2'] + site_label_offset if n_components > 1 else site_label_offset,
                    str(idx),
                    fontsize=site_fontsize,
                    alpha=0.7
                )
        
    # Plot habitat vectors with significance-based line styles
    from matplotlib.patches import FancyArrowPatch
    
    for habitat, row in habitat_vectors.iterrows():
        ld1_coef = row['LD1']
        ld2_coef = row['LD2'] if n_components > 1 else 0
        
        # Get significance information for this variable
        var_sig_info = env_significance_info.get(habitat, {'p_value': 0.5, 'significance': 'ns'})
        p_value = var_sig_info['p_value']
        sig_indicator = var_sig_info['significance']
        
        # Determine line style based on p-value
        # p < 0.05: solid line; p >= 0.05: dashed line
        is_significant = p_value < 0.05
        linestyle = '-' if is_significant else '--'
        arrow_alpha = 0.8 if is_significant else 0.5
        
        # Calculate arrow length
        arrow_length = np.sqrt(ld1_coef**2 + ld2_coef**2)
        
        # Draw arrow using plot + annotate for line style support
        # First draw the line (with appropriate style)
        ax.plot([0, ld1_coef], [0, ld2_coef], 
                color='blue', linestyle=linestyle, linewidth=2, alpha=arrow_alpha)
        
        # Add arrowhead at the tip
        ax.annotate('', xy=(ld1_coef, ld2_coef), xytext=(ld1_coef * 0.85, ld2_coef * 0.85),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2, mutation_scale=15),
                    annotation_clip=False)
        
        # Add label with significance closer to arrow tip
        # Position label slightly beyond arrow tip
        label_x = ld1_coef * (1 + env_label_offset)
        label_y = ld2_coef * (1 + env_label_offset)
        
        # Build label text with significance if requested
        if show_significance and sig_indicator != 'ns':
            label_text = f"{habitat} {sig_indicator}"
        else:
            label_text = habitat
        
        ax.text(
            label_x,
            label_y,
            label_text,
            fontsize=env_fontsize,
            color='blue',
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    # Add origin lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels and title
    var_exp = lda_model.explained_variance_ratio_
    ax.set_xlabel(f'LD1 ({var_exp[0]:.1%} of variance)', fontsize=12, fontweight='bold')
    if n_components > 1:
        ax.set_ylabel(f'LD2 ({var_exp[1]:.1%} of variance)', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('', fontsize=12)
    
    ax.set_title('LDA Triplot: Site Scores and Habitat Vectors', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if verbose:
        print("\n✓ LDA triplot created successfully!")
    
    return fig, ax


def plot_cluster_comparison(
    raw_data: pd.DataFrame,
    multiindex_data: pd.DataFrame,
    cluster_column: str = 'clusters',
    ref_column: str = 'if_ref',
    env_variables: Optional[List[str]] = None,
    standardize_env: bool = True,
    taxa_transformation: str = 'hellinger',
    anova_transform: Optional[str] = None,
    top_n_taxa: int = 16,
    taxa_order: Optional[List[str]] = ['Oligochaeta', 'Chironomidae', 'Nematoda', 'Sphaeriidae', 'Acari', 
              'Hexagenia', 'Caenis', 'Hirudinea', 'Turbellaria', 'Gastropoda', 
              'Hydrozoa', 'Other Trichoptera', 'Amphipoda', 'Hydropsychidae', 
              'Dreissena', 'Ceratopogonidae'],
    figsize: Tuple[float, float] = (18, 12),
    verbose: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 4-panel comparison figure showing habitat and taxa patterns across clusters.
    
    Panels:
    - (A) Upper left: Standardized habitat features across clusters (all sites)
    - (B) Upper right: Taxa composition across clusters (reference sites only)
    - (C) Lower left: Taxa composition across clusters (non-reference sites only)
    - (D) Lower right: Difference in taxa composition between non-ref and ref sites
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw data with cluster labels and reference indicators
    multiindex_data : pd.DataFrame
        MultiIndex data for extracting environmental and taxa data
    cluster_column : str, default='clusters'
        Column name containing cluster labels
    ref_column : str, default='if_ref'
        Column name indicating reference sites (True/False or 1/0)
    env_variables : Optional[List[str]], default=None
        List of environmental variables. If None, uses predefined subset
    standardize_env: bool = False,
        Whether to standardize environmental variables for panel A
    taxa_transformation : str, default='hellinger'
        Transformation for taxa data: 'hellinger', 'chord', 'octave', or 'none'
    anova_transform : Optional[str], default=None
        Transformation for ANOVA tests: None/'none', 'log', or 'box-cox'
    top_n_taxa : int, default=16
        Number of top taxa to display (by total abundance)
    taxa_order : Optional[List[str]], default=None
        Custom ordering of taxa on x-axis. If None, uses abundance-based order
    figsize : Tuple[float, float], default=(18, 12)
        Figure size in inches
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : np.ndarray
        Array of axes objects (2x2)
    """
    from zci.data_process import get_block
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    
    # ======================== HELPER FUNCTIONS ========================
    
    def transform_for_anova(data: pd.DataFrame, transform_type: Optional[str]) -> pd.DataFrame:
        """Apply transformation to data for ANOVA testing."""
        if transform_type is None or transform_type == 'none':
            return data.copy()
        elif transform_type == 'log':
            return np.log1p(data)
        elif transform_type in ('box-cox', 'boxcox'):
            transformed = data.copy()
            for col in transformed.columns:
                col_data = transformed[col].values.copy()
                if col_data.min() <= 0:
                    col_data = col_data - col_data.min() + 1e-6
                try:
                    transformed[col], _ = stats.boxcox(col_data)
                except:
                    transformed[col] = np.log1p(col_data)
            return transformed
        return data.copy()
    
    def perform_anova_test(data: pd.DataFrame, cluster_labels: pd.Series, 
                           transform_type: Optional[str] = None) -> Dict[str, float]:
        """Perform one-way ANOVA for each variable across clusters."""
        data_transformed = transform_for_anova(data, transform_type)
        p_values = {}
        
        for col in data_transformed.columns:
            groups = []
            for cluster in sorted(cluster_labels.unique()):
                cluster_sites = cluster_labels[cluster_labels == cluster].index
                available = [s for s in cluster_sites if s in data_transformed.index]
                if available:
                    groups.append(data_transformed.loc[available, col].dropna().values)
            
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                try:
                    _, p_val = stats.f_oneway(*groups)
                    p_values[col] = p_val
                except:
                    p_values[col] = 1.0
            else:
                p_values[col] = 1.0
        return p_values
    
    def perform_ttest_difference(data: pd.DataFrame, ref_mask: pd.Series,
                                  transform_type: Optional[str] = None) -> Dict[str, float]:
        """Perform independent samples t-test between non-ref and ref sites."""
        data_transformed = transform_for_anova(data, transform_type)
        p_values = {}
        
        for col in data_transformed.columns:
            ref_idx = ref_mask[ref_mask].index.intersection(data_transformed.index)
            nonref_idx = (~ref_mask)[~ref_mask].index.intersection(data_transformed.index)
            ref_data = data_transformed.loc[ref_idx, col].dropna()
            nonref_data = data_transformed.loc[nonref_idx, col].dropna()
            
            if len(ref_data) >= 2 and len(nonref_data) >= 2:
                try:
                    _, p_val = stats.ttest_ind(nonref_data, ref_data)
                    p_values[col] = p_val
                except:
                    p_values[col] = 1.0
            else:
                p_values[col] = 1.0
        return p_values
    
    def get_significance_marker(p_value: float) -> str:
        """Convert p-value to significance marker."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        return ''
    
    def calc_cluster_stats(site_subset: pd.DataFrame, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate mean and SEM for each cluster."""
        site_subset = site_subset.dropna(subset=[cluster_column])
        clusters = sorted(site_subset[cluster_column].unique())
        means, sems = [], []
        
        for cluster in clusters:
            cluster_sites = site_subset[site_subset[cluster_column] == cluster].index
            available = [s for s in cluster_sites if s in data.index]
            if available:
                cluster_data = data.loc[available]
                means.append(cluster_data.mean())
                sems.append(cluster_data.sem())
            else:
                means.append(pd.Series(0, index=data.columns))
                sems.append(pd.Series(0, index=data.columns))
        
        mean_df = pd.DataFrame(means, index=[f'Cluster {int(c)}' for c in clusters])
        sem_df = pd.DataFrame(sems, index=[f'Cluster {int(c)}' for c in clusters])
        return mean_df, sem_df
    
    # ======================== DATA EXTRACTION ========================
    
    if verbose:
        print("\n" + "="*80)
        print("CREATING CLUSTER COMPARISON FIGURE")
        print("="*80)
    
    # Extract environmental and taxa data blocks
    try:
        env_data_raw = get_block(multiindex_data, block='environmental', subblock='raw')
        taxa_data_raw = get_block(multiindex_data, block='taxa', subblock='raw')
        if verbose:
            print(f"\nExtracted data from multiindex blocks")
    except (KeyError, AttributeError):
        # Fallback extraction
        if hasattr(multiindex_data.columns, 'get_level_values'):
            level0 = multiindex_data.columns.get_level_values(0)
            taxa_mask = level0 == 'taxa'
            if taxa_mask.any():
                taxa_data_raw = multiindex_data.loc[:, taxa_mask].copy()
                taxa_data_raw.columns = taxa_data_raw.columns.get_level_values(-1)
            else:
                raise ValueError("No 'taxa' block found in multiindex")
        else:
            raise ValueError("Cannot extract taxa data from provided data")
        
        if env_variables is None:
            env_variables = ['MPS (Phi)', 'Measured Depth (m)', 
                            'Velocity  at bottom (m/sec)_Imputed', 'Temperature (oC)',
                            'Water DO Bottom (mg/L)', 'LOI (%)']
        available_env = [v for v in env_variables if v in raw_data.columns]
        env_data_raw = raw_data[available_env].copy()
    
    # Standardize habitat features if requested
    if standardize_env:
        scaler = StandardScaler()
        env_data = pd.DataFrame(
            scaler.fit_transform(env_data_raw),
            columns=env_data_raw.columns,
            index=env_data_raw.index
        )
    else:
        env_data = env_data_raw.copy()
    
    # Process taxa data
    taxa_data_raw = taxa_data_raw.select_dtypes(include=[np.number])
    
    # Select top N taxa by abundance
    if top_n_taxa is not None and top_n_taxa > 0:
        taxa_totals = taxa_data_raw.sum(axis=0)
        top_taxa_cols = taxa_totals.nlargest(top_n_taxa).index.tolist()
        taxa_data_raw = taxa_data_raw[top_taxa_cols]
        if verbose:
            print(f"Selected top {top_n_taxa} taxa by total abundance")
    
    # Apply custom taxa order if provided
    if taxa_order is not None:
        available_taxa = [t for t in taxa_order if t in taxa_data_raw.columns]
        taxa_data_raw = taxa_data_raw[available_taxa]
        if verbose:
            print(f"Applied custom taxa order: {len(available_taxa)} taxa")
    
    # Transform taxa data
    if taxa_transformation == 'hellinger':
        taxa_data = hellinger_transform(taxa_data_raw)
    elif taxa_transformation == 'chord':
        taxa_data = chord_transform(taxa_data_raw)
    elif taxa_transformation == 'octave':
        taxa_data = octave_transform(taxa_data_raw)
    else:
        taxa_data = taxa_data_raw.copy()
    
    # Filter environmental variables
    if env_variables is None:
        env_variables = ['MPS (Phi)', 'Measured Depth (m)', 
                        'Velocity  at bottom (m/sec)_Imputed', 'Temperature (oC)',
                        'Water DO Bottom (mg/L)', 'LOI (%)']
    env_data = env_data[[c for c in env_variables if c in env_data.columns]]
    
    # ======================== SPLIT BY REFERENCE STATUS ========================
    
    ref_mask = raw_data[ref_column].astype(bool)
    ref_data = raw_data[ref_mask].copy()
    nonref_data = raw_data[~ref_mask].copy()
    sites_with_clusters = raw_data.dropna(subset=[cluster_column])
    
    if verbose:
        print(f"\nReference sites: {ref_mask.sum()}, Non-reference: {(~ref_mask).sum()}")
        print(f"Taxa variables: {len(taxa_data.columns)}")
    
    # ======================== COMPUTE STATISTICS ========================
    
    # Habitat statistics (all sites)
    habitat_mean, habitat_sem = calc_cluster_stats(sites_with_clusters, env_data)
    
    # Taxa statistics for ANOVA (using transformed data)
    taxa_ref_mean_anova, taxa_ref_sem_anova = calc_cluster_stats(ref_data, taxa_data)
    taxa_nonref_mean_anova, taxa_nonref_sem_anova = calc_cluster_stats(nonref_data, taxa_data)
    
    # Taxa statistics for visualization (using relative abundance)
    taxa_relative = octave_to_relative_abundance(taxa_data)
    taxa_ref_mean, taxa_ref_sem = calc_cluster_stats(ref_data, taxa_relative)
    taxa_nonref_mean, taxa_nonref_sem = calc_cluster_stats(nonref_data, taxa_relative)
    
    # ======================== ANOVA TESTS ========================
    
    if verbose:
        print(f"\nPerforming ANOVA tests (transform: {anova_transform})...")
    
    all_cluster_labels = sites_with_clusters[cluster_column]
    habitat_pvalues = perform_anova_test(env_data, all_cluster_labels, anova_transform)
    
    # Use octave-transformed taxa data for ANOVA tests (proper for parametric tests)
    ref_cluster_labels = ref_data.dropna(subset=[cluster_column])[cluster_column]
    taxa_ref_pvalues = perform_anova_test(taxa_data, ref_cluster_labels, anova_transform)
    
    nonref_cluster_labels = nonref_data.dropna(subset=[cluster_column])[cluster_column]
    taxa_nonref_pvalues = perform_anova_test(taxa_data, nonref_cluster_labels, anova_transform)
    
    taxa_diff_pvalues = perform_ttest_difference(taxa_data, ref_mask, anova_transform)
    
    # ======================== CREATE FIGURE ========================
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    width = 0.25
    
    # -------------------- PANEL A: HABITAT FEATURES --------------------
    ax1 = axes[0, 0]
    var_names = list(habitat_mean.columns)
    n_vars = len(var_names)
    n_clusters = len(habitat_mean)
    x = np.arange(n_vars)
    
    for i, (cluster_name, row) in enumerate(habitat_mean.iterrows()):
        cluster_num = int(cluster_name.split()[-1])
        offset = (i - n_clusters/2 + 0.5) * width
        err_values = habitat_sem.loc[cluster_name].values
        lower_err = np.where(row.values < 0, err_values, 0)
        upper_err = np.where(row.values >= 0, err_values, 0)
        
        ax1.bar(x + offset, row.values, width, yerr=[lower_err, upper_err],
                label=cluster_name, color=cluster_colors[cluster_num], alpha=0.8,
                edgecolor='black', linewidth=0.5, capsize=3,
                error_kw={'linewidth': 1, 'elinewidth': 1})
    
    # ax1.set_xlabel('Habitat Variables', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Z-standardized Environmental Variables (±SE)', fontsize=14, fontweight='bold')
    ax1.set_title('Standardized Habitat Features Across Clusters (All Sites)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(var_names, rotation=30, ha='right', fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add significance markers for Panel A
    for i, var_name in enumerate(var_names):
        marker = get_significance_marker(habitat_pvalues.get(var_name, 1.0))
        if marker:
            # Find the cluster with max absolute mean value for this variable
            max_abs_cluster = None
            max_abs_val = -1
            for cn in habitat_mean.index:
                abs_val = abs(habitat_mean.loc[cn, var_name])
                if abs_val > max_abs_val:
                    max_abs_val = abs_val
                    max_abs_cluster = cn
            
            mean_val = habitat_mean.loc[max_abs_cluster, var_name]
            sem_val = habitat_sem.loc[max_abs_cluster, var_name]
            
            if mean_val >= 0:
                # Positive value: place marker above the bar
                marker_y = mean_val - 0.1
                va = 'bottom'
            else:
                # Negative value: place marker below the bar
                marker_y = mean_val + 0.1
                va = 'top'
            
            ax1.text(x[i] + offset + 0.15, marker_y, marker, ha='left', va=va,
                    fontsize=10, fontweight='bold', color='black')
    
    # -------------------- PANELS B, C: TAXA BY CLUSTER --------------------
    
    def plot_taxa_panel(ax, mean_df, sem_df, title, p_values):
        """Plot taxa bars with significance markers."""
        taxa_names = list(mean_df.columns)
        n_taxa = len(taxa_names)
        n_clust = len(mean_df)
        x_taxa = np.arange(n_taxa)
        
        for i, (cluster_name, row) in enumerate(mean_df.iterrows()):
            cluster_num = int(cluster_name.split()[-1])
            offset = (i - n_clust/2 + 0.5) * width
            err_values = sem_df.loc[cluster_name].values
            
            ax.bar(x_taxa + offset, row.values, width,
                   yerr=[np.zeros_like(err_values), err_values],
                   label=cluster_name, color=cluster_colors[cluster_num], alpha=0.8,
                   edgecolor='black', linewidth=0.5, capsize=2,
                   error_kw={'linewidth': 0.8, 'elinewidth': 0.8})
        
        ax.set_xlabel('Taxa', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Zoobenthic Relative Abundance(±SE) (%)', 
                     fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_taxa)
        ax.set_xticklabels(taxa_names, rotation=30, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add significance markers next to highest bar
        for i, var_name in enumerate(taxa_names):
            marker = get_significance_marker(p_values.get(var_name, 1.0))
            if marker:
                # Find the bar with largest absolute value
                max_val = max(mean_df.loc[cn, var_name] for cn in mean_df.index)
                min_val = min(mean_df.loc[cn, var_name] for cn in mean_df.index)
            
                if abs(max_val) >= abs(min_val):
                    # Positive bar is dominant - place marker above positive bar
                    marker_y = max_val - 0.01
                    va = 'top'
                else:
                    # Negative bar is dominant - place marker below negative bar
                    marker_y = min_val + 0.01
                    va = 'bottom'
                
                ax.text(x_taxa[i] + offset + 0.2, marker_y, marker, ha='left', va=va,
                       fontsize=9, fontweight='bold', color='black')
    
    ax2 = axes[0, 1]
    if not taxa_ref_mean.empty:
        plot_taxa_panel(ax2, taxa_ref_mean, taxa_ref_sem,
                       'Reference Sites: Taxa by Cluster', taxa_ref_pvalues)
    else:
        ax2.text(0.5, 0.5, 'No reference sites\nwith cluster labels',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Reference Sites: Taxa by Cluster', fontsize=14, fontweight='bold')
    
    ax3 = axes[1, 0]
    if not taxa_nonref_mean.empty:
        plot_taxa_panel(ax3, taxa_nonref_mean, taxa_nonref_sem,
                       'Non-Reference Sites: Taxa by Cluster', taxa_nonref_pvalues)
    else:
        ax3.text(0.5, 0.5, 'No non-reference sites\nwith cluster labels',
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Non-Reference Sites: Taxa by Cluster', fontsize=14, fontweight='bold')
    
    # -------------------- PANEL D: DIFFERENCE (NON-REF - REF) --------------------
    ax4 = axes[1, 1]
    
    if not taxa_ref_mean.empty and not taxa_nonref_mean.empty:
        common_clusters = sorted(set(taxa_nonref_mean.index) & set(taxa_ref_mean.index))
        
        if common_clusters:
            taxa_names = list(taxa_ref_mean.columns)
            n_taxa = len(taxa_names)
            x_taxa = np.arange(n_taxa)
            
            for cluster_name in common_clusters:
                cluster_num = int(cluster_name.split()[-1])
                diff = taxa_nonref_mean.loc[cluster_name] - taxa_ref_mean.loc[cluster_name]
                diff_sem = np.sqrt(taxa_nonref_sem.loc[cluster_name]**2 + 
                                  taxa_ref_sem.loc[cluster_name]**2)
                
                offset = (common_clusters.index(cluster_name) - len(common_clusters)/2 + 0.5) * width
                lower_err = np.where(diff.values < 0, diff_sem.values, 0)
                upper_err = np.where(diff.values >= 0, diff_sem.values, 0)
                
                ax4.bar(x_taxa + offset, diff.values, width,
                       yerr=[lower_err, upper_err], label=cluster_name,
                       color=cluster_colors[cluster_num], alpha=0.8,
                       edgecolor='black', linewidth=0.5, capsize=2,
                       error_kw={'linewidth': 0.8, 'elinewidth': 0.8})
            
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
            ax4.set_xlabel('Taxa', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Difference of Relative Abundance(±SE) (%)', fontsize=14, fontweight='bold')
            ax4.set_title('Average Difference in Taxa Composition\nbetween Non-Reference and Reference Sites', 
                         fontsize=14, fontweight='bold')
            ax4.set_xticks(x_taxa)
            ax4.set_xticklabels(taxa_names, rotation=30, ha='right', fontsize=12)
            ax4.tick_params(axis='y', labelsize=12)
            # ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(axis='y', alpha=0.3)
            
            # Add test annotation
            ax4.text(0.98, 0.95, r'(One-Sample $t$-test for $H_0$: $\Delta \mu = 0$)', 
                    transform=ax4.transAxes, fontsize=14, fontstyle='italic',
                    ha='right', va='top')
            
            # Add significance markers
            for i, var_name in enumerate(taxa_names):
                marker = get_significance_marker(taxa_diff_pvalues.get(var_name, 1.0))
                if marker:
                    # Find the bar with largest absolute difference value
                    diff_values = [(cn, taxa_nonref_mean.loc[cn, var_name] - taxa_ref_mean.loc[cn, var_name]) 
                                   for cn in common_clusters]
                    max_diff_cluster, max_diff = max(diff_values, key=lambda x: abs(x[1]))
                    
                    if max_diff >= 0:
                        # Positive bar is dominant - place marker above
                        marker_y = max_diff - 0.01
                        va = 'top'
                    else:
                        # Negative bar is dominant - place marker below
                        marker_y = max_diff + 0.01
                        va = 'bottom'
                    
                    ax4.text(x_taxa[i] + offset + 0.2, marker_y, marker, ha='left', va=va,
                           fontsize=9, fontweight='bold', color='black')
        else:
            ax4.text(0.5, 0.5, 'No matching clusters', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Average Difference in Taxa Composition\nbetween Non-Reference and Reference Sites',
                         fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Average Difference in Taxa Composition\nbetween Non-Reference and Reference Sites',
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if verbose:
        print("\n✓ Cluster comparison figure created successfully!")
    
    return fig, axes


# =============================================================================
# EXCEL TABLE FORMATTING FUNCTIONS FOR LDA RESULTS
# =============================================================================

def create_lda_confusion_matrix_table(
    lda_results: Dict[str, Any],
    table_type: str = 'training'
) -> pd.DataFrame:
    """
    Create a publication-ready confusion matrix table for LDA.
    
    Format matches publication style with:
    - Group column with cluster names
    - % Correct column showing classification accuracy per cluster
    - Predicted cluster columns with counts
    - Total row with overall accuracy and column totals
    
    Parameters
    ----------
    lda_results : dict
        Results from perform_lda_analysis() containing:
        - 'confusion_matrix': numpy array confusion matrix
        - 'cluster_names': list of cluster names
    table_type : str, default='training'
        Type of confusion matrix: 'training' for full model
        
    Returns
    -------
    pd.DataFrame
        Formatted confusion matrix table ready for Excel export
    """
    cm = lda_results['confusion_matrix']
    cluster_names = lda_results['cluster_names']
    n_clusters = len(cluster_names)
    
    # Calculate % Correct for each cluster (row-wise accuracy)
    row_totals = cm.sum(axis=1)
    correct_counts = np.diag(cm)
    pct_correct = np.where(row_totals > 0, (correct_counts / row_totals) * 100, 0)
    
    # Build rows
    rows = []
    for i, name in enumerate(cluster_names):
        cluster_num = int(name.split()[-1])
        row = {
            'Group': f'Cluster C{cluster_num + 1}',
            '% Correct': int(round(pct_correct[i]))
        }
        # Add predicted cluster columns
        for j, pred_name in enumerate(cluster_names):
            pred_num = int(pred_name.split()[-1])
            row[f'Cluster C{pred_num + 1}'] = int(cm[i, j])
        rows.append(row)
    
    # Add Total row
    total_correct = correct_counts.sum()
    total_samples = row_totals.sum()
    overall_pct = int(round((total_correct / total_samples) * 100)) if total_samples > 0 else 0
    col_totals = cm.sum(axis=0)
    
    total_row = {
        'Group': 'Total',
        '% Correct': overall_pct
    }
    for j, pred_name in enumerate(cluster_names):
        pred_num = int(pred_name.split()[-1])
        total_row[f'Cluster C{pred_num + 1}'] = int(col_totals[j])
    rows.append(total_row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Group')
    df.index.name = ''
    
    return df


def create_lda_classification_report_table(
    lda_results: Dict[str, Any],
    n_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a publication-ready classification report table for LDA.
    
    Formats the classification report with Precision, Recall, F1-Score, 
    and Support columns, matching the style in the reference image.
    
    Parameters
    ----------
    lda_results : dict
        Results from perform_lda_analysis() containing:
        - 'classification_report': dict with metrics per class
        - 'accuracy': overall accuracy
        - 'cluster_names': list of cluster names
    n_samples : int, optional
        Total number of samples (for the note at bottom)
        
    Returns
    -------
    pd.DataFrame
        Formatted classification report table ready for Excel export
    """
    report = lda_results['classification_report']
    cluster_names = lda_results['cluster_names']
    accuracy = lda_results['accuracy']
    
    # Build the table rows
    rows = []
    
    # Per-class metrics
    for name in cluster_names:
        if name in report:
            rows.append({
                'Class': name,
                'Precision': round(report[name]['precision'], 2),
                'Recall': round(report[name]['recall'], 2),
                'F1-Score': round(report[name]['f1-score'], 2),
                'Support': int(report[name]['support'])
            })
    
    # Add empty row before summary metrics
    rows.append({'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    
    # Accuracy row (only F1-Score column filled for accuracy)
    total_support = sum(report[name]['support'] for name in cluster_names if name in report)
    rows.append({
        'Class': 'Accuracy',
        'Precision': '–',
        'Recall': '–',
        'F1-Score': round(accuracy, 2),
        'Support': total_support
    })
    
    # Macro avg
    if 'macro avg' in report:
        rows.append({
            'Class': 'Macro avg',
            'Precision': round(report['macro avg']['precision'], 2),
            'Recall': round(report['macro avg']['recall'], 2),
            'F1-Score': round(report['macro avg']['f1-score'], 2),
            'Support': total_support
        })
    
    # Weighted avg
    if 'weighted avg' in report:
        rows.append({
            'Class': 'Weighted avg',
            'Precision': round(report['weighted avg']['precision'], 2),
            'Recall': round(report['weighted avg']['recall'], 2),
            'F1-Score': round(report['weighted avg']['f1-score'], 2),
            'Support': total_support
        })
    
    # Add note row
    rows.append({'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    note = f"Note: Overall Accuracy = {accuracy:.4f}; Number of sites = {total_support}"
    rows.append({'Class': note, 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    
    df = pd.DataFrame(rows)
    df = df.set_index('Class')
    df.index.name = ''
    
    return df


def create_mccv_confusion_matrix_table(
    cv_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a publication-ready aggregate confusion matrix from Monte Carlo CV.
    
    Format matches publication style with:
    - Group column with cluster names
    - % Correct column showing classification accuracy per cluster
    - Predicted cluster columns with counts
    - Total row with overall accuracy and column totals
    
    Parameters
    ----------
    cv_results : dict
        Results from perform_monte_carlo_cv() containing:
        - 'aggregate_confusion_matrix': combined confusion matrix from all iterations
        - 'cluster_names': list of cluster names
        - 'n_iterations': number of CV iterations
        - 'test_size': test set proportion
        
    Returns
    -------
    pd.DataFrame
        Formatted aggregate confusion matrix table ready for Excel export
    """
    cm = cv_results['aggregate_confusion_matrix']
    cluster_names = cv_results['cluster_names']
    n_iterations = cv_results['n_iterations']
    test_size = cv_results['test_size']
    n_clusters = len(cluster_names)
    
    # Calculate % Correct for each cluster (row-wise accuracy)
    row_totals = cm.sum(axis=1)
    correct_counts = np.diag(cm)
    pct_correct = np.where(row_totals > 0, (correct_counts / row_totals) * 100, 0)
    
    # Build rows
    rows = []
    for i, name in enumerate(cluster_names):
        cluster_num = int(name.split()[-1])
        row = {
            'Group': f'Cluster C{cluster_num + 1}',
            '% Correct': int(round(pct_correct[i]))
        }
        # Add predicted cluster columns
        for j, pred_name in enumerate(cluster_names):
            pred_num = int(pred_name.split()[-1])
            row[f'Cluster C{pred_num + 1}'] = int(cm[i, j])
        rows.append(row)
    
    # Add Total row
    total_correct = correct_counts.sum()
    total_samples = row_totals.sum()
    overall_pct = int(round((total_correct / total_samples) * 100)) if total_samples > 0 else 0
    col_totals = cm.sum(axis=0)
    
    total_row = {
        'Group': 'Total',
        '% Correct': overall_pct
    }
    for j, pred_name in enumerate(cluster_names):
        pred_num = int(pred_name.split()[-1])
        total_row[f'Cluster C{pred_num + 1}'] = int(col_totals[j])
    rows.append(total_row)
    
    # Add empty row then note row
    empty_row = {col: '' for col in ['Group', '% Correct'] + [f'Cluster C{int(n.split()[-1]) + 1}' for n in cluster_names]}
    rows.append(empty_row)
    
    note_row = empty_row.copy()
    note_row['Group'] = f"Note: Combined results from {n_iterations:,} CV iterations (test size: {test_size*100:.0f}%)"
    rows.append(note_row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Group')
    df.index.name = ''
    
    return df


def create_mccv_classification_report_table(
    cv_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a publication-ready classification report from Monte Carlo CV.
    
    Formats the average classification metrics with mean ± std format,
    matching the style in the reference image.
    
    Parameters
    ----------
    cv_results : dict
        Results from perform_monte_carlo_cv() containing:
        - 'avg_classification_report': dict with mean±std metrics per class
        - 'mean_accuracy', 'std_accuracy': accuracy statistics
        - 'median_accuracy': median accuracy
        - 'cluster_names': list of cluster names
        - 'n_iterations': number of CV iterations
        - 'test_size': test set proportion
        
    Returns
    -------
    pd.DataFrame
        Formatted classification report table ready for Excel export
    """
    report = cv_results['avg_classification_report']
    cluster_names = cv_results['cluster_names']
    mean_acc = cv_results['mean_accuracy']
    std_acc = cv_results['std_accuracy']
    median_acc = cv_results['median_accuracy']
    n_iterations = cv_results['n_iterations']
    test_size = cv_results['test_size']
    
    # Get support values from original data (approximate from first iteration)
    # Use the mean of test set sizes
    all_true = cv_results['all_true_labels']
    unique_clusters = sorted(set(all_true))
    
    # Count occurrences per cluster across all iterations
    cluster_counts = {}
    for c in unique_clusters:
        cluster_counts[c] = sum(1 for label in all_true if label == c) / n_iterations
    
    def format_mean_std(mean, std):
        """Format as 'mean ± std'"""
        return f"{mean:.3f} ± {std:.3f}"
    
    # Build the table rows
    rows = []
    
    # Per-class metrics
    for name in cluster_names:
        if name in report:
            cluster_id = int(name.split()[-1])
            support = int(round(cluster_counts.get(cluster_id, 0)))
            
            rows.append({
                'Class': name,
                'Precision': format_mean_std(report[name]['precision_mean'], report[name]['precision_std']),
                'Recall': format_mean_std(report[name]['recall_mean'], report[name]['recall_std']),
                'F1-Score': format_mean_std(report[name]['f1-score_mean'], report[name]['f1-score_std']),
                'Support': support
            })
    
    # Add empty row before summary metrics
    rows.append({'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    
    # Weighted avg
    if 'weighted avg' in report:
        total_support = sum(int(round(v)) for v in cluster_counts.values())
        rows.append({
            'Class': 'Weighted Avg',
            'Precision': format_mean_std(report['weighted avg']['precision_mean'], 
                                         report['weighted avg']['precision_std']),
            'Recall': format_mean_std(report['weighted avg']['recall_mean'], 
                                      report['weighted avg']['recall_std']),
            'F1-Score': format_mean_std(report['weighted avg']['f1-score_mean'], 
                                        report['weighted avg']['f1-score_std']),
            'Support': total_support
        })
    
    # Add empty row
    rows.append({'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    
    # Mean accuracy row
    rows.append({
        'Class': 'Mean accuracy',
        'Precision': '',
        'Recall': '',
        'F1-Score': format_mean_std(mean_acc, std_acc),
        'Support': ''
    })
    
    # Median accuracy row
    rows.append({
        'Class': 'Median accuracy',
        'Precision': '',
        'Recall': '',
        'F1-Score': f"{median_acc:.4f}",
        'Support': ''
    })
    
    # Add note rows
    rows.append({'Class': '', 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    note = f"Note: Classification metrics shown as mean ± standard deviation across {n_iterations:,} CV iterations"
    rows.append({'Class': note, 'Precision': '', 'Recall': '', 'F1-Score': '', 'Support': ''})
    
    df = pd.DataFrame(rows)
    df = df.set_index('Class')
    df.index.name = ''
    
    return df


def save_lda_tables_to_excel(
    lda_results: Dict[str, Any],
    cv_results: Dict[str, Any],
    save_path: str,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Save all LDA-related tables to Excel files.
    
    Creates four Excel files:
    1. lda_confusion_matrix.xlsx - Full model confusion matrix
    2. lda_classification_report.xlsx - Full model classification report
    3. mccv_confusion_matrix.xlsx - Monte Carlo CV aggregate confusion matrix
    4. mccv_classification_report.xlsx - Monte Carlo CV classification report
    
    Parameters
    ----------
    lda_results : dict
        Results from perform_lda_analysis()
    cv_results : dict
        Results from perform_monte_carlo_cv()
    save_path : str
        Directory path to save the Excel files
    verbose : bool, default=True
        Whether to print save confirmations
        
    Returns
    -------
    dict
        Dictionary mapping table names to file paths
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    saved_files = {}
    
    # 1. LDA Confusion Matrix (Training Data)
    cm_table = create_lda_confusion_matrix_table(lda_results)
    cm_path = os.path.join(save_path, 'lda_confusion_matrix.xlsx')
    cm_table.to_excel(cm_path)
    saved_files['lda_confusion_matrix'] = cm_path
    if verbose:
        print(f"  ✓ Saved: {cm_path}")
    
    # 2. LDA Classification Report (Training Data)
    report_table = create_lda_classification_report_table(lda_results)
    report_path = os.path.join(save_path, 'lda_classification_report.xlsx')
    report_table.to_excel(report_path)
    saved_files['lda_classification_report'] = report_path
    if verbose:
        print(f"  ✓ Saved: {report_path}")
    
    # 3. Monte Carlo CV Confusion Matrix
    mccv_cm_table = create_mccv_confusion_matrix_table(cv_results)
    mccv_cm_path = os.path.join(save_path, 'mccv_confusion_matrix.xlsx')
    mccv_cm_table.to_excel(mccv_cm_path)
    saved_files['mccv_confusion_matrix'] = mccv_cm_path
    if verbose:
        print(f"  ✓ Saved: {mccv_cm_path}")
    
    # 4. Monte Carlo CV Classification Report
    mccv_report_table = create_mccv_classification_report_table(cv_results)
    mccv_report_path = os.path.join(save_path, 'mccv_classification_report.xlsx')
    mccv_report_table.to_excel(mccv_report_path)
    saved_files['mccv_classification_report'] = mccv_report_path
    if verbose:
        print(f"  ✓ Saved: {mccv_report_path}")
    
    return saved_files
