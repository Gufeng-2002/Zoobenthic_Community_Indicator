"""
PCA assessment module for chemical contamination data

This module provides specialized functions to perform PCA on chemical data blocks 
from the hierarchical master DataFrame structure. It integrates with the ecoindex 
dataframe_ops module to extract chemical data subsets and perform contamination 
assessment via principal component analysis.

The module supports working with both raw and transformed chemical data 
(e.g., raw, hellinger, logz subblocks) from the master data structure.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Union, Literal, Tuple, Dict, Any
from .dataframe_ops import get_block, is_three_level
from .chemical_weights import TYPE_WEIGHTS as TYPE_WEIGHTS_FIG, VARIABLE_TYPE_BY_NAME as VAR_TYPE_BY_NAME_FIG, get_variable_weight as _get_weight, build_weights_for_columns as _build_weights

class PCAContaminationResult:
    """
    Result object for PCA-based contamination assessment of chemical data.
    
    Attributes:
        scores (pd.DataFrame): Site-level PC scores (sites x components)
        loadings (pd.DataFrame): Variable loadings (chemicals x components) 
        explained_variance (np.ndarray): Explained variance per component
        explained_variance_ratio (np.ndarray): Proportion of variance explained
        pca (sklearn.decomposition.PCA): Fitted PCA object
        chemical_block (str): Name of the chemical block analyzed
        subblock (str or None): Name of the chemical subblock analyzed
        n_sites (int): Number of sites included in analysis
        n_chemicals (int): Number of chemical variables included
        contamination_scores (pd.DataFrame): Standardized PC1 scores for contamination ranking
    """
    
    def __init__(self, scores: pd.DataFrame, loadings: pd.DataFrame, 
                 explained_variance: np.ndarray, explained_variance_ratio: np.ndarray,
                 pca: PCA, chemical_block: str, subblock: Optional[str] = None,
                 custom_weights: Optional[Dict[str, float]] = None):
        self.scores = scores
        self.loadings = loadings
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        self.pca = pca
        self.chemical_block = chemical_block
        self.subblock = subblock
        self.n_sites = scores.shape[0]
        self.n_chemicals = loadings.shape[0]
        self.custom_weights = custom_weights # added

        # Create contamination scores based on PC1 (assuming higher PC1 = more contaminated)
        self.contamination_scores = self._calculate_contamination_scores()
    
    def _calculate_contamination_scores(self) -> pd.DataFrame:
        """Calculate standardized contamination scores based on PC1."""
        pc1_scores = self.scores.iloc[:, 0]  # First principal component
        
        # Standardize to 0-100 scale (higher = more contaminated)
        min_score = pc1_scores.min()
        max_score = pc1_scores.max()
        standardized = ((pc1_scores - min_score) / (max_score - min_score)) * 100
        
        return pd.DataFrame({
            'contamination_score': standardized,
            'contamination_rank': standardized.rank(ascending=False, method='min').astype(int)
        }, index=pc1_scores.index)
    
    def get_most_contaminated_sites(self, n: int = 10) -> pd.DataFrame:
        """Return the n most contaminated sites based on PC1 scores."""
        return self.contamination_scores.nlargest(n, 'contamination_score')
    
    def get_least_contaminated_sites(self, n: int = 10) -> pd.DataFrame:
        """Return the n least contaminated sites based on PC1 scores."""
        return self.contamination_scores.nsmallest(n, 'contamination_score')
    
    def get_key_contaminants(self, pc: int = 1, n: int = 10) -> pd.DataFrame:
        """
        Return the chemicals with highest absolute loadings for a given PC.
        
        Args:
            pc: Principal component number (1-based)
            n: Number of top chemicals to return
        """
        pc_col = f'PC{pc}'
        if pc_col not in self.loadings.columns:
            raise ValueError(f"PC{pc} not available. Available PCs: {list(self.loadings.columns)}")
        
        loadings_abs = self.loadings[pc_col].abs()
        top_chemicals = loadings_abs.nlargest(n)
        
        return pd.DataFrame({
            'loading': self.loadings.loc[top_chemicals.index, pc_col],
            'abs_loading': top_chemicals
        })
    
    def __repr__(self):
        subblock_str = f", subblock='{self.subblock}'" if self.subblock else ""
        return (
            f"PCAContaminationResult(block='{self.chemical_block}'{subblock_str}, "
            f"sites={self.n_sites}, chemicals={self.n_chemicals}, "
            f"components={len(self.explained_variance)}, "
            f"var_explained_PC1={self.explained_variance_ratio[0]:.3f})"
        )


def pca_chemical_assessment(
    master_df: pd.DataFrame, 
    chemical_block: str = "chemical",
    subblock: Optional[str] = None,
    n_components: Optional[int] = None,
    standardize: bool = True,
    min_variance_threshold: float = 1e-10,
    dropna_threshold: float = 0.5,
    apply_weights: bool = False,
    custom_weights: Optional[Dict[str, float]] = None
) -> PCAContaminationResult:
    """
    Perform PCA-based contamination assessment on chemical data from master DataFrame.
    
    Args:
        master_df: Master DataFrame with hierarchical column structure
        chemical_block: Name of the chemical block (default: "chemical")
        subblock: Name of chemical subblock (e.g., "raw", "hellinger", "logz")
        n_components: Number of components to retain (None = all components)
        standardize: Whether to z-score standardize variables before PCA
        min_variance_threshold: Minimum variance threshold for including variables
        dropna_threshold: Drop variables with >threshold fraction of missing values
        apply_weights: Whether to apply variable weighting based on contamination importance
        custom_weights: Custom weight dictionary (optional)
    
    Returns:
        PCAContaminationResult object with scores, loadings, and contamination metrics
    """
    # Extract chemical data block
    try:
        chemical_data = get_block(master_df, chemical_block, subblock)
    except Exception as e:
        available_blocks = _get_available_blocks(master_df)
        raise ValueError(
            f"Could not extract chemical block '{chemical_block}'{f':{subblock}' if subblock else ''}. "
            f"Available blocks: {available_blocks}"
        ) from e
    
    # Data cleaning and preprocessing
    chemical_data = _preprocess_chemical_data(
        chemical_data, 
        min_variance_threshold=min_variance_threshold,
        dropna_threshold=dropna_threshold
    )
    
    if chemical_data.shape[1] < 2:
        raise ValueError(f"Insufficient chemical variables for PCA (found {chemical_data.shape[1]}, need ≥2)")
    
    # Prepare data matrix
    X = chemical_data.values.astype(float)
    
    # Handle missing values
    if np.any(np.isnan(X)):
        # Remove rows with any missing values (sites with incomplete data)
        complete_mask = ~np.any(np.isnan(X), axis=1)
        X = X[complete_mask]
        chemical_data = chemical_data.loc[complete_mask]
        print(f"Removed {(~complete_mask).sum()} sites with missing chemical data")
    
    if X.shape[0] < 3:
        raise ValueError(f"Insufficient sites for PCA after removing missing data (found {X.shape[0]}, need ≥3)")
    
    # Standardization
    if standardize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0, ddof=1)
        # Avoid division by zero for constant variables
        X_std[X_std < min_variance_threshold] = 1.0
        X = (X - X_mean) / X_std
    
    # Apply variable weighting based on contamination importance
    if apply_weights:
        weights = np.ones(X.shape[1])
        if custom_weights is not None:
            weight_dict = custom_weights
        else:
            weight_dict = {}
            for name in chemical_data.columns:
                # Prefer the exact mapping from the figure
                if name in VAR_TYPE_BY_NAME_FIG:
                    vtype = VAR_TYPE_BY_NAME_FIG[name]
                    weight_dict[name] = TYPE_WEIGHTS_FIG.get(vtype, 1.0)
                else:
                    # Default for unknowns
                    weight_dict[name] = 1.0

        for i, name in enumerate(chemical_data.columns):
            weights[i] = weight_dict.get(name, 1.0)

        # Scale by sqrt(weights) so variable variance is multiplied by weight
        X = X * np.sqrt(weights)

        print(
            f"Applied variable weights:"
            f"\n[3.0, -): {np.sum(weights >= 3.0)} vars, "
            f"\n[2.0, 3.0): {np.sum((weights >= 2.0) & (weights < 3.0))} vars, "
            f"\n[1.0, 2.0): {np.sum((1.0 < weights) & (weights < 2.0))} vars, "
            f"\n=1.0: {np.sum(weights == 1.0)} vars"
        )
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    
    # Create output DataFrames
    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, index=chemical_data.index, columns=pc_names)
    loadings_df = pd.DataFrame(loadings, index=chemical_data.columns, columns=pc_names)
    
    return PCAContaminationResult(
        scores=scores_df,
        loadings=loadings_df,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca=pca,
        chemical_block=chemical_block,
        subblock=subblock,
        custom_weights=custom_weights # added
    )


def _preprocess_chemical_data(
    df: pd.DataFrame,
    min_variance_threshold: float = 1e-10,
    dropna_threshold: float = 0.5
) -> pd.DataFrame:
    """Preprocess chemical data by removing low-variance and high-missing variables."""
    df_clean = df.copy()
    
    # Remove variables with too many missing values
    missing_fractions = df_clean.isnull().sum() / len(df_clean)
    high_missing = missing_fractions > dropna_threshold
    if high_missing.any():
        print(f"Removing {high_missing.sum()} variables with >{dropna_threshold} missing values")
        df_clean = df_clean.loc[:, ~high_missing]
    
    # Remove low-variance variables (after converting to numeric)
    df_numeric = df_clean.apply(pd.to_numeric, errors='coerce')
    variances = df_numeric.var()
    low_variance = variances < min_variance_threshold
    if low_variance.any():
        print(f"Removing {low_variance.sum()} variables with variance <{min_variance_threshold}")
        df_clean = df_clean.loc[:, ~low_variance]
    
    return df_clean


def _get_available_blocks(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary of available blocks in master DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        return {"error": "DataFrame does not have MultiIndex columns"}
    
    if is_three_level(df):
        # Three-level: (block, subblock, var)
        blocks = {}
        for block in df.columns.get_level_values(0).unique():
            subblocks = df.columns.get_level_values(1)[df.columns.get_level_values(0) == block].unique()
            blocks[block] = list(subblocks)
        return blocks
    else:
        # Two-level: (block, var)
        blocks = df.columns.get_level_values(0).unique()
        return {block: None for block in blocks}


# Legacy function for backward compatibility
def pca_assessment(df: pd.DataFrame, n_components: Optional[int] = None, 
                  standardize: bool = True, return_result: bool = True) -> Union[PCAContaminationResult, Tuple]:
    """
    Legacy PCA function for backward compatibility.
    
    Performs basic PCA on a DataFrame and returns either a result object or tuple.
    For new code, use pca_chemical_assessment() instead.
    """
    X = df.values.astype(float)
    
    # Handle missing values
    if np.any(np.isnan(X)):
        complete_mask = ~np.any(np.isnan(X), axis=1)
        X = X[complete_mask]
        df = df.loc[complete_mask]
    
    if standardize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
    
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    
    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, index=df.index, columns=pc_names)
    loadings_df = pd.DataFrame(loadings, index=df.columns, columns=pc_names)
    
    if return_result:
        return PCAContaminationResult(
            scores=scores_df,
            loadings=loadings_df,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            pca=pca,
            chemical_block="unknown",
            subblock=None
        )
    else:
        return scores_df, loadings_df, pca


def select_pcs_by_weighted_loadings(
    result: PCAContaminationResult,
    high_weight_threshold: float = 2.5,
    loading_threshold: float = 0.3,
    top_k_variables_per_pc: int = 10,
    prefer_positive: bool | None = None,
) -> dict:
    """
    Select principal components whose strongest loadings are dominated by high-weight variables.

    Args:
        result: PCAContaminationResult returned by pca_chemical_assessment.
        high_weight_threshold: Minimum variable weight to consider a variable as "high-weight".
        loading_threshold: Absolute loading cutoff to consider a variable strongly associated with a PC.
        top_k_variables_per_pc: Consider top-K absolute loadings per PC when evaluating dominance.
        prefer_positive: If True, only consider positive loadings; if False, only negative; if None, consider absolute.

    Returns:
        A dict with keys:
          - selected_pcs: list of PC names (e.g., ["PC1", "PC2"]) that meet criteria
          - pc_summaries: dict per PC with counts and lists of high-weight contributors (pos/neg/abs)
    """
    loadings = result.loadings
    pc_names = list(loadings.columns)

    # Build variable weights for the variables in this result using the figure mapping
    var_names = list(loadings.index)
    weight_map = _build_weights(var_names, weights_by_name = result.custom_weights)

    pc_summaries = {}
    selected = []

    explained_variance_by_pcs = {pc: result.explained_variance_ratio[i] for i, pc in enumerate(pc_names)}
    cumulative_explained_variance = 0

    for pc in pc_names:
        series = loadings[pc]
        # Sort by absolute loading to get the dominant variables on this PC
        dominant = series.reindex(series.abs().sort_values(ascending=False).index)
        if top_k_variables_per_pc is not None:
            dominant = dominant.iloc[:top_k_variables_per_pc]

        # Apply sign preference
        if prefer_positive is True:
            dominant = dominant[dominant > 0]
        elif prefer_positive is False:
            dominant = dominant[dominant < 0]

        # Apply minimum loading cutoff
        candidates = dominant[dominant.abs() >= loading_threshold]

        # Count high-weight variables in the candidates
        pos_high = []
        neg_high = []
        abs_high = []
        for var, val in candidates.items():
            w = weight_map.get(var, 1.0)
            if w >= high_weight_threshold:
                abs_high.append((var, float(val), float(w)))
                if val > 0:
                    pos_high.append((var, float(val), float(w)))
                elif val < 0:
                    neg_high.append((var, float(val), float(w)))

        pc_summaries[pc] = {
            "count_high_weight": len(abs_high),
            "pos_high": pos_high,
            "neg_high": neg_high,
            "abs_high": abs_high,
            "considered": [(v, dominant[v]) for v in dominant.index],
            "thresholds": {
                "weight": high_weight_threshold,
                "loading": loading_threshold,
            },
        }

        # Selection rule: if high-weight variables dominate the considered set, select this PC
        denom = max(1, len(dominant))
        if len(abs_high) / denom >= 0.5:
            selected.append(pc)
            cumulative_explained_variance += explained_variance_by_pcs[pc]
        
            
        
        

    return {"selected_pcs": selected, "pc_summaries": pc_summaries, "cumulative_explained_variance": cumulative_explained_variance}


def compute_pollution_scores_with_labels(
    result: PCAContaminationResult,
    filtered_pcs: list[str],
    quantiles: tuple[float, float] = (0.33, 0.67),
) -> pd.DataFrame:
    """
    Summarize selected PCs into a single SumReal score and assign quality labels.

    Parameters
    ----------
    result : PCAContaminationResult
        Result from pca_chemical_assessment containing `.scores` (rows are sites).
    filtered_pcs : list[str]
        PC names to sum for the SumReal pollution score (e.g., ["PC1", "PC2"]).
    quantiles : tuple(float, float), default (0.33, 0.67)
        Lower and upper quantile thresholds used to bin sites into
        ['reference', 'medium', 'degraded'] based on SumReal.

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame with columns: ['StationID', 'SumReal', 'Quality'].
    """
    # Validate PCs
    missing = [pc for pc in filtered_pcs if pc not in result.scores.columns]
    if missing:
        raise ValueError(f"Requested PCs not found in result.scores: {missing}")

    # Compute SumReal as sum of selected PCs per site
    sumreal = result.scores[filtered_pcs].sum(axis=1)
    sumreal.name = "SumReal"

    # Determine thresholds
    low_q, high_q = float(quantiles[0]), float(quantiles[1])
    low_thr = sumreal.quantile(low_q)
    high_thr = sumreal.quantile(high_q)

    # Assign quality labels
    def _label(v: float) -> str:
        if v <= low_thr:
            return "reference"
        if v <= high_thr:
            return "medium"
        return "degraded"

    quality = sumreal.apply(_label)

    # Build tidy output
    out = pd.DataFrame(
        {
            "StationID": result.scores.index,
            "SumReal": sumreal.values,
            "Quality": quality.values,
        }
    )
    return out
