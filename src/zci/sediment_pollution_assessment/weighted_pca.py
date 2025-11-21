"""
PCA assessment module for chemical contamination data

This module provides specialized functions to perform PCA on input stressor matrix.

It comprises three major functions: 
- weighted_PCA_computation: Perform PCA with weighted variables
- PCs_filter_loading_vs_weight: Select principal components based on weighted loadings
- pollution_scores_of_subPCs: Compute pollution scores from selected PCs
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Define a class to hold the composite PCA results
class PCAResult:
    def __init__(self, scores: pd.DataFrame, loadings: pd.DataFrame,
                 weights: dict, explained_var_ratio: np.ndarray):
        self.scores = scores
        self.loadings = loadings
        self.weights = weights
        self.explained_var_ratio = explained_var_ratio
        


# Function to perform weighted PCA
def weighted_PCA_computation(data: pd.DataFrame, custom_weights: dict) -> PCAResult:
    """
    Perform PCA on the data with custom variable weights.

    Parameters:
    - data: pd.DataFrame, input data matrix with samples as rows and variables as columns.
    It should be transformed before passed in if it needs (e.g., Logarithm).
    - custom_weights: dict, mapping of variable names to their weights

    Returns:
    - PCAResult: object containing scores, loadings, and explained_var_ratio
    """
    # Apply weights to the data
    X = data.values.astype(float).copy()

    # Produce the weights
    weights = np.array([custom_weights.get(col, 1.0) for col in data.columns])
    weighted_X = X * weights
    
    # Perform PCA
    pca = PCA()
    pca.fit(weighted_X)

    # Get scores and loadings
    scores = pd.DataFrame(pca.transform(X=X), # X is the standardized original data, no weights applied here
                          index=data.index, 
                          columns=[f'PC{i+1}' for i in range(len(pca.components_))])
    
    loadings = pd.DataFrame(pca.components_.T, 
                            index=data.columns, 
                            columns=[f'PC{i+1}' for i in range(len(pca.components_))])
    
    eigenvalues = pca.explained_variance_
    
    explained_var_ratio = eigenvalues/np.sum(eigenvalues)

    return PCAResult(scores=scores, loadings=loadings,
                     weights=custom_weights, explained_var_ratio=explained_var_ratio)


# Function to select PCs based on the weighted variables and their loadings
def PCs_filter_loading_vs_weight(loadings: pd.DataFrame,
                                 custom_weights: dict,
                                 explained_var_ratio: np.ndarray,
                                weight_threshold: float = 1.0) -> dict:
    """
    Select principal components based on weighted loadings.

    Parameters:
    - loadings: pd.DataFrame, PCA loadings with variables as rows and PCs as columns.
    - weight_threshold: float, threshold to consider a variable as high-weighted.
        It does have to be the highest in the 'custom_weights' mapping.

    Returns:
    - selected_pcs: list of str, names of selected principal components.
    """
    
    # all PC names
    pc_names = loadings.columns.tolist()
    
    # all variable names and their weights
    var_names = loadings.index.tolist()
    
    # get dict of variable weights
    var_weights = {var: custom_weights.get(var, 1.0) for var in var_names}
    
    # identify the number of high-weighted variables
    high_weighted_vars = [var for var, wt in var_weights.items() if wt >= weight_threshold]
    num_high_weighted = len(high_weighted_vars)
    
    # The rule to select PCs by the loadings of the high-weighted vars and minor-toxic vars
    selected_pcs = []
    weights_of_selected_pcs = []
    minor_toxic_vars = ["Al", "Ca", "Fe", "K", "Mg", "Na"]
    for i, pc in enumerate(pc_names):
        # The loadings of high-weighted variables should rank in the top loadings of the slelected PC
        pc_loadings = loadings[pc]
            
        # get the ranking of all loadings in this PC
        ranked_loadings = pc_loadings.sort_values(ascending=False)

        # check the number of minor-toxic variables in top 6 loadings (1.5 times of the minor-toxic variables)
        num_minor_toxic_in_top = sum(1 for var in ranked_loadings.index[:int(6)] if var in minor_toxic_vars)
        if num_minor_toxic_in_top >= 3:
            continue  # skip this PC if too many minor-toxic variables are in top loadings
        if pc_loadings.abs().max() >= 0.5: # if any loading is too high for any vars
            continue  # skip this PC 
        # count how many high-weighted variables are in the top loadings
        num_high_weighted_in_top = sum(1 for var in high_weighted_vars if var in ranked_loadings.index[:num_high_weighted])
        if explained_var_ratio[i] >= 0.05 and num_high_weighted_in_top >= num_high_weighted/2:
        # and num_high_weighted_in_top >= num_high_weighted * 0.5: # at least half of high-weighted vars in the top many
            # select this toxicity-related PC
            selected_pcs.append(pc)
            # set a weight that emphasizes PCs with more high-weighted variables in top loadings
            weight_of_pf = np.exp(num_high_weighted_in_top / num_high_weighted) if num_high_weighted > 0 else 1.0
            weights_of_selected_pcs.append(weight_of_pf)
            
    # Organize selected PCs information into a table format
    if selected_pcs:
        print("\n=== Selected Principal Components ===")
        print(f"{'PC':<8} {'Explained Var':<15} {'High-Weighted Variable Loadings'}")
        print("-" * 80)
        
        for pc in selected_pcs:
            pc_loadings = loadings[pc]
            
            # Get explained variance (you'll need to pass eigenvalues to this function)
            pc_index = int(pc.replace('PC', '')) - 1
            explained_ratio = explained_var_ratio[pc_index].round(4) if explained_var_ratio is not None else 'N/A'

            # Format loadings for high-weighted variables
            loadings_of_high_weights = pc_loadings[high_weighted_vars]
            loadings_str = ", ".join([f"{var}: {loading:.3f}" for var, loading in loadings_of_high_weights.items()])

            print(f"{pc:<8} {explained_ratio:<15} {loadings_str}")
        
        print("-" * 80)
        print(f"Total selected PCs: {len(selected_pcs)}")
        print(f"High-weighted variables ({len(high_weighted_vars)}): {', '.join(high_weighted_vars)}")
        print(f"Weights of selected PCs: {', '.join([f'{pc}: {wt:.3f}' for pc, wt in zip(selected_pcs, weights_of_selected_pcs)])}")
    else:
        print("No principal components were selected based on the criteria.")
        
    return {"selected_pcs": selected_pcs, "selected_pc_weights": weights_of_selected_pcs}

# Function to compute the pollution scores from selected PCs
def pollution_scores_of_subPCs(pca_scores: pd.DataFrame,
                               selected_pcs: list,
                               weights_of_selected_pcs: list = None,
                               with_group_labels: bool = False,
                               group_thresholds = (0.2, 0.8)) -> pd.Series:
    """
    Compute pollution scores from selected principal components.

    Parameters:
    - pca_scores: pd.DataFrame, PCA scores with samples as rows and PCs as columns.
    - selected_pcs: list of str, names of selected principal components.
    - weights_of_selected_pcs: list of float, weights for each selected PC. If None, defaults to equal weights.
    - with_group_labels: bool, whether to return group labels along with scores.
    - group_thresholds: tuple, quantile thresholds for grouping (reference, medium, degraded).

    Returns:
    - pollution_scores: pd.Series or pd.DataFrame, computed pollution scores for each sample.
    """
    if not selected_pcs:
        raise ValueError("No principal components selected for pollution score computation.")

    # Set default weights if not provided
    if weights_of_selected_pcs is None:
        weights_of_selected_pcs = [1.0] * len(selected_pcs)

    # Ensure weights list has the same length as selected PCs
    if len(weights_of_selected_pcs) != len(selected_pcs):
        raise ValueError(f"Length mismatch: {len(weights_of_selected_pcs)} weights for {len(selected_pcs)} PCs")

    # Sum the scores of the selected PCs with their weights to get pollution scores
    # Convert weights to numpy array for proper broadcasting
    weights_array = np.array(weights_of_selected_pcs)
    selected_scores = pca_scores[selected_pcs]
    
    # Apply weights: multiply each PC column by its corresponding weight
    weighted_scores = selected_scores * weights_array
    pollution_scores = weighted_scores.sum(axis=1)

    # with_group_labels: partition the sites with labels for easier reference
    if with_group_labels:
        pollution_quality = pd.cut(
            pollution_scores,
            bins=[-np.inf, 
                pollution_scores.quantile(group_thresholds[0]),
                pollution_scores.quantile(group_thresholds[1]),
                np.inf],
            labels=["reference", "medium", "degraded"]
        )

        pollution_scores_labels = pd.DataFrame({
            'pollution_score': pollution_scores,
            'pollution_quality': pollution_quality
        })

        return pollution_scores_labels
    
    else: 
        return pollution_scores

# stack the functions together and make a fit_transfrom style function to match with sklearn
class WeightedPCA_Scores:
    def __init__(self, custom_weights: dict,
                 weight_threshold: float = 3.0,
                 group_thresholds = (0.2, 0.8)):
        self.custom_weights = custom_weights
        self.weight_threshold = weight_threshold
        self.group_thresholds = group_thresholds
        self.pca_results = None
        self.selected_pcs = None
        self.selected_pcs_weights = None
        self.X = None
        self.composite_scores = None
        
    # the function that fits and returns the pollution scores (without labels)
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # store the input data
        self.X = X.copy()
        
        # Step 1: Perform weighted PCA
        pca_results = weighted_PCA_computation(X, self.custom_weights)
        self.pca_results = pca_results
        
        # Step 2: Select PCs based on weighted loadings
        selected_pcs_result = PCs_filter_loading_vs_weight(
            pca_results.loadings,
            self.custom_weights,
            pca_results.explained_var_ratio,
            weight_threshold=self.weight_threshold)
        self.selected_pcs = selected_pcs_result["selected_pcs"]
        self.selected_pcs_weights = selected_pcs_result["selected_pc_weights"]

        # Step 3: Compute pollution scores from selected PCs
        composite_scores = pollution_scores_of_subPCs(
            pca_results.scores,
            self.selected_pcs,
            weights_of_selected_pcs=self.selected_pcs_weights,  # Use the PC weights
            with_group_labels=True,
            group_thresholds=self.group_thresholds
        )

        self.composite_scores = composite_scores
        return composite_scores
        
    
        
        