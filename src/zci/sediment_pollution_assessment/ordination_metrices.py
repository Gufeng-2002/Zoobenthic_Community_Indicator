"""Summary of the ordination metrics module.

Assuming there are sites have been classified into 3 pollution levels:
reference, medium and degraded. 

This module is to evalute the reference sites that:
- Do they represent low concentrations and variability in weighted raw pollutants?
- Do they represent low concentrations on other average-weighted stressor variables?

"""
import numpy as np 
import pandas as pd




# A simple function to perform groupby aggregation on the streesor data with quality labels
def groupby_aggregation(stressor, 
                        quality_labels,
                        custom_weights = None,
                        agg_functions = ['mean', 'std'],
                        weight_threshold = 3):
    """
    Perform groupby aggregation on a multi-level column DataFrame.
    
    Parameters:
    -----------
    stressor : pandas.DataFrame
        The raw stressor DataFrame that contains only one level column index and 
        is not transformed.
    quality_labels : list or pandas.Series
        List/Series of quality labels to group by
    custom_weights : dict, optional
        A dictionary specifying custom weights for certain columns,
        if given, the aggregation will only be performed on these high-weighted columns.
    agg_functions : str or list or dict, default ['mean', 'std']
        Aggregation function(s) to apply. Can be:
        - String: single function like 'mean', 'sum', 'count', 'std'
        - List: multiple functions like ['mean', 'std']
        - Dict: mapping of column names to functions
    weight_threshold : float, default 3
        A threshold to filter high-weighted columns based on their weights.

    Returns:
    --------
    pandas.DataFrame : Aggregated results with groupby variable as index
    """
    # combine the stressor with the position aligned quality labels
    stressor = stressor.copy()
    
    if custom_weights:
        # filter high-weighted variables based on the weight mapping and threshold
        high_weight_vars = [var for var, wt in custom_weights.items() if wt >= weight_threshold]
        # Only keep variables that exist in the stressor DataFrame
        high_weight_vars = [var for var in high_weight_vars if var in stressor.columns]
        if high_weight_vars:
            stressor = stressor[high_weight_vars]
        else:
            print(f"Warning: No variables found with weight >= {weight_threshold}")
    
    # concat the quality labels as a new column for the groupby operation
    stressor['quality_label'] = quality_labels

    # Remove the quality_label column before aggregation
    stressor_for_agg = stressor.drop('quality_label', axis=1)
    
    # perform groupby aggregation
    result = stressor_for_agg.groupby(stressor['quality_label'], observed=True).agg(agg_functions)

    return result.T


# Below are groups of metrices to test different quality of the grouping results
# (A) Representativeness of selected PCs in weighted PCA compared to 'equal important' PCs in ordinal PCA
# Algorithm 2: measure of the representativeness of selected PCs by weighted loadings
def representativeness_of_selected_PCs(fitted_weighted_pca_grader: 'WeightedPCAContaminationGrader') -> float:
    """
    Measure how well the selected PCs by weighted loadings represent the overall variance
    captured by all PCs in the weighted PCA.

    Parameters:
    -----------
    fitted_weighted_pca_grader : WeightedPCA_Scores
        A fitted WeightedPCA_Scores instance after fitting the data.

    Returns:
    --------
    float : Representativeness score between 0 and 1.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA
        
    # get the input standardized data - X and the weights
    X = fitted_weighted_pca_grader.X
    custom_weights = fitted_weighted_pca_grader.custom_weights
    
    # get the selected PCs from te fitted grader
    selected_pcs = fitted_weighted_pca_grader.selected_pcs
    
    # take out the selected eigenvectors in the PC space of weighted X
    selected_eigenvectors = fitted_weighted_pca_grader.pca_results.loadings[selected_pcs]
    # get the indices of selected PCs in the full loadings matrix
    selected_pc_indices = [i for i, pc in enumerate(fitted_weighted_pca_grader.pca_results.loadings.columns) if pc in selected_pcs]
    
    # compute the ordinal PCs if using these eigenvectors that were selected from weighted PCA
    weighted_declined_PCs = X.values @ selected_eigenvectors.values
    
    # compute the ordinal PCs from X without weights and select the same number of PCs of equal importance order
    pca_ordinal = PCA()
    pca_ordinal.fit(X.values)
    ordinal_PCs = pca_ordinal.transform(X.values)
    comparable_ordinal_PCs = ordinal_PCs[:, selected_pc_indices]
    # print(f"Comparable ordinal PCs: {comparable_ordinal_PCs}")
    
    # Now, compute the R²'s for X between the weighted PCs and the comparable ordinal PCs
    def compute_r2(PCs, X):
        # Ensure X is 2D for sklearn
        if PCs.ndim == 1:
            PCs = PCs.reshape(-1, 1)

        # Fit the model
        model = LinearRegression()
        model.fit(PCs, X)
        
        # Make predictions for R-squared calculation
        X_pred = model.predict(PCs)
        r2 = r2_score(X, X_pred)

        return r2

    r2_weighted_declined_PCs = compute_r2(weighted_declined_PCs, X)
    r2_comparable_ordinal_PCs = compute_r2(comparable_ordinal_PCs, X)
    
    kept_variance_ratio = r2_weighted_declined_PCs / r2_comparable_ordinal_PCs
    print(f"Kept variance ratio: {kept_variance_ratio}")

def distinguishability_of_weighted_stressors(fitted_weighted_pca_grader: 'WeightedPCAContaminationGrader') -> float:
    """
    Measure how well the selected PCs by weighted loadings can distinguish the high-weighted stressors
    better than the oridinal PCA.

    Parameters:
    -----------
    fitted_weighted_pca_grader : WeightedPCA_Scores
        A fitted WeightedPCA_Scores instance after fitting the data.

    Returns:
    --------
    float : Representativeness score between 0 and 1.
    """
    
    # get the input standardized data - X and the weights
    X = fitted_weighted_pca_grader.X
    custom_weights = fitted_weighted_pca_grader.custom_weights
    
    # get the stressors that are high-weighted
    weight_threshold = fitted_weighted_pca_grader.weight_threshold
    high_weighted_vars = [var for var, wt in custom_weights.items() if wt >= weight_threshold]
    num_high_weighted = len(high_weighted_vars)

    # take out the reference sites from the weighted PCA composite scores
    composite_scores = fitted_weighted_pca_grader.composite_scores
    rf_sites_weighted_pca = X.loc[composite_scores['pollution_quality'] == 'reference']
    
    # Now, compute an ordinal PCA results and select PCs by considering the high-weighted stressors but without weights
    from sklearn.decomposition import PCA
    # fit an ordinal PCA
    pca_ordinal = PCA()
    pca_ordinal.fit(X)
    ordinal_PCs = pca_ordinal.transform(X)
    # Get the loadings from ordinal PCA
    ordinal_PCA_loadings = pd.DataFrame(pca_ordinal.components_.T, 
                           index=X.columns, 
                           columns=[f'PC{i+1}' for i in range(pca_ordinal.components_.shape[0])])
    # consider the high-weighted stressors, it selects the ordinal PCs that have high loadings on these stressors
    explained_var_ratio = pca_ordinal.explained_variance_ratio_ # the explained variance ratio for each PC
    # filter out the PCs that have explained variance less than 5%
    ordinal_PCA_loadings = ordinal_PCA_loadings.loc[:, explained_var_ratio >= 0.05]
    ordinal_PCA_selected_pcs = []
    # filter the loadings of the high-weighted variables
    top_wt_var_loadings_distribution = np.array([])
    for high_wt_var in high_weighted_vars:
        loadings_of_var = ordinal_PCA_loadings.loc[high_wt_var]
        # indicate where are the top loadings as many as the number of high-weighted variables
        top_loading_wt_var_indices = loadings_of_var >= loadings_of_var.sort_values(ascending=False).iloc[num_high_weighted - 1]
        top_wt_var_loadings_distribution = np.vstack([top_wt_var_loadings_distribution, top_loading_wt_var_indices]) if top_wt_var_loadings_distribution.size else top_loading_wt_var_indices.values

    # add up the occurrence of top loadings of the high-weighted variables along all PCs
    if top_wt_var_loadings_distribution.shape[0] != 30:
        stacked_top_wt_var_loadings_distribution = np.sum(top_wt_var_loadings_distribution, axis=0)

    elif top_wt_var_loadings_distribution.shape[0] == 30:
        stacked_top_wt_var_loadings_distribution = top_wt_var_loadings_distribution

    # compute the number of high loadings of high-weighted variables in each ordinal PC
    ordinal_top_loading_wt_var_counts = pd.Series(stacked_top_wt_var_loadings_distribution, index=ordinal_PCA_loadings.columns)
    # sort the PCs by the counts of high loadings of high-weighted variables
    ordinal_top_loading_wt_var_counts = ordinal_top_loading_wt_var_counts.sort_values(ascending=False)
    # select the same number of ordinal PCs as the number of weighted-selected PCs, the ordinal PCs with higher counts will be selected first
    num_weighted_selected_pcs = len(fitted_weighted_pca_grader.selected_pcs)
    ordinal_PCA_selected_pcs = ordinal_top_loading_wt_var_counts.index.tolist()[:num_weighted_selected_pcs]
    
    # take out the reference sites from the ordinal PCA composite scores
    pca_scores = pd.DataFrame(ordinal_PCs, 
                             index=X.index, 
                             columns=[f'PC{i+1}' for i in range(ordinal_PCs.shape[1])]) # the PC scores from ordinal PCA
    
    ordinal_pollution_scores = pca_scores[ordinal_PCA_selected_pcs].sum(axis=1)
    
    # block the sites into rf/med/degraded by the PCA composite scores
    ordinal_pollution_quality = pd.cut(
            ordinal_pollution_scores,
            bins=[-np.inf, 
                ordinal_pollution_scores.quantile(fitted_weighted_pca_grader.group_thresholds[0]),
                ordinal_pollution_scores.quantile(fitted_weighted_pca_grader.group_thresholds[1]),
                np.inf],
            labels=["reference", "medium", "degraded"]
        )

    ordinal_pollution_scores_labels = pd.DataFrame({
        'pollution_score': ordinal_pollution_scores,
        'pollution_quality': ordinal_pollution_quality
    })
    
    # compare the high-weighted stressor means of reference groups between weighted PCA and ordinal PCA, the weighted PCA should have lower means
    rf_sites_ordinal_pca = X.loc[ordinal_pollution_scores_labels['pollution_quality'] == 'reference']
    print(rf_sites_ordinal_pca)
    
    # return ordinal_pollution_scores_labels

