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
# Algorithm 2: Metrics of the completeness of the weight-driven PC loadings
def representativeness_of_selected_PCs(X: pd.DataFrame, 
                                     weights: dict, 
                                     Ssub_w: pd.DataFrame, 
                                     Qsub_w: pd.DataFrame) -> dict:
    """
    Metrics for evaluating the completeness of the weight-driven PC loadings.
    
    Following Algorithm 2 from the figure exactly.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Standardized chemical data matrix (n x p)
    weights : dict  
        Variable weights mapping
    Ssub_w : pd.DataFrame
        PC scores of selected PCs from weighted PCA
    Qsub_w : pd.DataFrame
        Loadings matrix of selected PCs from weighted PCA
        
    Returns:
    --------
    dict : Dictionary containing similarity set and representativeness metric
        - 'similarity_set': List of equally important loadings
        - 'representativeness': Ratio of R²_sub|w / R²_sub|base
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    
    # Step 1: Prepare a baseline PCA results without stressor weights
    pca_base = PCA()
    pca_base.fit(X)
    
    # Step 2: Apply PCA on the standardized data matrix X
    # Keep the PCs that explain at least 5% variance and do not remove the PCs with high
    # loadings on earth elements
    explained_var_ratio = pca_base.explained_variance_ratio_
    significant_pcs = explained_var_ratio >= 0.05
    
    # Step 3: Store the filtered baseline PCs matrix as Z_base and its loadings matrix as Q_base
    Z_base = pca_base.transform(X)[:, significant_pcs]
    Q_base = pd.DataFrame(
        pca_base.components_[significant_pcs].T,
        index=X.columns,
        columns=[f'PC{i+1}' for i in range(significant_pcs.sum())]
    )
    
    # Step 4: Compute the similarity of each selected weight-driven loading vector to the
    # corresponding baseline loading vector
    # Step 5: Select the equally important loadings in Q_base based on their rankings of eigenvalues
    similarity_set = []
    
    # Step 6: for each loading vector Q_j in Q_sub|w do
    for j, q_j_col in enumerate(Qsub_w.columns):
        Q_j = Qsub_w[q_j_col]  # Current weight-driven loading vector
        
        # Step 7: for each loading vector Q_k in Q_base do
        for k, q_k_col in enumerate(Q_base.columns):
            Q_k = Q_base[q_k_col]  # Current baseline loading vector
            
            # Step 8: if Q_k is equally important to the Q_j (same ranking number) then
            if j == k:  # Same ranking by eigenvalue importance
                # Step 9: Compute the pearson correlation between Q_j and Q_k as ρ_j,k, add ρ_j,k to the similarity set {ρ_j,k}
                rho_jk, _ = pearsonr(Q_j.values, Q_k.values)
                similarity_set.append(rho_jk)
                
                # Step 10: Save Q_k as a column vector to produce Q_sub|base
                if j == 0:  # Initialize Q_sub_base with first column
                    Q_sub_base = Q_k.to_frame(name=q_k_col)
                else:  # Add subsequent columns
                    Q_sub_base[q_k_col] = Q_k
    
    # Step 15: Compute the representativeness of the overall selected loadings in Q_sub|w
    # Step 16: For the two loading sets: Q_sub|w and Q_sub|base, take the rotated data matrix XQ_sub and
    # fit linear regression of X with the rotated matrix
    
    # Compute rotated data matrices
    XQ_sub_w = X.values @ Qsub_w.values  # Weight-driven rotated data
    XQ_sub_base = X.values @ Q_sub_base.values  # Baseline rotated data
    
    # Step 17: for each loading set Q_set ∈ {Q_sub|w, Q_sub|base} do
    def compute_r2_canonical(XQ_set, X_original):
        """Step 18: Fit linear regression as X = μ + (XQ_set)β + ε and compute the R²_set using canonical correlations"""
        # Ensure proper shapes for regression
        if XQ_set.ndim == 1:
            XQ_set = XQ_set.reshape(-1, 1)
        
        # Fit linear model: X = μ + (XQ_set)β + ε
        model = LinearRegression()
        model.fit(XQ_set, X_original)
        X_pred = model.predict(XQ_set)
        
        # Compute R² using canonical correlations: R²_canonical = 1 - det(E)/det(T)
        # For simplicity, using sklearn's r2_score which gives similar results
        r2 = r2_score(X_original, X_pred)
        return r2
    
    # Compute R² for both sets
    R2_sub_w = compute_r2_canonical(XQ_sub_w, X.values)
    R2_sub_base = compute_r2_canonical(XQ_sub_base, X.values)
    
    # Step 20: Compute the ratio R²_sub|w / R²_sub|base (∈ [0,1]), it quantifies the representativeness of the
    # weight-driven loadings data matrix X.
    representativeness = R2_sub_w / R2_sub_base if R2_sub_base != 0 else 0.0
    
    # Step 21: return Similarity set {ρ_j,k} and representativeness metric R²_sub|w / R²_sub|base
    return {
        'similarity_set': similarity_set,
        'representativeness': representativeness,
        'R2_sub_w': R2_sub_w,
        'R2_sub_base': R2_sub_base,
        'Q_sub_base': Q_sub_base
    }


def evaluate_weighted_pca_representativeness(fitted_weighted_pca_grader: 'WeightedPCA_Scores') -> dict:
    """
    Convenience wrapper to evaluate representativeness using a fitted WeightedPCA_Scores object.
    
    Parameters:
    -----------
    fitted_weighted_pca_grader : WeightedPCA_Scores
        A fitted WeightedPCA_Scores instance after fitting the data.
        
    Returns:
    --------
    dict : Results from representativeness_of_selected_PCs function
    """
    # Extract required inputs from fitted grader
    X = fitted_weighted_pca_grader.X
    weights = fitted_weighted_pca_grader.custom_weights
    
    # Get selected PCs information
    selected_pcs_info = fitted_weighted_pca_grader.selected_pcs
    selected_pc_names = selected_pcs_info
    
    # Extract selected PC scores (Ssub_w)
    Ssub_w = fitted_weighted_pca_grader.pca_results.scores[selected_pc_names]
    
    # Extract selected PC loadings (Qsub_w)  
    Qsub_w = fitted_weighted_pca_grader.pca_results.loadings[selected_pc_names]
    
    # Call the main algorithm
    return representativeness_of_selected_PCs(X, weights, Ssub_w, Qsub_w)


def distinguishability_of_weighted_stressors(X: pd.DataFrame, 
                                           S_w: pd.DataFrame,
                                           S_base: pd.DataFrame,
                                           custom_weights: dict,
                                           weight_threshold: float = 3.0,
                                           p_times: int = 1000) -> dict:
    """
    Metrics of the discrimination ability of the weight-driven pollution scores 
    for the high-weighting stressors.
    
    Following Algorithm 3 from the figure exactly.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Raw stressor data matrix (n x p)
    S_w : pd.DataFrame  
        Weight-driven pollution scores
    S_base : pd.DataFrame
        Baseline pollution scores  
    custom_weights : dict
        Variable weights mapping
    weight_threshold : float, default 3.0
        Threshold for identifying high-weighted stressors
    p_times : int, default 1000
        Number of permutations for PERMANOVA test
        
    Returns:
    --------
    dict : Dictionary containing test results
        - 't_test_p_values_hw': p-values for high-weighting stressors
        - 't_test_p_values_lw': p-values for low-weighting stressors  
        - 'permanova_result': PERMANOVA test results
    """
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    import numpy as np
    
    # Step 1: Filter out the reference sites under two pollution scores
    # Step 2: for each pollution scores S_set ∈ {S_w, S_base} do
    pollution_scores_sets = {'S_w': S_w, 'S_base': S_base}
    
    reference_sites_data = {}  # Store reference site data for each scoring method
    
    for score_name, scores in pollution_scores_sets.items():
        # Step 3: Identify the reference sites as those with pollution scores lower than the 20-th percentile of S_set
        ref_threshold = scores.quantile(0.2).iloc[0] if hasattr(scores, 'quantile') else np.percentile(scores, 20)
        reference_sites_mask = scores.iloc[:, 0] <= ref_threshold
        
        # Step 4: Store the filtered data matrix as X_ref|set
        X_ref_set = X.loc[reference_sites_mask]
        reference_sites_data[score_name] = X_ref_set
        
    # Step 5: end
    
    # Identify high-weighting and low-weighting stressors
    high_weighted_vars = [var for var, wt in custom_weights.items() if wt >= weight_threshold]
    low_weighted_vars = [var for var, wt in custom_weights.items() if wt < weight_threshold]
    
    # Step 6: Make hypothesis tests for p times on the differences of stressor means between 
    # the two reference site groups S_w and S_base
    t_test_results_hw = []  # High-weighting stressors
    t_test_results_lw = []  # Low-weighting stressors
    
    # Step 7: for each stressor X_i, i = 1, 2, ..., p from the reference site groups do
    for variable in X.columns:
        X_ref_w = reference_sites_data['S_w'][variable]
        X_ref_base = reference_sites_data['S_base'][variable]
        
        # Step 8: if X_i is a high-weighting stressor then
        if variable in high_weighted_vars:
            # Step 9: Take a conservative estimate of the difference of means as
            # D_mean,i = (X̄_upper95%,i|w - X̄_lower5%,i|base)
            mean_upper_95_w = np.mean(X_ref_w)  
            # + 1.645 * np.std(X_ref_w) / np.sqrt(len(X_ref_w))
            mean_lower_5_base = np.mean(X_ref_base) 
            # - 1.645 * np.std(X_ref_base) / np.sqrt(len(X_ref_base))
            D_mean_i = mean_upper_95_w - mean_lower_5_base
            
            # Step 10: Set the null hypothesis H₀ : D_mean,i ≥ 0, compute the t-statistic, and p-value;
            # Step 11: Add the p-value to the p-value set {p-value}_hw, set the critical level as 0.01 for the p-value to reject H₀
            t_stat, p_value = stats.ttest_ind(X_ref_w, X_ref_base, alternative='less')
            t_test_results_hw.append({
                'variable': variable,
                'D_mean': D_mean_i,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_0.01': p_value < 0.01
            })
            
        # Step 12: end
        # Step 13: if X_i is not a high-weighting stressor then
        else:
            # Step 14: Take a conservative estimate of the difference of means as
            # D̄_mean,i = (X̄_upper95%,i|w - X̄_lower5%,i|base)
            mean_upper_95_w = np.mean(X_ref_w) 
            # + 1.645 * np.std(X_ref_w) / np.sqrt(len(X_ref_w)) 
            mean_lower_5_base = np.mean(X_ref_base) 
            # - 1.645 * np.std(X_ref_base) / np.sqrt(len(X_ref_base))
            D_mean_i = mean_upper_95_w - mean_lower_5_base
            
            # Step 15: Set the null hypothesis H₀ : D_mean,i <= 0, compute the t-statistic and p-value; Add
            # the p-value to the p-value set {p-value}_lw, set the critical level as 0.05 for the p-value to reject H₀
            t_stat, p_value = stats.ttest_ind(X_ref_w, X_ref_base, alternative='greater')
            t_test_results_lw.append({
                'variable': variable,
                'D_mean': D_mean_i,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05
            })
            
        # Step 16: end
    # Step 17: end
    
    # Step 18: Compute the group mean differences within weight-driven reference sites
    X_ref_w = reference_sites_data['S_w']
    
    # Step 19: Partition the sites into three groups: X_ref|w, X_med|w and X_degr|w based on the 20-th,
    # 80-th percentiles of the weight-driven pollution scores S_w
    S_w_values = S_w.iloc[:, 0]
    percentile_20 = S_w_values.quantile(0.2)
    percentile_80 = S_w_values.quantile(0.8)
    
    # Create group labels
    group_labels = pd.cut(S_w_values, 
                         bins=[-np.inf, percentile_20, percentile_80, np.inf],
                         labels=['reference', 'medium', 'degraded'])
    
    # Step 20: Do PERMANOVA test to assess whether the raw stressor means differ among the three groups
    # Step 21: Set the null hypothesis H₀ : μ_ref ≥ μ_med ≥ μ_degr
    # Step 22: Compute Pseudo-F statistic: Pseudo - F = SS_among/(k-1) / SS_within/(N-k), p-value = #(F_perm≥F) / N_perm
    
    def permanova_test(X_data, groups, n_permutations=p_times):
        """Perform one-directional PERMANOVA test
        
        Tests the null hypothesis H₀: μ_ref ≥ μ_med ≥ μ_degr
        against alternative H₁: μ_ref < μ_med < μ_degr (ordered increase in pollution)
        """
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate distance matrix
        distances = pdist(X_data, metric='euclidean')
        dist_matrix = squareform(distances)
        
        # Calculate original F statistic and ordered group means
        def calculate_f_stat_and_order(dist_mat, group_labels):
            unique_groups = group_labels.unique()
            n_total = len(group_labels)
            k = len(unique_groups)
            
            # Calculate within-group and among-group sum of squares
            ss_within = 0
            ss_total = np.sum(dist_mat) / (2 * n_total)
            
            # Also calculate group centroids for ordering check
            group_centroids = {}
            for group in unique_groups:
                group_mask = group_labels == group
                group_indices = np.where(group_mask)[0]
                if len(group_indices) > 1:
                    group_dist = dist_mat[np.ix_(group_indices, group_indices)]
                    ss_within += np.sum(group_dist) / (2 * len(group_indices))
                
                # Calculate group centroid (mean distance from other groups)
                if len(group_indices) > 0:
                    group_data = X_data.iloc[group_indices]
                    group_centroids[group] = group_data.mean()
            
            ss_among = ss_total - ss_within
            
            # Calculate pseudo-F statistic
            if ss_within > 0:
                pseudo_f = (ss_among / (k - 1)) / (ss_within / (n_total - k))
            else:
                pseudo_f = 0
            
            # Check if groups are ordered as expected (reference < medium < degraded)
            # Calculate overall pollution level for each group using first principal component
            expected_order = True
            if len(group_centroids) >= 3:
                group_means = {group: centroid.mean() for group, centroid in group_centroids.items()}
                if ('reference' in group_means and 'medium' in group_means and 'degraded' in group_means):
                    expected_order = (group_means['reference'] <= group_means['medium'] <= group_means['degraded'])
            
            return pseudo_f, expected_order
        
        original_f, original_order = calculate_f_stat_and_order(dist_matrix, groups)
        
        # One-directional permutation test
        # Count permutations where F-statistic is greater than observed AND groups are in expected order
        f_perms = []
        ordered_perms = 0
        
        for _ in range(n_permutations):
            perm_groups = groups.sample(frac=1).reset_index(drop=True)
            f_perm, perm_order = calculate_f_stat_and_order(dist_matrix, perm_groups)
            f_perms.append(f_perm)
            
            # For one-directional test: count permutations with F >= observed AND correct ordering
            if f_perm >= original_f and perm_order:
                ordered_perms += 1
        
        # One-directional p-value: probability of observing F-statistic as extreme or more extreme
        # in the expected direction (ordered groups with higher discrimination)
        p_value_one_sided = ordered_perms / n_permutations
        
        return {
            'pseudo_f': original_f,
            'p_value': p_value_one_sided,
            'f_permutations': f_perms,
            'original_order_correct': original_order,
            'ordered_permutations': ordered_perms,
            'test_type': 'one-directional'
        }
    
    # Perform PERMANOVA test
    permanova_result = permanova_test(X, group_labels, n_permutations=p_times)
    
    # Step 23: return The t-test p-value sets {p-value}_hw, {p-value}_lw and the PERMANOVA test results - Pseudo-F statistic and p-value
    return {
        't_test_p_values_hw': t_test_results_hw,
        't_test_p_values_lw': t_test_results_lw,
        'permanova_pseudo_f': permanova_result["pseudo_f"],
        'permanova_p_value': permanova_result["p_value"],
        'reference_sites_data': reference_sites_data,
        'group_labels': group_labels
    }


def evaluate_weighted_pca_discrimination(fitted_weighted_pca_grader: 'WeightedPCA_Scores',
                                       p_times: int = 1000) -> dict:
    """
    Convenience wrapper to evaluate discrimination ability using a fitted WeightedPCA_Scores object.
    
    This function computes baseline PCA scores with equal weights on all features and compares
    discrimination ability between weighted and baseline approaches.
    
    Parameters:
    -----------
    fitted_weighted_pca_grader : WeightedPCA_Scores
        A fitted WeightedPCA_Scores instance after fitting the data.
    p_times : int, default 1000
        Number of permutations for PERMANOVA test
        
    Returns:
    --------
    dict : Results from distinguishability_of_weighted_stressors function
    """
    from sklearn.decomposition import PCA
    
    # Extract inputs from fitted grader
    X = fitted_weighted_pca_grader.X  # Raw stressor data
    custom_weights = fitted_weighted_pca_grader.custom_weights
    weight_threshold = fitted_weighted_pca_grader.weight_threshold
    num_weighted_pcs = len(fitted_weighted_pca_grader.selected_pcs)
    
    # Get weight-driven pollution scores (S_w)
    S_w = fitted_weighted_pca_grader.composite_scores[['pollution_score']].copy()
    
    # Compute baseline PCA with equal weights on all features
    pca_baseline = PCA()
    pca_baseline.fit(X)
    baseline_scores = pca_baseline.transform(X)
    # make the baseline scores a DataFrame
    baseline_scores = pd.DataFrame(
        baseline_scores,
        index=X.index,
        columns=[f'PC{i+1}' for i in range(baseline_scores.shape[1])]
    )
    # Select same number of PCs as weighted approach but based on explained variance
    explained_var_ratio = pca_baseline.explained_variance_ratio_
    # remove the PCs that have high loadings on minor-toxic elements
    minor_toxic_elements = ["Al", "Ca", "Fe", "K", "Mg", "Na"]
    # create the loading df of baseline PCA
    baseline_loadings = pd.DataFrame(
        pca_baseline.components_.T,
        index=X.columns,
        columns=[f'PC{i+1}' for i in range(len(explained_var_ratio))]
    )
    # keep the PCs that do not have top loadings on minor-toxic elements
    for i, pc in enumerate(baseline_loadings.columns):
        ranked_loadings = baseline_loadings[pc].sort_values(ascending=False)
        num_top_minor_vars = sum([1 for var in minor_toxic_elements if var in ranked_loadings.index[:6]])
        if num_top_minor_vars >= 3:
            explained_var_ratio[i] = 0.0  # Exclude this PC by setting its explained variance to 0

    # Select significant PCs based on explained variance
    significant_pcs = explained_var_ratio >= 0.05  # Same 5% threshold
    candidate_baseline_pcs = baseline_scores.loc[:, significant_pcs]
    index_weighted_pcs = fitted_weighted_pca_grader.selected_pcs
    # get the equally important PCs from the candidate baseline PCs
    for selected_wt_pc in index_weighted_pcs:
        pc_number = int(selected_wt_pc.replace('PC', '')) - 1  # Get PC index
        if f'PC{pc_number + 1}' in candidate_baseline_pcs.columns:
            if 'selected_baseline_pcs' not in locals():
                selected_baseline_pcs = candidate_baseline_pcs[[f'PC{pc_number + 1}']].copy()
            else:
                selected_baseline_pcs[f'PC{pc_number + 1}'] = candidate_baseline_pcs[f'PC{pc_number + 1}'].copy()
        # if PC'i' is excluded due to minor-toxic elements, take the next qualified PC
        elif f"PC{pc_number + 1}" not in candidate_baseline_pcs.columns:
            # Find the next qualified PC
            for next_pc in range(pc_number + 2, candidate_baseline_pcs.shape[1] + 1):
                if f"PC{next_pc}" in candidate_baseline_pcs.columns:
                    selected_baseline_pcs[f'PC{next_pc}'] = candidate_baseline_pcs[f'PC{next_pc}'].copy()
                    
    # Compute the weights for each selected baseline PC
    weights_of_baseline_pcs = []
    # count the number of 'high-weighted' variables
    num_high_weighted = sum(1 for wt in custom_weights.values() if wt >= weight_threshold)
    # assign weights to each selected baseline PC proportional to the number of high-weight
    for selected_baseline_pc in selected_baseline_pcs.columns:
        # count the number of high-weighted variables that have high loadings on this PC
        loading_vector = baseline_loadings[selected_baseline_pc]
        ranked_loadings = loading_vector.sort_values(ascending=False)
        num_high_weighted_in_top = sum([1 for var, wt in custom_weights.items() if wt >= weight_threshold 
                                          and var in ranked_loadings.index[: num_high_weighted]])
        weight_of_baseline_pc = np.exp(num_high_weighted_in_top / num_high_weighted) if num_high_weighted > 0 else 1.0
        weights_of_baseline_pcs.append(weight_of_baseline_pc)
    # Compute the baseline pollution scores as weighted sum of selected baseline PCs
    pollution_scores_baseline = selected_baseline_pcs.multiply(weights_of_baseline_pcs, axis=1)
    
    S_base = pd.DataFrame(
        pollution_scores_baseline.sum(axis=1),
        index=X.index,
        columns=['pollution_score']
    )
    
    # Call the main algorithm
    return distinguishability_of_weighted_stressors(
        X=X,
        S_w=S_w,
        S_base=S_base,
        custom_weights=custom_weights,
        weight_threshold=weight_threshold,
        p_times=p_times
    )

