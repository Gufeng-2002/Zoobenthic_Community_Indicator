"""
Bootstrap inference for Piecewise Quantile Regression Models.

This module provides functions for:
1. Bias correction of quantile regression residuals
2. Bootstrap resampling for uncertainty quantification
3. Confidence interval estimation for breakpoints and predictions
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.nonparametric.kde import KDEUnivariate

from .pqrm_model import fit_pqrm_pipeline


def correct_quantile_residuals(
    residuals: np.ndarray,
    X: np.ndarray,
    tau: float = 0.5
) -> np.ndarray:
    """
    Perform bias correction for quantile regression residuals.
    
    This implements the bias correction method that accounts for the asymmetric
    nature of quantile regression. The correction uses kernel density estimation
    and leverage scores to adjust residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Raw residuals from quantile regression (y - y_hat)
    X : np.ndarray
        Design matrix of shape (n_samples, n_features) or (n_samples,)
        Used to compute leverage scores. Only first column is used if 2D.
    tau : float, default=0.5
        Quantile level used in regression
    
    Returns
    -------
    corrected_residuals : np.ndarray
        Bias-corrected residuals
    
    Notes
    -----
    The bias correction formula is:
        corrected = residuals - (1/f(0)) * h * psi_tau
    where:
        - f(0) is the density of residuals at zero (from KDE)
        - h is the leverage (diagonal of hat matrix)
        - psi_tau = tau - I(residuals < 0)
    
    References
    ----------
    Koenker, R. (2005). Quantile Regression. Cambridge University Press.
    """
    n = len(residuals)
    
    # Estimate f(0): density of residuals at 0 using KDE
    kde = KDEUnivariate(residuals)
    kde.fit()
    f_0 = kde.evaluate(0)[0]  # density at 0
    
    # Avoid division by zero
    if f_0 < 1e-10:
        f_0 = 1e-10
    
    # Compute leverage for each observation
    # Using simplified leverage: h_i = x_i^2 / sum(x_j^2)
    x_flat = X[:, 0] if X.ndim > 1 else X
    x_squared_sum = np.sum(x_flat ** 2)
    if x_squared_sum < 1e-10:
        x_squared_sum = 1e-10
    h = x_flat ** 2 / x_squared_sum
    
    # Compute psi_tau (quantile loss derivative)
    psi = tau - (residuals < 0).astype(float)
    
    # Bias-corrected residuals
    corrected = residuals - (1 / f_0) * h * psi
    
    return corrected


def bootstrap_sample(
    data: pd.Series,
    n_samples: Optional[int] = None,
    random_state: Optional[int] = None
) -> pd.Series:
    """
    Generate a bootstrap sample (with replacement).
    
    Parameters
    ----------
    data : pd.Series
        Data to resample
    n_samples : int, optional
        Number of samples to draw. If None, uses len(data)
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    bootstrap_sample : pd.Series
        Resampled data with same length as input
    """
    if n_samples is None:
        n_samples = len(data)
    
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(data), size=n_samples)
    
    return data.iloc[indices]


def generate_bootstrap_response(
    fitted_model: object,
    data_with_breaks: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    use_bias_correction: bool = True,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate bootstrap response values using residual resampling.
    
    This function implements the wild bootstrap procedure for quantile regression:
    1. Extract residuals from fitted model
    2. Apply bias correction (optional)
    3. Bootstrap resample the residuals
    4. Add resampled residuals to fitted values
    
    Parameters
    ----------
    fitted_model : statsmodels result object
        Fitted PQRM model
    data_with_breaks : pd.DataFrame
        Data with breakpoint hinge features
    x_col : str, default='x'
        Name of predictor variable
    y_col : str, default='y_continuous'
        Name of response variable
    tau : float, default=0.5
        Quantile level
    use_bias_correction : bool, default=True
        Whether to apply bias correction to residuals
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    y_bootstrap : np.ndarray
        Bootstrap response values
    """
    # Get residuals
    residuals = fitted_model.resid
    
    # Apply bias correction if requested
    if use_bias_correction:
        X = data_with_breaks[[x_col]].values
        residuals_corrected = correct_quantile_residuals(residuals, X, tau)
    else:
        residuals_corrected = residuals
    
    # Bootstrap resample residuals
    n_samples = len(residuals_corrected)
    bootstrapped_residuals = bootstrap_sample(
        pd.Series(residuals_corrected),
        n_samples=n_samples,
        random_state=random_state
    )
    
    # Get fitted values
    y_fitted = fitted_model.predict(data_with_breaks)
    
    # Create bootstrap response
    y_bootstrap = y_fitted.values + bootstrapped_residuals.values
    
    return y_bootstrap


def bootstrap_pqrm_inference(
    fitted_model: object,
    data_with_breaks: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    max_breakpoints: int = 2,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    initial_guess: Optional[np.ndarray] = None,
    bounds: Optional[list] = None,
    max_iter: int = 5000,
    use_bias_correction: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Perform bootstrap inference for PQRM to obtain confidence intervals.
    
    This function implements the complete bootstrap procedure:
    1. Generate B bootstrap datasets using residual resampling
    2. Fit PQRM to each bootstrap dataset
    3. Collect predictions and breakpoints from each fit
    4. Compute confidence intervals from bootstrap distributions
    
    Parameters
    ----------
    fitted_model : statsmodels result object
        Base fitted PQRM model
    data_with_breaks : pd.DataFrame
        Original data with breakpoint features
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    tau : float, default=0.5
        Quantile level
    max_breakpoints : int, default=2
        Number of breakpoints
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    initial_guess : np.ndarray, optional
        Initial breakpoint guess for optimization
    bounds : list, optional
        Bounds for breakpoint optimization
    max_iter : int, default=5000
        Max iterations for quantile regression
    use_bias_correction : bool, default=True
        Whether to use bias-corrected residuals
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'bootstrap_predictions': DataFrame of predictions (rows=obs, cols=bootstrap)
        - 'bootstrap_breakpoints': DataFrame of breakpoints (rows=breakpoint, cols=bootstrap)
        - 'prediction_ci_lower': Lower CI for predictions
        - 'prediction_ci_upper': Upper CI for predictions
        - 'breakpoint_ci': List of tuples (lower, upper) for each breakpoint
        - 'breakpoint_ci_lower': Series of lower CI bounds
        - 'breakpoint_ci_upper': Series of upper CI bounds
    
    Examples
    --------
    >>> results = bootstrap_pqrm_inference(
    ...     fitted_model, data_with_breaks,
    ...     n_bootstrap=200, confidence_level=0.95
    ... )
    >>> lower_ci = results['prediction_ci_lower']
    >>> upper_ci = results['prediction_ci_upper']
    >>> breakpoint_cis = results['breakpoint_ci']
    """
    bootstrap_predictions = {}
    bootstrap_breakpoints = {}
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    if verbose:
        print(f"Starting bootstrap inference with {n_bootstrap} iterations...")
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} iterations")
        
        # Generate bootstrap response
        y_bootstrap = generate_bootstrap_response(
            fitted_model,
            data_with_breaks,
            x_col=x_col,
            y_col=y_col,
            tau=tau,
            use_bias_correction=use_bias_correction,
            random_state=None  # Use global seed set above
        )
        
        # Create bootstrap dataset
        data_bootstrap = data_with_breaks.copy()
        data_bootstrap['y_bootstrap'] = y_bootstrap
        
        # Fit PQRM to bootstrap data
        try:
            boot_model, boot_breakpoints, data_boot_breaks = fit_pqrm_pipeline(
                data=data_bootstrap,
                x_col=x_col,
                y_col='y_bootstrap',
                tau=tau,
                max_breakpoints=max_breakpoints,
                initial_guess=initial_guess,
                bounds=bounds,
                max_iter=max_iter,
                verbose=False
            )
            
            # Store predictions and breakpoints
            bootstrap_predictions[i] = boot_model.predict(data_boot_breaks)
            bootstrap_breakpoints[i] = boot_breakpoints
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Bootstrap iteration {i} failed: {e}")
            continue
    
    if verbose:
        print(f"Bootstrap inference complete. Successful: {len(bootstrap_predictions)}/{n_bootstrap}")
    
    # Convert to DataFrames
    bootstrap_predictions_df = pd.DataFrame(bootstrap_predictions)
    bootstrap_breakpoints_df = pd.DataFrame(bootstrap_breakpoints)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    # Prediction CIs
    prediction_ci_lower = bootstrap_predictions_df.quantile(lower_q, axis=1)
    prediction_ci_upper = bootstrap_predictions_df.quantile(upper_q, axis=1)
    
    # Breakpoint CIs
    breakpoint_ci_lower = bootstrap_breakpoints_df.quantile(lower_q, axis=1)
    breakpoint_ci_upper = bootstrap_breakpoints_df.quantile(upper_q, axis=1)
    breakpoint_ci = list(zip(breakpoint_ci_lower, breakpoint_ci_upper))
    
    return {
        'bootstrap_predictions': bootstrap_predictions_df,
        'bootstrap_breakpoints': bootstrap_breakpoints_df,
        'prediction_ci_lower': prediction_ci_lower,
        'prediction_ci_upper': prediction_ci_upper,
        'breakpoint_ci': breakpoint_ci,
        'breakpoint_ci_lower': breakpoint_ci_lower,
        'breakpoint_ci_upper': breakpoint_ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': len(bootstrap_predictions)
    }


def compute_prediction_bands(
    x_values: np.ndarray,
    bootstrap_predictions_df: pd.DataFrame,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction bands from bootstrap predictions.
    
    Parameters
    ----------
    x_values : np.ndarray
        X values for predictions
    bootstrap_predictions_df : pd.DataFrame
        Bootstrap predictions (rows=observations, cols=bootstrap iterations)
    confidence_level : float, default=0.95
        Confidence level
    
    Returns
    -------
    lower_band : np.ndarray
        Lower confidence band
    upper_band : np.ndarray
        Upper confidence band
    """
    alpha = 1 - confidence_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    lower_band = bootstrap_predictions_df.quantile(lower_q, axis=1).values
    upper_band = bootstrap_predictions_df.quantile(upper_q, axis=1).values
    
    return lower_band, upper_band
