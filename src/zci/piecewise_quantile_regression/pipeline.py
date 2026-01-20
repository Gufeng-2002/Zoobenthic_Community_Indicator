"""
High-level pipeline for Piecewise Quantile Regression Model analysis.

This module provides a complete end-to-end pipeline for PQRM fitting,
bootstrap inference, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
import warnings

from .pqrm_model import fit_pqrm_pipeline, PiecewiseQuantileRegression
from .bootstrap import bootstrap_pqrm_inference
from .visualizations import (
    plot_pqrm_fit,
    plot_pqrm_with_bootstrap_ci,
    plot_residual_diagnostics
)


def pqrm_analysis_pipeline(
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    max_breakpoints: int = 2,
    initial_guess: Optional[Union[List[float], np.ndarray]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_iter: int = 5000,
    run_bootstrap: bool = True,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    use_bias_correction: bool = True,
    create_visualizations: bool = True,
    show_residuals: bool = False,
    verbose: bool = True,
    random_state: Optional[int] = None
) -> Dict:
    """
    Complete PQRM analysis pipeline with bootstrap inference and visualization.
    
    This is the main high-level function that orchestrates the entire PQRM workflow:
    1. Fit the base PQRM model with breakpoint estimation
    2. Optionally perform bootstrap inference for confidence intervals
    3. Optionally create comprehensive visualizations
    4. Return all results in a structured dictionary
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing predictor and response variables
    x_col : str, default='x'
        Name of predictor variable column
    y_col : str, default='y_continuous'
        Name of response variable column
    tau : float, default=0.5
        Quantile level for regression (0 < tau < 1)
        - 0.5 = median regression
        - 0.1 = 10th percentile
        - 0.9 = 90th percentile
    max_breakpoints : int, default=2
        Number of breakpoints to estimate in the piecewise model
    initial_guess : array-like, optional
        Initial guess for breakpoint locations
        If None, uses evenly spaced points across x range
    bounds : list of tuples, optional
        Bounds (min, max) for each breakpoint during optimization
        If None, automatically segments the x range
    max_iter : int, default=5000
        Maximum iterations for quantile regression solver
    run_bootstrap : bool, default=True
        Whether to perform bootstrap inference for confidence intervals
    n_bootstrap : int, default=100
        Number of bootstrap iterations (if run_bootstrap=True)
        Recommended: 100-500 for exploratory, 500-1000 for publication
    confidence_level : float, default=0.95
        Confidence level for bootstrap intervals (e.g., 0.95 for 95% CI)
    use_bias_correction : bool, default=True
        Whether to apply bias correction to residuals in bootstrap
    create_visualizations : bool, default=True
        Whether to generate plots
    show_residuals : bool, default=False
        Whether to create residual diagnostic plots
    verbose : bool, default=True
        Whether to print progress messages
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    results : dict
        Comprehensive results dictionary containing:
        
        Model Results:
        - 'model': Fitted statsmodels quantile regression result
        - 'breakpoints': Estimated breakpoint locations (np.ndarray)
        - 'data_with_breaks': Data with breakpoint hinge features (pd.DataFrame)
        - 'tau': Quantile level used
        - 'predictions': Model predictions on input data
        - 'residuals': Model residuals
        
        Bootstrap Results (if run_bootstrap=True):
        - 'bootstrap_results': Full bootstrap inference results dict
        - 'prediction_ci_lower': Lower CI for predictions
        - 'prediction_ci_upper': Upper CI for predictions
        - 'breakpoint_ci': List of (lower, upper) tuples for each breakpoint
        - 'n_bootstrap': Number of successful bootstrap iterations
        
        Visualization Results (if create_visualizations=True):
        - 'figures': Dict of matplotlib figures
          - 'main_plot': PQRM fit with/without bootstrap CI
          - 'residuals': Residual diagnostics (if show_residuals=True)
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> results = pqrm_analysis_pipeline(data)
    >>> print(f"Breakpoints: {results['breakpoints']}")
    >>> fig = results['figures']['main_plot']
    >>> fig.savefig('pqrm_fit.png')
    
    Custom configuration for publication-quality analysis:
    
    >>> results = pqrm_analysis_pipeline(
    ...     data,
    ...     x_col='pollution_level',
    ...     y_col='species_diversity',
    ...     tau=0.1,  # 10th percentile
    ...     max_breakpoints=3,
    ...     n_bootstrap=500,
    ...     confidence_level=0.95,
    ...     initial_guess=[0.2, 0.5, 0.8],
    ...     show_residuals=True,
    ...     random_state=42
    ... )
    
    Quick analysis without bootstrap (faster):
    
    >>> results = pqrm_analysis_pipeline(
    ...     data,
    ...     run_bootstrap=False,
    ...     verbose=False
    ... )
    
    Notes
    -----
    - Bootstrap inference can be time-consuming for large datasets
    - Consider using fewer bootstrap iterations for exploratory analysis
    - Set random_state for reproducible results
    - Breakpoint bounds should ensure proper ordering (increasing)
    """
    if verbose:
        print("=" * 70)
        print("PQRM Analysis Pipeline")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - Quantile level (τ): {tau}")
        print(f"  - Number of breakpoints: {max_breakpoints}")
        print(f"  - Bootstrap inference: {run_bootstrap}")
        if run_bootstrap:
            print(f"  - Bootstrap iterations: {n_bootstrap}")
            print(f"  - Confidence level: {confidence_level}")
        print()
    
    # ========================================================================
    # Step 1: Fit base PQRM model
    # ========================================================================
    if verbose:
        print("Step 1: Fitting base PQRM model...")
    
    fitted_model, breakpoints, data_with_breaks = fit_pqrm_pipeline(
        data=data,
        x_col=x_col,
        y_col=y_col,
        tau=tau,
        max_breakpoints=max_breakpoints,
        initial_guess=initial_guess,
        bounds=bounds,
        max_iter=max_iter,
        verbose=verbose
    )
    
    if verbose:
        print(f"  ✓ Model fitted successfully")
        print(f"  ✓ Estimated breakpoints: {breakpoints}")
        print()
    
    # Get predictions and residuals
    predictions = fitted_model.predict(data_with_breaks)
    residuals = fitted_model.resid
    
    # Initialize results dictionary
    results = {
        'model': fitted_model,
        'breakpoints': breakpoints,
        'data_with_breaks': data_with_breaks,
        'tau': tau,
        'predictions': predictions,
        'residuals': residuals,
        'x_col': x_col,
        'y_col': y_col,
        'max_breakpoints': max_breakpoints
    }
    
    # ========================================================================
    # Step 2: Bootstrap inference (optional)
    # ========================================================================
    bootstrap_results = None
    if run_bootstrap:
        if verbose:
            print(f"Step 2: Performing bootstrap inference ({n_bootstrap} iterations)...")
        
        try:
            bootstrap_results = bootstrap_pqrm_inference(
                fitted_model=fitted_model,
                data_with_breaks=data_with_breaks,
                x_col=x_col,
                y_col=y_col,
                tau=tau,
                max_breakpoints=max_breakpoints,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                initial_guess=initial_guess,
                bounds=bounds,
                max_iter=max_iter,
                use_bias_correction=use_bias_correction,
                random_state=random_state,
                verbose=verbose
            )
            
            if verbose:
                n_success = bootstrap_results['n_bootstrap']
                print(f"  ✓ Bootstrap completed: {n_success}/{n_bootstrap} successful")
                print()
            
            # Add bootstrap results to main results dict
            results['bootstrap_results'] = bootstrap_results
            results['prediction_ci_lower'] = bootstrap_results['prediction_ci_lower']
            results['prediction_ci_upper'] = bootstrap_results['prediction_ci_upper']
            results['breakpoint_ci'] = bootstrap_results['breakpoint_ci']
            results['n_bootstrap'] = bootstrap_results['n_bootstrap']
            
        except Exception as e:
            warnings.warn(f"Bootstrap inference failed: {e}")
            if verbose:
                print(f"  ✗ Bootstrap failed: {e}")
                print()
    
    # ========================================================================
    # Step 3: Create visualizations (optional)
    # ========================================================================
    figures = {}
    
    if create_visualizations:
        if verbose:
            print("Step 3: Creating visualizations...")
        
        try:
            if run_bootstrap and bootstrap_results is not None:
                # Plot with bootstrap CI
                fig, ax = plot_pqrm_with_bootstrap_ci(
                    data=data,
                    data_with_breaks=data_with_breaks,
                    fitted_model=fitted_model,
                    breakpoints=breakpoints,
                    bootstrap_results=bootstrap_results,
                    x_col=x_col,
                    y_col=y_col,
                    tau=tau
                )
                figures['main_plot'] = fig
                
            else:
                # Plot without bootstrap CI
                fig, ax = plot_pqrm_fit(
                    data=data_with_breaks,
                    fitted_model=fitted_model,
                    breakpoints=breakpoints,
                    x_col=x_col,
                    y_col=y_col,
                    tau=tau
                )
                figures['main_plot'] = fig
            
            if verbose:
                print("  ✓ Main plot created")
            
            # Residual diagnostics
            if show_residuals:
                fig_res, axes_res = plot_residual_diagnostics(
                    fitted_model=fitted_model,
                    data_with_breaks=data_with_breaks,
                    x_col=x_col,
                    y_col=y_col,
                    tau=tau
                )
                figures['residuals'] = fig_res
                
                if verbose:
                    print("  ✓ Residual diagnostics created")
            
            if verbose:
                print()
            
        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")
            if verbose:
                print(f"  ✗ Visualization failed: {e}")
                print()
    
    results['figures'] = figures
    
    # ========================================================================
    # Summary
    # ========================================================================
    if verbose:
        print("=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)
        print(f"Results summary:")
        print(f"  - Breakpoints: {breakpoints}")
        if run_bootstrap and bootstrap_results is not None:
            print(f"  - Bootstrap iterations: {bootstrap_results['n_bootstrap']}")
            print(f"  - Breakpoint CIs: {bootstrap_results['breakpoint_ci']}")
        print(f"  - Figures created: {list(figures.keys())}")
        print("=" * 70)
    
    return results


def fit_multiple_quantiles(
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    taus: List[float] = [0.1, 0.5, 0.9],
    max_breakpoints: int = 2,
    initial_guess: Optional[Union[List[float], np.ndarray]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_iter: int = 5000,
    verbose: bool = True
) -> Dict[float, Dict]:
    """
    Fit PQRM models for multiple quantile levels.
    
    This function fits separate PQRM models for different quantiles,
    useful for understanding the full conditional distribution.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    taus : list of float, default=[0.1, 0.5, 0.9]
        Quantile levels to fit
    max_breakpoints : int, default=2
        Number of breakpoints per model
    initial_guess : array-like, optional
        Initial breakpoint guess (same for all quantiles)
    bounds : list of tuples, optional
        Breakpoint bounds (same for all quantiles)
    max_iter : int, default=5000
        Maximum iterations
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    results : dict
        Dictionary mapping tau -> pipeline results for that quantile
    
    Examples
    --------
    >>> results = fit_multiple_quantiles(
    ...     data,
    ...     taus=[0.1, 0.25, 0.5, 0.75, 0.9]
    ... )
    >>> for tau, res in results.items():
    ...     print(f"τ={tau}: breakpoints={res['breakpoints']}")
    """
    if verbose:
        print(f"Fitting PQRM for {len(taus)} quantile levels...")
        print()
    
    results = {}
    
    for i, tau in enumerate(taus):
        if verbose:
            print(f"[{i+1}/{len(taus)}] Fitting τ={tau}")
        
        result = pqrm_analysis_pipeline(
            data=data,
            x_col=x_col,
            y_col=y_col,
            tau=tau,
            max_breakpoints=max_breakpoints,
            initial_guess=initial_guess,
            bounds=bounds,
            max_iter=max_iter,
            run_bootstrap=False,
            create_visualizations=False,
            verbose=False
        )
        
        results[tau] = result
        
        if verbose:
            print(f"  Breakpoints: {result['breakpoints']}")
            print()
    
    if verbose:
        print("All quantiles fitted successfully!")
    
    return results


def compare_quantile_breakpoints(
    results_dict: Dict[float, Dict],
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compare breakpoint estimates across different quantile levels.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary from fit_multiple_quantiles()
    confidence_level : float, default=0.95
        Confidence level (unused if no bootstrap results)
    
    Returns
    -------
    comparison : pd.DataFrame
        DataFrame with breakpoints for each quantile level
    """
    data = []
    
    for tau, results in sorted(results_dict.items()):
        breakpoints = results['breakpoints']
        
        row = {'tau': tau}
        for i, bp in enumerate(breakpoints):
            row[f'breakpoint_{i+1}'] = bp
        
        # Add bootstrap CIs if available
        if 'breakpoint_ci' in results:
            for i, (lower, upper) in enumerate(results['breakpoint_ci']):
                row[f'bp_{i+1}_ci_lower'] = lower
                row[f'bp_{i+1}_ci_upper'] = upper
        
        data.append(row)
    
    return pd.DataFrame(data)
