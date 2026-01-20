"""
Piecewise Quantile Regression Model (PQRM) Module

This module provides a comprehensive toolkit for fitting piecewise quantile regression
models with automatic breakpoint estimation and bootstrap inference for uncertainty
quantification.

Main Components:
----------------
1. pqrm_model: Core model fitting with breakpoint optimization
2. bootstrap: Bootstrap inference for confidence intervals
3. visualizations: Plotting functions for results
4. pipeline: High-level orchestration functions

Quick Start:
------------
For most users, the pipeline function provides everything needed:

    >>> from zci.piecewise_quantile_regression import pqrm_analysis_pipeline
    >>> results = pqrm_analysis_pipeline(
    ...     data,
    ...     x_col='pollution',
    ...     y_col='biodiversity',
    ...     tau=0.5,
    ...     max_breakpoints=2,
    ...     n_bootstrap=200
    ... )
    >>> print(f"Breakpoints: {results['breakpoints']}")
    >>> results['figures']['main_plot'].show()

For more control, use individual components:

    >>> from zci.piecewise_quantile_regression import (
    ...     fit_pqrm_pipeline,
    ...     bootstrap_pqrm_inference,
    ...     plot_pqrm_with_bootstrap_ci
    ... )
    >>> # Fit model
    >>> model, breakpoints, data_breaks = fit_pqrm_pipeline(data, tau=0.5)
    >>> # Bootstrap inference
    >>> boot_results = bootstrap_pqrm_inference(model, data_breaks, n_bootstrap=200)
    >>> # Visualize
    >>> fig, ax = plot_pqrm_with_bootstrap_ci(data, data_breaks, model, breakpoints, boot_results)

What is PQRM?
-------------
Piecewise Quantile Regression Models (PQRM) extend standard quantile regression
by allowing the relationship between predictor and response to change at unknown
breakpoints. This is particularly useful for:

- Detecting thresholds in ecological responses
- Identifying regime shifts in time series
- Modeling non-linear relationships with distinct phases
- Quantifying uncertainty in threshold locations

The model has the form:
    Q_tau(y|x) = beta_0 + beta_1*x + sum_i(gamma_i * max(x - alpha_i, 0))

where:
    - Q_tau is the tau-th conditional quantile
    - alpha_i are the estimated breakpoints
    - The max(x - alpha_i, 0) terms create "hinge" functions

Key Features:
-------------
- Automatic breakpoint estimation via optimization
- Bootstrap inference for breakpoint and prediction uncertainty
- Bias correction for quantile regression residuals
- Comprehensive visualization functions
- Scikit-learn style API with fit/predict methods
- Support for multiple quantile levels

Examples:
---------
1. Basic median regression with 2 breakpoints:

>>> import pandas as pd
>>> from zci.piecewise_quantile_regression import pqrm_analysis_pipeline
>>> 
>>> # Prepare your data
>>> data = pd.DataFrame({
...     'x': x_values,
...     'y_continuous': y_values
... })
>>> 
>>> # Run complete analysis
>>> results = pqrm_analysis_pipeline(
...     data,
...     tau=0.5,  # median
...     max_breakpoints=2,
...     n_bootstrap=200,
...     confidence_level=0.95
... )
>>> 
>>> # Access results
>>> print("Breakpoints:", results['breakpoints'])
>>> print("Breakpoint CIs:", results['breakpoint_ci'])
>>> results['figures']['main_plot'].savefig('pqrm_fit.png')

2. Compare multiple quantiles:

>>> from zci.piecewise_quantile_regression import fit_multiple_quantiles
>>> from zci.piecewise_quantile_regression import plot_multiple_quantiles
>>> 
>>> # Fit models for different quantiles
>>> multi_results = fit_multiple_quantiles(
...     data,
...     taus=[0.1, 0.25, 0.5, 0.75, 0.9],
...     max_breakpoints=2
... )
>>> 
>>> # Compare breakpoints
>>> for tau, res in multi_results.items():
...     print(f"τ={tau}: {res['breakpoints']}")
>>> 
>>> # Visualize all quantiles together
>>> models_dict = {
...     tau: (res['model'], res['breakpoints'], res['data_with_breaks'])
...     for tau, res in multi_results.items()
... }
>>> fig, ax = plot_multiple_quantiles(data, models_dict)

3. Using the PiecewiseQuantileRegression class:

>>> from zci.piecewise_quantile_regression import PiecewiseQuantileRegression
>>> 
>>> # Create and fit model
>>> pqrm = PiecewiseQuantileRegression(
...     tau=0.5,
...     max_breakpoints=2,
...     verbose=True
... )
>>> pqrm.fit(data, x_col='x', y_col='y_continuous')
>>> 
>>> # Make predictions
>>> predictions = pqrm.predict(new_data)
>>> 
>>> # Get breakpoints
>>> print("Breakpoints:", pqrm.breakpoints_)
>>> 
>>> # Get model summary
>>> print(pqrm.summary())

4. Custom breakpoint bounds:

>>> results = pqrm_analysis_pipeline(
...     data,
...     tau=0.5,
...     max_breakpoints=2,
...     initial_guess=[0.3, 0.7],  # Starting values
...     bounds=[(0.2, 0.4), (0.6, 0.8)],  # Constrain search
...     n_bootstrap=100
... )

References:
-----------
- Koenker, R. (2005). Quantile Regression. Cambridge University Press.
- Muggeo, V. M. (2003). Estimating regression models with unknown break-points.
  Statistics in Medicine, 22(19), 3055-3071.
- Chernozhukov, V., & Hansen, C. (2008). Instrumental variable quantile regression:
  A robust inference approach. Journal of Econometrics, 142(1), 379-398.

Module Structure:
-----------------
piecewise_quantile_regression/
├── __init__.py           # This file
├── pqrm_model.py         # Core model fitting
├── bootstrap.py          # Bootstrap inference
├── visualizations.py     # Plotting functions
└── pipeline.py           # High-level orchestration

See Also:
---------
- contamination_assessment: For PCA-based pollution analysis
- env_driven_taxa_clusters: For LDA-based habitat separation
"""

# =============================================================================
# Main Pipeline (Recommended Entry Point)
# =============================================================================
from .pipeline import (
    pqrm_analysis_pipeline,
    fit_multiple_quantiles,
    compare_quantile_breakpoints
)

# =============================================================================
# Core Model Fitting
# =============================================================================
from .pqrm_model import (
    fit_pqrm_pipeline,
    fit_pqrm_with_breakpoints,
    quantile_check_loss,
    PiecewiseQuantileRegression
)

# =============================================================================
# Bootstrap Inference
# =============================================================================
from .bootstrap import (
    bootstrap_pqrm_inference,
    correct_quantile_residuals,
    generate_bootstrap_response,
    bootstrap_sample,
    compute_prediction_bands
)

# =============================================================================
# Visualizations
# =============================================================================
from .visualizations import (
    plot_pqrm_fit,
    plot_pqrm_with_bootstrap_ci,
    plot_residual_diagnostics,
    plot_bootstrap_distribution,
    plot_multiple_quantiles
)

# =============================================================================
# Version and Metadata
# =============================================================================
__version__ = '1.0.0'
__author__ = 'ZCI Research Team'
__all__ = [
    # Pipeline functions (recommended)
    'pqrm_analysis_pipeline',
    'fit_multiple_quantiles',
    'compare_quantile_breakpoints',
    
    # Core model
    'fit_pqrm_pipeline',
    'fit_pqrm_with_breakpoints',
    'quantile_check_loss',
    'PiecewiseQuantileRegression',
    
    # Bootstrap inference
    'bootstrap_pqrm_inference',
    'correct_quantile_residuals',
    'generate_bootstrap_response',
    'bootstrap_sample',
    'compute_prediction_bands',
    
    # Visualizations
    'plot_pqrm_fit',
    'plot_pqrm_with_bootstrap_ci',
    'plot_residual_diagnostics',
    'plot_bootstrap_distribution',
    'plot_multiple_quantiles',
]
