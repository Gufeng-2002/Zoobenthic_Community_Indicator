"""
Core Piecewise Quantile Regression Model (PQRM) implementation.

This module provides functions for fitting piecewise quantile regression models
with automatic breakpoint estimation via optimization.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from typing import Tuple, Optional, List, Union


def fit_pqrm_with_breakpoints(
    data: pd.DataFrame,
    breakpoints: Union[List[float], np.ndarray],
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    max_iter: int = 5000
) -> Tuple[object, pd.DataFrame]:
    """
    Fit a piecewise quantile regression model with specified breakpoints.
    
    This function creates hinge features at each breakpoint and fits a quantile
    regression model with statsmodels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing predictor and response variables
    breakpoints : list or array
        Breakpoint locations (alpha values) for the piecewise model
    x_col : str, default='x'
        Name of the predictor variable column
    y_col : str, default='y_continuous'
        Name of the response variable column
    tau : float, default=0.5
        Quantile level for regression (0.5 = median)
    max_iter : int, default=5000
        Maximum iterations for quantile regression solver
    
    Returns
    -------
    result : statsmodels result object
        Fitted quantile regression model
    data_with_breaks : pd.DataFrame
        Data with added breakpoint hinge features (x_break1, x_break2, ...)
    
    Notes
    -----
    For each breakpoint alpha_i, creates a hinge feature:
        x_break_i = max(x - alpha_i, 0)
    
    The model is: Q_tau(y|x) = beta_0 + beta_1*x + sum_i(gamma_i * x_break_i)
    """
    data_with_breaks = data.copy()
    
    # Create hinge features for each breakpoint
    formula_parts = []
    for i, alpha in enumerate(breakpoints):
        break_col = f'{x_col}_break{i+1}'
        data_with_breaks[break_col] = np.maximum(data_with_breaks[x_col] - alpha, 0)
        formula_parts.append(break_col)
    
    # Build formula string
    formula_x = ' + '.join(formula_parts)
    formula = f'{y_col} ~ {x_col} + {formula_x}'
    
    # Fit quantile regression
    model = smf.quantreg(formula, data=data_with_breaks)
    result = model.fit(q=tau, max_iter=max_iter)
    
    return result, data_with_breaks


def quantile_check_loss(
    breakpoints: Union[List[float], np.ndarray],
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    max_iter: int = 5000
) -> float:
    """
    Compute quantile check loss for a given set of breakpoints.
    
    This is the objective function minimized during breakpoint optimization.
    The quantile check loss is: sum(rho_tau(residuals)) where
    rho_tau(u) = u * (tau - I(u < 0))
    
    Parameters
    ----------
    breakpoints : list or array
        Candidate breakpoint locations
    data : pd.DataFrame
        Input data
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    tau : float, default=0.5
        Quantile level
    max_iter : int, default=5000
        Maximum iterations for solver
    
    Returns
    -------
    loss : float
        Quantile check loss value
    """
    # Fit model with current breakpoints
    result, data_with_breaks = fit_pqrm_with_breakpoints(
        data, breakpoints, x_col, y_col, tau, max_iter
    )
    
    # Compute quantile check loss
    residuals = data_with_breaks[y_col] - result.predict(data_with_breaks)
    loss = np.sum(np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals))
    
    return loss


def fit_pqrm_pipeline(
    data: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    max_breakpoints: int = 2,
    initial_guess: Optional[Union[List[float], np.ndarray]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_iter: int = 5000,
    verbose: bool = False
) -> Tuple[object, np.ndarray, pd.DataFrame]:
    """
    Complete pipeline for fitting PQRM with automatic breakpoint estimation.
    
    This is the main entry point for PQRM fitting. It optimizes breakpoint
    locations by minimizing the quantile check loss, then fits the final model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with predictor and response columns
    x_col : str, default='x'
        Name of predictor variable
    y_col : str, default='y_continuous'
        Name of response variable
    max_breakpoints : int, default=2
        Number of breakpoints to estimate
    tau : float, default=0.5
        Quantile level (0 < tau < 1)
    initial_guess : array-like, optional
        Initial breakpoint locations. If None, uses evenly spaced values
    bounds : list of tuples, optional
        Bounds for each breakpoint (min, max). If None, splits x range into segments
    max_iter : int, default=5000
        Maximum iterations for quantile regression solver
    verbose : bool, default=False
        Whether to print optimization progress
    
    Returns
    -------
    result : statsmodels result object
        Fitted PQRM model
    optimal_breakpoints : np.ndarray
        Optimized breakpoint locations
    data_with_breaks : pd.DataFrame
        Data with breakpoint hinge features added
    
    Examples
    --------
    >>> data = pd.DataFrame({'x': x_values, 'y_continuous': y_values})
    >>> result, breakpoints, data_breaks = fit_pqrm_pipeline(
    ...     data, tau=0.5, max_breakpoints=2
    ... )
    >>> print(f"Breakpoints: {breakpoints}")
    >>> predictions = result.predict(data_breaks)
    
    Notes
    -----
    - Uses scipy.optimize.minimize with L-BFGS-B method
    - Breakpoints are constrained to lie within the x data range
    - Each breakpoint has its own bounds to ensure proper ordering
    """
    # Set initial guess if not provided
    if initial_guess is None:
        x_min, x_max = data[x_col].min(), data[x_col].max()
        initial_guess = np.linspace(x_min, x_max, max_breakpoints + 2)[1:-1]
    else:
        initial_guess = np.array(initial_guess)
    
    # Set bounds if not provided
    if bounds is None:
        x_min, x_max = float(data[x_col].min()), float(data[x_col].max())
        segment_length = (x_max - x_min) / (max_breakpoints + 1)
        bounds = [
            (x_min + i * segment_length, x_min + (i + 1) * segment_length)
            for i in range(1, max_breakpoints + 1)
        ]
    
    # Optimize breakpoints
    if verbose:
        print(f"Optimizing {max_breakpoints} breakpoints...")
        print(f"Initial guess: {initial_guess}")
    
    optimization_result = minimize(
        quantile_check_loss,
        x0=initial_guess,
        args=(data, x_col, y_col, tau, max_iter),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimal_breakpoints = optimization_result.x
    
    if verbose:
        print(f"Optimal breakpoints: {optimal_breakpoints}")
        print(f"Optimization success: {optimization_result.success}")
        print(f"Final loss: {optimization_result.fun:.4f}")
    
    # Fit final model with optimal breakpoints
    result, data_with_breaks = fit_pqrm_with_breakpoints(
        data, optimal_breakpoints, x_col, y_col, tau, max_iter
    )
    
    return result, optimal_breakpoints, data_with_breaks


class PiecewiseQuantileRegression:
    """
    Scikit-learn style wrapper for Piecewise Quantile Regression Model.
    
    This class provides a convenient interface for fitting and predicting with
    piecewise quantile regression models, with automatic breakpoint estimation.
    
    Parameters
    ----------
    tau : float, default=0.5
        Quantile level for regression
    max_breakpoints : int, default=2
        Number of breakpoints to estimate
    initial_guess : array-like, optional
        Initial breakpoint locations
    bounds : list of tuples, optional
        Bounds for breakpoint optimization
    max_iter : int, default=5000
        Maximum iterations for quantile regression
    verbose : bool, default=False
        Whether to print fitting progress
    
    Attributes
    ----------
    breakpoints_ : np.ndarray
        Fitted breakpoint locations
    model_ : statsmodels result object
        Fitted quantile regression model
    x_col_ : str
        Name of predictor variable used in fitting
    y_col_ : str
        Name of response variable used in fitting
    is_fitted_ : bool
        Whether the model has been fitted
    
    Examples
    --------
    >>> pqrm = PiecewiseQuantileRegression(tau=0.5, max_breakpoints=2)
    >>> pqrm.fit(data, x_col='x', y_col='y_continuous')
    >>> predictions = pqrm.predict(new_data)
    >>> print(pqrm.breakpoints_)
    """
    
    def __init__(
        self,
        tau: float = 0.5,
        max_breakpoints: int = 2,
        initial_guess: Optional[Union[List[float], np.ndarray]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_iter: int = 5000,
        verbose: bool = False
    ):
        self.tau = tau
        self.max_breakpoints = max_breakpoints
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Attributes set during fitting
        self.breakpoints_ = None
        self.model_ = None
        self.x_col_ = None
        self.y_col_ = None
        self.is_fitted_ = False
    
    def fit(
        self,
        data: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y_continuous'
    ) -> 'PiecewiseQuantileRegression':
        """
        Fit the piecewise quantile regression model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        x_col : str, default='x'
            Name of predictor variable
        y_col : str, default='y_continuous'
            Name of response variable
        
        Returns
        -------
        self : PiecewiseQuantileRegression
            Fitted model instance
        """
        self.x_col_ = x_col
        self.y_col_ = y_col
        
        result, breakpoints, self.data_with_breaks_ = fit_pqrm_pipeline(
            data=data,
            x_col=x_col,
            y_col=y_col,
            tau=self.tau,
            max_breakpoints=self.max_breakpoints,
            initial_guess=self.initial_guess,
            bounds=self.bounds,
            max_iter=self.max_iter,
            verbose=self.verbose
        )
        
        self.model_ = result
        self.breakpoints_ = breakpoints
        self.is_fitted_ = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with predictor variable
        
        Returns
        -------
        predictions : np.ndarray
            Predicted quantile values
        
        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Add breakpoint features to prediction data
        data_with_breaks = data.copy()
        for i, alpha in enumerate(self.breakpoints_):
            break_col = f'{self.x_col_}_break{i+1}'
            data_with_breaks[break_col] = np.maximum(data_with_breaks[self.x_col_] - alpha, 0)
        
        predictions = self.model_.predict(data_with_breaks)
        return predictions.values
    
    def get_residuals(self) -> np.ndarray:
        """
        Get residuals from fitted model.
        
        Returns
        -------
        residuals : np.ndarray
            Residuals (y - y_hat)
        
        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting residuals. Call fit() first.")
        
        return self.model_.resid
    
    def summary(self) -> str:
        """
        Get model summary.
        
        Returns
        -------
        summary : str
            Model summary from statsmodels
        
        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting summary. Call fit() first.")
        
        return self.model_.summary()
