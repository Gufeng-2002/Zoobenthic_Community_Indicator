"""
Visualization functions for Piecewise Quantile Regression Models.

This module provides functions for visualizing:
1. PQRM fits with data
2. Bootstrap confidence intervals
3. Breakpoint uncertainty regions
4. Residual diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_pqrm_fit(
    data: pd.DataFrame,
    fitted_model: object,
    breakpoints: np.ndarray,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_breakpoints: bool = True,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Plot PQRM fitted line with original data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with breakpoint features
    fitted_model : statsmodels result object
        Fitted PQRM model
    breakpoints : np.ndarray
        Estimated breakpoint locations
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    tau : float, default=0.5
        Quantile level
    figsize : tuple, default=(10, 6)
        Figure size
    dpi : int, default=300
        Figure DPI
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    show_breakpoints : bool, default=True
        Whether to show vertical lines at breakpoints
    ax : matplotlib Axes, optional
        Existing axes to plot on
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()
    
    # Scatter plot of data
    ax.scatter(
        data[x_col], data[y_col],
        alpha=0.6, s=50,
        color='steelblue', edgecolor='black', linewidth=0.5,
        label='Observed Data'
    )
    
    # Create smooth prediction line
    x_min, x_max = data[x_col].min(), data[x_col].max()
    x_smooth = np.linspace(x_min, x_max, 500)
    
    # Build prediction dataframe with breakpoint features
    pred_data = pd.DataFrame({x_col: x_smooth})
    for i, alpha in enumerate(breakpoints):
        pred_data[f'{x_col}_break{i+1}'] = np.maximum(x_smooth - alpha, 0)
    
    y_pred = fitted_model.predict(pred_data)
    
    ax.plot(
        x_smooth, y_pred,
        color='red', linewidth=2.5,
        label=f'PQRM Fit (τ={tau})'
    )
    
    # Show breakpoints
    if show_breakpoints:
        y_min, y_max = ax.get_ylim()
        for i, bp in enumerate(breakpoints):
            ax.axvline(
                bp, color='orange', linestyle='--',
                linewidth=1.5, alpha=0.7,
                label=f'Breakpoint {i+1}: {bp:.3f}' if i < 3 else None
            )
    
    # Labels and title
    ax.set_xlabel(xlabel or x_col, fontsize=12)
    ax.set_ylabel(ylabel or y_col, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Piecewise Quantile Regression (τ={tau})', fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig, ax


def plot_pqrm_with_bootstrap_ci(
    data: pd.DataFrame,
    data_with_breaks: pd.DataFrame,
    fitted_model: object,
    breakpoints: np.ndarray,
    bootstrap_results: Dict,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 300,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_breakpoint_ci: bool = True,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Plot PQRM fit with bootstrap confidence intervals.
    
    This creates a comprehensive visualization showing:
    - Original data points
    - Fitted PQRM line
    - Bootstrap confidence bands for predictions
    - Bootstrap confidence regions for breakpoints (optional)
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data without breakpoint features
    data_with_breaks : pd.DataFrame
        Data with breakpoint features
    fitted_model : statsmodels result object
        Fitted PQRM model
    breakpoints : np.ndarray
        Estimated breakpoints
    bootstrap_results : dict
        Results from bootstrap_pqrm_inference()
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    tau : float, default=0.5
        Quantile level
    figsize : tuple, default=(12, 7)
        Figure size
    dpi : int, default=300
        Figure DPI
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    show_breakpoint_ci : bool, default=True
        Whether to show breakpoint confidence regions
    ax : matplotlib Axes, optional
        Existing axes
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()
    
    confidence_level = bootstrap_results['confidence_level']
    n_bootstrap = bootstrap_results['n_bootstrap']
    
    # Scatter plot of data
    ax.scatter(
        data[x_col], data[y_col],
        alpha=0.6, s=50,
        color='green', edgecolor='black', linewidth=0.5,
        label='Observed Data', zorder=3
    )
    
    # Fitted line
    x_sorted = np.sort(data_with_breaks[x_col])
    x_break_sorted = [
        np.sort(data_with_breaks[f'{x_col}_break{i+1}'])
        for i in range(len(breakpoints))
    ]
    
    pred_dict = {x_col: x_sorted}
    for i in range(len(breakpoints)):
        pred_dict[f'{x_col}_break{i+1}'] = x_break_sorted[i]
    
    y_pred = fitted_model.predict(pred_dict)
    
    ax.plot(
        x_sorted, y_pred,
        color='blue', linewidth=2.5,
        label='PQRM Fit', zorder=4
    )
    
    # Confidence intervals for predictions
    ci_lower = bootstrap_results['prediction_ci_lower']
    ci_upper = bootstrap_results['prediction_ci_upper']
    
    ax.fill_between(
        data_with_breaks[x_col].sort_values(),
        ci_lower.sort_index(),
        ci_upper.sort_index(),
        color='lightgray', alpha=0.5,
        label=f'{confidence_level*100:.0f}% CI (Predictions)',
        zorder=1
    )
    
    ax.plot(
        data_with_breaks[x_col].sort_values(),
        ci_lower.sort_index(),
        color='red', linestyle='--', linewidth=1, zorder=2
    )
    ax.plot(
        data_with_breaks[x_col].sort_values(),
        ci_upper.sort_index(),
        color='red', linestyle='--', linewidth=1, zorder=2
    )
    
    # Breakpoint confidence regions
    if show_breakpoint_ci:
        breakpoint_ci = bootstrap_results['breakpoint_ci']
        colors = ['orange', 'purple', 'brown', 'pink']
        
        for i, (bp_lower, bp_upper) in enumerate(breakpoint_ci):
            color = colors[i % len(colors)]
            ax.axvspan(
                bp_lower, bp_upper,
                color=color, alpha=0.25,
                label=f'{confidence_level*100:.0f}% CI (α_{i+1})',
                zorder=0
            )
    
    # Labels and title
    ax.set_xlabel(xlabel or x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel or y_col, fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(
            f'PQRM (τ={tau}) with Bootstrap Inference (B={n_bootstrap})',
            fontsize=14, fontweight='bold'
        )
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig, ax


def plot_residual_diagnostics(
    fitted_model: object,
    data_with_breaks: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    tau: float = 0.5,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 200
) -> Tuple[Figure, np.ndarray]:
    """
    Create residual diagnostic plots for PQRM.
    
    Creates three panels:
    1. Residuals vs fitted values
    2. Residuals vs predictor
    3. Histogram of residuals
    
    Parameters
    ----------
    fitted_model : statsmodels result object
        Fitted PQRM model
    data_with_breaks : pd.DataFrame
        Data with breakpoint features
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    tau : float, default=0.5
        Quantile level
    figsize : tuple, default=(14, 5)
        Figure size
    dpi : int, default=200
        Figure DPI
    
    Returns
    -------
    fig : matplotlib Figure
    axes : np.ndarray of Axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    residuals = fitted_model.resid
    fitted_values = fitted_model.predict(data_with_breaks)
    x_values = data_with_breaks[x_col]
    
    # 1. Residuals vs Fitted
    axes[0].scatter(fitted_values, residuals, alpha=0.6, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Fitted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predictor
    axes[1].scatter(x_values, residuals, alpha=0.6, s=30, color='green', edgecolor='black', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel(x_col, fontsize=11)
    axes[1].set_ylabel('Residuals', fontsize=11)
    axes[1].set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Histogram of Residuals
    axes[2].hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Residuals', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f'Residual Distribution (τ={tau})', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig, axes


def plot_bootstrap_distribution(
    bootstrap_results: Dict,
    breakpoint_idx: int = 0,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 200
) -> Tuple[Figure, Axes]:
    """
    Plot the bootstrap distribution for a specific breakpoint.
    
    Parameters
    ----------
    bootstrap_results : dict
        Results from bootstrap_pqrm_inference()
    breakpoint_idx : int, default=0
        Which breakpoint to visualize (0-indexed)
    figsize : tuple, default=(10, 5)
        Figure size
    dpi : int, default=200
        Figure DPI
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    bootstrap_breakpoints = bootstrap_results['bootstrap_breakpoints']
    breakpoint_values = bootstrap_breakpoints.loc[breakpoint_idx, :]
    
    confidence_level = bootstrap_results['confidence_level']
    ci_lower = bootstrap_results['breakpoint_ci_lower'].iloc[breakpoint_idx]
    ci_upper = bootstrap_results['breakpoint_ci_upper'].iloc[breakpoint_idx]
    
    # Histogram
    ax.hist(
        breakpoint_values, bins=30,
        alpha=0.7, color='steelblue',
        edgecolor='black', density=True,
        label='Bootstrap Distribution'
    )
    
    # CI lines
    ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'{confidence_level*100:.0f}% CI Lower')
    ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'{confidence_level*100:.0f}% CI Upper')
    
    # Mean
    mean_val = breakpoint_values.mean()
    ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label='Bootstrap Mean')
    
    ax.set_xlabel(f'Breakpoint α_{breakpoint_idx+1}', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        f'Bootstrap Distribution for Breakpoint {breakpoint_idx+1}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig, ax


def plot_multiple_quantiles(
    data: pd.DataFrame,
    models_dict: Dict[float, Tuple],
    x_col: str = 'x',
    y_col: str = 'y_continuous',
    figsize: Tuple[int, int] = (12, 7),
    dpi: int = 300,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot multiple PQRM fits for different quantile levels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    models_dict : dict
        Dictionary mapping tau -> (fitted_model, breakpoints, data_with_breaks)
    x_col : str, default='x'
        Predictor variable name
    y_col : str, default='y_continuous'
        Response variable name
    figsize : tuple, default=(12, 7)
        Figure size
    dpi : int, default=300
        Figure DPI
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Scatter plot of data
    ax.scatter(
        data[x_col], data[y_col],
        alpha=0.4, s=40,
        color='gray', edgecolor='black', linewidth=0.5,
        label='Observed Data', zorder=1
    )
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (tau, (model, breakpoints, data_breaks)) in enumerate(sorted(models_dict.items())):
        color = colors[i % len(colors)]
        
        # Create prediction line
        x_sorted = np.sort(data_breaks[x_col])
        x_break_sorted = [
            np.sort(data_breaks[f'{x_col}_break{j+1}'])
            for j in range(len(breakpoints))
        ]
        
        pred_dict = {x_col: x_sorted}
        for j in range(len(breakpoints)):
            pred_dict[f'{x_col}_break{j+1}'] = x_break_sorted[j]
        
        y_pred = model.predict(pred_dict)
        
        ax.plot(
            x_sorted, y_pred,
            color=color, linewidth=2.5,
            label=f'τ={tau}', zorder=2+i
        )
    
    ax.set_xlabel(xlabel or x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel or y_col, fontsize=12, fontweight='bold')
    ax.set_title(
        title or 'PQRM Fits for Multiple Quantiles',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return fig, ax
