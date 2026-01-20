# Piecewise Quantile Regression Model (PQRM) Module

A comprehensive Python module for fitting piecewise quantile regression models with automatic breakpoint estimation and bootstrap inference for uncertainty quantification.

## Overview

This module provides tools for fitting **Piecewise Quantile Regression Models (PQRM)**, which extend standard quantile regression by allowing relationships to change at unknown breakpoints. This is particularly useful in ecological and environmental studies for:

- Detecting thresholds in ecosystem responses
- Identifying regime shifts in ecological data
- Modeling non-linear relationships with distinct phases
- Quantifying uncertainty in threshold locations

## Features

- 🔧 **Automatic breakpoint estimation** via optimization
- 📊 **Bootstrap inference** for confidence intervals
- 🎯 **Bias correction** for quantile regression residuals
- 📈 **Comprehensive visualizations** with matplotlib
- 🔄 **Scikit-learn style API** (fit/predict interface)
- 🎨 **Multiple quantile comparison** tools
- 📦 **Well-documented** with examples

## Installation

The module is part of the `zci` package. Make sure you have the required dependencies:

```bash
pip install numpy pandas scipy statsmodels matplotlib
```

## Quick Start

### Basic Usage

```python
from zci.piecewise_quantile_regression import pqrm_analysis_pipeline
import pandas as pd

# Prepare your data
data = pd.DataFrame({
    'x': x_values,
    'y_continuous': y_values
})

# Run complete analysis
results = pqrm_analysis_pipeline(
    data,
    x_col='x',
    y_col='y_continuous',
    tau=0.5,  # Median regression
    max_breakpoints=2,
    n_bootstrap=200,
    confidence_level=0.95
)

# Access results
print(f"Breakpoints: {results['breakpoints']}")
print(f"Breakpoint CIs: {results['breakpoint_ci']}")

# Show plot
results['figures']['main_plot'].show()
```

### Using Individual Components

```python
from zci.piecewise_quantile_regression import (
    fit_pqrm_pipeline,
    bootstrap_pqrm_inference,
    plot_pqrm_with_bootstrap_ci
)

# Step 1: Fit the model
model, breakpoints, data_with_breaks = fit_pqrm_pipeline(
    data, tau=0.5, max_breakpoints=2
)

# Step 2: Bootstrap inference
boot_results = bootstrap_pqrm_inference(
    model, data_with_breaks, n_bootstrap=200
)

# Step 3: Visualize
fig, ax = plot_pqrm_with_bootstrap_ci(
    data, data_with_breaks, model, breakpoints, boot_results
)
```

### Scikit-learn Style Interface

```python
from zci.piecewise_quantile_regression import PiecewiseQuantileRegression

# Create and fit model
pqrm = PiecewiseQuantileRegression(tau=0.5, max_breakpoints=2)
pqrm.fit(data, x_col='x', y_col='y_continuous')

# Make predictions
predictions = pqrm.predict(new_data)

# Get breakpoints
print(pqrm.breakpoints_)
```

## Module Structure

```
piecewise_quantile_regression/
├── __init__.py           # Main imports and documentation
├── pqrm_model.py         # Core model fitting functions
├── bootstrap.py          # Bootstrap inference methods
├── visualizations.py     # Plotting functions
└── pipeline.py           # High-level orchestration
```

## Main Functions

### Pipeline Function (Recommended)

**`pqrm_analysis_pipeline()`** - Complete end-to-end analysis

```python
results = pqrm_analysis_pipeline(
    data,
    x_col='x',
    y_col='y_continuous',
    tau=0.5,
    max_breakpoints=2,
    run_bootstrap=True,
    n_bootstrap=100,
    confidence_level=0.95,
    create_visualizations=True
)
```

### Core Functions

- **`fit_pqrm_pipeline()`** - Fit PQRM with breakpoint optimization
- **`fit_pqrm_with_breakpoints()`** - Fit with specified breakpoints
- **`bootstrap_pqrm_inference()`** - Bootstrap confidence intervals
- **`correct_quantile_residuals()`** - Bias correction for residuals

### Visualization Functions

- **`plot_pqrm_fit()`** - Basic PQRM fit plot
- **`plot_pqrm_with_bootstrap_ci()`** - Plot with confidence intervals
- **`plot_residual_diagnostics()`** - Residual diagnostic plots
- **`plot_bootstrap_distribution()`** - Bootstrap distribution plots
- **`plot_multiple_quantiles()`** - Compare multiple quantiles

### Utility Functions

- **`fit_multiple_quantiles()`** - Fit PQRM for multiple tau values
- **`compare_quantile_breakpoints()`** - Compare breakpoints across quantiles

## Examples

### Example 1: Median Regression with Bootstrap

```python
results = pqrm_analysis_pipeline(
    data,
    tau=0.5,
    max_breakpoints=2,
    initial_guess=[0.3, 0.6],
    bounds=[(0.2, 0.4), (0.5, 0.7)],
    n_bootstrap=200,
    random_state=42
)

# Save figure
results['figures']['main_plot'].savefig('pqrm_fit.png', dpi=300)
```

### Example 2: Multiple Quantiles

```python
from zci.piecewise_quantile_regression import fit_multiple_quantiles, plot_multiple_quantiles

# Fit multiple quantiles
multi_results = fit_multiple_quantiles(
    data,
    taus=[0.1, 0.25, 0.5, 0.75, 0.9],
    max_breakpoints=2
)

# Prepare for plotting
models_dict = {
    tau: (res['model'], res['breakpoints'], res['data_with_breaks'])
    for tau, res in multi_results.items()
}

# Visualize
fig, ax = plot_multiple_quantiles(data, models_dict)
plt.show()
```

### Example 3: Custom Analysis

```python
# Fit with custom settings
model, breakpoints, data_breaks = fit_pqrm_pipeline(
    data,
    tau=0.1,  # 10th percentile
    max_breakpoints=3,  # 3 breakpoints
    initial_guess=[0.2, 0.5, 0.8],
    max_iter=10000,
    verbose=True
)

# Bootstrap with bias correction
boot_results = bootstrap_pqrm_inference(
    model,
    data_breaks,
    n_bootstrap=500,
    use_bias_correction=True,
    confidence_level=0.95
)

# Custom visualization
from zci.piecewise_quantile_regression import plot_pqrm_with_bootstrap_ci
fig, ax = plot_pqrm_with_bootstrap_ci(
    data, data_breaks, model, breakpoints, boot_results,
    title='10th Percentile PQRM with 95% CI'
)
```

## Model Details

### The PQRM Equation

The piecewise quantile regression model is defined as:

```
Q_τ(y|x) = β₀ + β₁x + Σᵢ γᵢ · max(x - αᵢ, 0)
```

Where:
- `Q_τ(y|x)` is the τ-th conditional quantile of y given x
- `αᵢ` are the breakpoint locations (estimated)
- `max(x - αᵢ, 0)` creates "hinge" functions at each breakpoint
- `β` and `γ` are regression coefficients

### Breakpoint Estimation

Breakpoints are estimated by minimizing the quantile check loss:

```
L(α) = Σ ρ_τ(yᵢ - Q_τ(yᵢ|xᵢ, α))
```

where `ρ_τ(u) = u(τ - I(u < 0))` is the check loss function.

### Bootstrap Inference

The module uses wild bootstrap with bias-corrected residuals:

1. Fit base PQRM model
2. Extract and bias-correct residuals
3. Bootstrap resample residuals
4. Add resampled residuals to fitted values
5. Fit new PQRM to bootstrap data
6. Repeat B times to get empirical distributions

## Parameters

### Key Parameters

- **`tau`** (float): Quantile level (0 < tau < 1)
  - 0.5 = median regression
  - 0.1 = 10th percentile
  - 0.9 = 90th percentile

- **`max_breakpoints`** (int): Number of breakpoints to estimate

- **`initial_guess`** (array-like): Starting values for breakpoint optimization

- **`bounds`** (list of tuples): (min, max) bounds for each breakpoint

- **`n_bootstrap`** (int): Number of bootstrap iterations
  - 100-200: Quick exploratory analysis
  - 500-1000: Publication-quality results

- **`confidence_level`** (float): Confidence level for intervals (default: 0.95)

- **`use_bias_correction`** (bool): Apply bias correction to residuals (default: True)

## Output Structure

The `pqrm_analysis_pipeline()` returns a dictionary with:

```python
{
    'model': statsmodels quantile regression result,
    'breakpoints': np.ndarray of breakpoint locations,
    'data_with_breaks': pd.DataFrame with hinge features,
    'predictions': model predictions,
    'residuals': model residuals,
    'bootstrap_results': full bootstrap inference dict,
    'prediction_ci_lower': lower CI for predictions,
    'prediction_ci_upper': upper CI for predictions,
    'breakpoint_ci': list of (lower, upper) for each breakpoint,
    'figures': dict of matplotlib figures
}
```

## Testing

A comprehensive test notebook is provided: `notebooks/07_PQRM_Module_Test.ipynb`

Run the tests:
```python
# In Jupyter
%run notebooks/07_PQRM_Module_Test.ipynb
```

## Performance Tips

1. **Bootstrap iterations**: Start with 50-100 for quick testing, use 500+ for final results
2. **Initial guesses**: Provide good initial guesses to speed up optimization
3. **Bounds**: Constrain breakpoint bounds to reasonable ranges
4. **Max iterations**: Increase if optimization doesn't converge (default: 5000)

## Dependencies

- `numpy >= 1.20`
- `pandas >= 1.3`
- `scipy >= 1.7`
- `statsmodels >= 0.13`
- `matplotlib >= 3.4`

## References

1. Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

2. Muggeo, V. M. (2003). Estimating regression models with unknown break-points. *Statistics in Medicine*, 22(19), 3055-3071.

3. Chernozhukov, V., & Hansen, C. (2008). Instrumental variable quantile regression: A robust inference approach. *Journal of Econometrics*, 142(1), 379-398.

## Citation

If you use this module in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information]

## Authors

ZCI Research Team

## Support

For issues or questions:
- Check the test notebook for examples
- Review the docstrings in each module
- Open an issue on the project repository

---

**Note**: This module is designed to integrate seamlessly with other `zci` modules for comprehensive ecological data analysis.
