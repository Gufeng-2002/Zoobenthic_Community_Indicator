# PQRM Module - Quick Reference Guide

## Module Location
```
src/zci/piecewise_quantile_regression/
├── __init__.py           # Main module interface
├── pqrm_model.py         # Core fitting functions
├── bootstrap.py          # Bootstrap inference
├── visualizations.py     # Plotting functions
├── pipeline.py           # High-level pipeline
└── README.md             # Full documentation
```

## Quick Import

```python
# Recommended: Use the pipeline function
from zci.piecewise_quantile_regression import pqrm_analysis_pipeline

# Or import individual components
from zci.piecewise_quantile_regression import (
    fit_pqrm_pipeline,
    bootstrap_pqrm_inference,
    plot_pqrm_with_bootstrap_ci,
    PiecewiseQuantileRegression
)
```

## Common Use Cases

### 1. Quick Analysis (One-liner)

```python
results = pqrm_analysis_pipeline(data, tau=0.5, max_breakpoints=2, n_bootstrap=200)
print(f"Breakpoints: {results['breakpoints']}")
results['figures']['main_plot'].show()
```

### 2. Step-by-Step Analysis

```python
# Step 1: Fit model
model, breakpoints, data_breaks = fit_pqrm_pipeline(data, tau=0.5)

# Step 2: Bootstrap inference
boot_results = bootstrap_pqrm_inference(model, data_breaks, n_bootstrap=200)

# Step 3: Visualize
fig, ax = plot_pqrm_with_bootstrap_ci(data, data_breaks, model, breakpoints, boot_results)
```

### 3. Multiple Quantiles

```python
multi_results = fit_multiple_quantiles(data, taus=[0.1, 0.5, 0.9])
models_dict = {tau: (r['model'], r['breakpoints'], r['data_with_breaks']) 
               for tau, r in multi_results.items()}
fig, ax = plot_multiple_quantiles(data, models_dict)
```

### 4. Scikit-learn Style

```python
pqrm = PiecewiseQuantileRegression(tau=0.5, max_breakpoints=2)
pqrm.fit(data, x_col='x', y_col='y_continuous')
predictions = pqrm.predict(new_data)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.5 | Quantile level (0.1=10th, 0.5=median, 0.9=90th) |
| `max_breakpoints` | 2 | Number of breakpoints to estimate |
| `n_bootstrap` | 100 | Bootstrap iterations (100-200 for testing, 500+ for publication) |
| `confidence_level` | 0.95 | Confidence level for intervals |
| `initial_guess` | None | Starting values for breakpoints |
| `bounds` | None | (min, max) bounds for each breakpoint |
| `use_bias_correction` | True | Apply bias correction to residuals |

## Return Values

### From `pqrm_analysis_pipeline()`

```python
results = {
    'model': fitted statsmodels result,
    'breakpoints': np.array of breakpoint locations,
    'data_with_breaks': DataFrame with hinge features,
    'predictions': model predictions,
    'residuals': model residuals,
    'bootstrap_results': full bootstrap dict,
    'prediction_ci_lower': lower CI for predictions,
    'prediction_ci_upper': upper CI for predictions,
    'breakpoint_ci': [(lower, upper), ...] for each breakpoint,
    'figures': {'main_plot': fig, 'residuals': fig}
}
```

## Examples by Use Case

### Ecological Threshold Detection

```python
# Detect thresholds in species response to pollution
results = pqrm_analysis_pipeline(
    data,
    x_col='pollution_concentration',
    y_col='species_diversity',
    tau=0.5,
    max_breakpoints=2,
    n_bootstrap=500,
    confidence_level=0.95
)
```

### Regime Shift Analysis

```python
# Identify regime shifts in time series
results = pqrm_analysis_pipeline(
    data,
    x_col='time',
    y_col='ecosystem_state',
    tau=0.5,
    max_breakpoints=3,
    initial_guess=[1990, 2000, 2010],
    n_bootstrap=300
)
```

### Lower Tail Analysis

```python
# Analyze 10th percentile (conservative estimates)
results = pqrm_analysis_pipeline(
    data,
    tau=0.1,
    max_breakpoints=2,
    n_bootstrap=200
)
```

### Upper Tail Analysis

```python
# Analyze 90th percentile (optimistic estimates)
results = pqrm_analysis_pipeline(
    data,
    tau=0.9,
    max_breakpoints=2,
    n_bootstrap=200
)
```

## Troubleshooting

### Issue: Optimization doesn't converge
**Solution**: 
- Increase `max_iter` (default: 5000)
- Provide better `initial_guess`
- Constrain `bounds` more tightly

### Issue: Bootstrap takes too long
**Solution**:
- Reduce `n_bootstrap` for exploratory analysis
- Use `verbose=False` to reduce output
- Consider using fewer data points for initial testing

### Issue: Breakpoints at boundaries
**Solution**:
- Adjust `bounds` to allow wider search
- Check if data actually has breakpoints
- Try different `initial_guess` values

### Issue: Wide confidence intervals
**Solution**:
- Increase `n_bootstrap`
- Check if data has sufficient signal
- Verify breakpoints are identifiable

## Performance Tips

1. **Start small**: Use `n_bootstrap=50` for initial exploration
2. **Good initial guesses**: Plot data first, estimate breakpoints visually
3. **Constrain bounds**: Use tight bounds if you know approximate locations
4. **Parallel processing**: Future feature - stay tuned!

## Testing Notebook

Run the comprehensive test notebook:
```
notebooks/07_PQRM_Module_Test.ipynb
```

## Related Modules in ZCI

- `contamination_assessment`: PCA-based pollution scoring
- `env_driven_taxa_clusters`: LDA-based habitat separation
- `community_composition_measures`: Ecological diversity metrics

## Need Help?

1. Check the full README: `src/zci/piecewise_quantile_regression/README.md`
2. Review docstrings: `help(pqrm_analysis_pipeline)`
3. Run test notebook: `notebooks/07_PQRM_Module_Test.ipynb`
4. See notebook 06: `notebooks/06_PQRM_Spec_Pollution.ipynb`
