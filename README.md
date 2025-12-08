# Zoobenthic Community Indicators (ZCI) - Project Code

> **Project background:**  
> This repository is part of **Feng Gu’s Master’s thesis project** —  
> *“Zoobenthic Community Indicator of Sediment Contamination.”*  
> A complete research proposal can be found [here](https://drive.google.com/file/d/1o4wm0Ox7t6uW84AtK3pRNr8zHJ5kNsW7/view?usp=drive_link).

A Python package for analyzing zoobenthic community indicators of sediment contamination using multivariate statistical methods.

## Project Structure

```text
Project_Code/
├── src/zci/                    # Main Python package
├── notebooks/                  # Jupyter notebooks for analysis
├── data/                       # Data storage (raw, processed, interim)
├── artifacts/                  # Generated outputs and results
├── results/                    # Final analysis results
└── pyproject.toml             # Package configuration
```


## Notebooks

Interactive analysis workflows:

- **`00*_build_data_operation.ipynb`** - Data loading, cleaning, and preprocessing pipeline
- **`01_weighted_PCA_scores.ipynb`** - Weighted PCA analysis and pollution scoring
- **`02_ordination_metrices.ipynb`** - Ordination analysis and distance metrics


## ZCI Package (`src/zci/`)

The core Python package organized into three main modules:

### `data_process/`

Core data manipulation and transformation utilities:

- **`dataframe_ops.py`** - Multi-index DataFrame operations, data alignment, and merging
- **`transform.py`** - Data transformations (Hellinger, log1p standardization)

### `sediment_pollution_assessment/`

Statistical methods for pollution assessment:

- **`weighted_pca.py`** - Weighted Principal Component Analysis implementation
- **`chemical_weights.py`** - Chemical variable weighting schemes
- **`ordination_metrices.py`** - Ordination distance metrics and evaluation

### `environmental_partition_cluster/`

Environmental clustering and partitioning methods (under development)

## Data Organization

```text
data/
├── raw/                       # Original datasets
├── interim/                   # Intermediate processed data
├── processed/                 # Final processed datasets
├── maps/                      # Spatial data (shapefiles, etc.)
└── data_documents/           # Data documentation
```

## Key Features

- **Multi-index DataFrame Management** - Efficient handling of complex ecological datasets
- **Weighted PCA Analysis** - Priority-based principal component analysis for pollution assessment
- **Data Transformation Pipeline** - Standardized ecological data transformations
- **Ordination Methods** - Multiple distance metrics and ordination techniques
- **Reproducible Workflows** - Jupyter notebook-based analysis pipelines

## Research Context

This analysis supports research on:

- Zoobenthic community structure analysis
- Sediment contamination assessment
- Environmental gradient detection
- Multivariate ecological statistics

---

*For detailed usage examples, see the notebook collection. For function documentation, refer to docstrings in the source code.*
