# Zoobenthic Community Indicators (ZCI) - Project Code

A Python package for analyzing zoobenthic community indicators of sediment contamination using multivariate statistical methods.

## 🏗️ Project Structure

```text
Project_Code/
├── src/zci/                    # Main Python package
├── notebooks/                  # Jupyter notebooks for analysis
├── data/                       # Data storage (raw, processed, interim)
├── artifacts/                  # Generated outputs and results
├── results/                    # Final analysis results
└── pyproject.toml             # Package configuration
```

## 📦 ZCI Package (`src/zci/`)

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

## 📓 Notebooks

Interactive analysis workflows:

- **`00*_build_data_operation.ipynb`** - Data loading, cleaning, and preprocessing pipeline
- **`01_weighted_PCA_scores.ipynb`** - Weighted PCA analysis and pollution scoring
- **`02_ordination_metrices.ipynb`** - Ordination analysis and distance metrics

## 📁 Data Organization

```text
data/
├── raw/                       # Original datasets
├── interim/                   # Intermediate processed data
├── processed/                 # Final processed datasets
├── maps/                      # Spatial data (shapefiles, etc.)
└── data_documents/           # Data documentation
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Project_Code

# Install in development mode
pip install -e .
```

### Usage

```python
# Import ZCI modules
from zci.data_process import dataframe_ops, transform
from zci.sediment_pollution_assessment import weighted_pca, chemical_weights

# Example: Data preprocessing
master_df = dataframe_ops.concat_blocks([env_data, taxa_data, chemical_data])
transformed_taxa = transform.hellinger_transform(taxa_data)

# Example: Weighted PCA analysis
weights = chemical_weights.build_weights_for_columns(chemical_columns)
pca_result = weighted_pca.weighted_pca_analysis(chemical_data, weights)
```

## 🔬 Key Features

- **Multi-index DataFrame Management** - Efficient handling of complex ecological datasets
- **Weighted PCA Analysis** - Priority-based principal component analysis for pollution assessment
- **Data Transformation Pipeline** - Standardized ecological data transformations
- **Ordination Methods** - Multiple distance metrics and ordination techniques
- **Reproducible Workflows** - Jupyter notebook-based analysis pipelines

## 📋 Requirements

- Python ≥ 3.9
- pandas, numpy, scikit-learn, scipy
- See `pyproject.toml` for complete dependencies

## 📊 Research Context

This package supports research on:

- Zoobenthic community structure analysis
- Sediment contamination assessment
- Environmental gradient detection
- Multivariate ecological statistics

---

*For detailed usage examples, see the notebook collection. For function documentation, refer to docstrings in the source code.*
