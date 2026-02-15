"""
Environment-Driven Taxa Clusters Module

This module provides comprehensive tools for analyzing how environmental conditions
drive benthic macroinvertebrate community composition and classifying sites into
habitat-based clusters.

The module integrates:
1. RDA (Redundancy Analysis) - Ordination to link environment and community composition
2. LDA (Linear Discriminant Analysis) - Classification into habitat-based clusters
3. Cross-validation and prediction for non-reference sites
4. Comprehensive visualization and statistical testing

Main Pipeline:
-------------
perform_env_taxa_analysis : Complete integrated analysis pipeline

RDA Functions:
-------------
perform_rda_analysis : RDA ordination with permutation tests
create_rda_axes_summary_table : Summary table for RDA axes
create_rda_terms_summary_table : Summary table for environmental terms
plot_rda_triplot : Publication-ready RDA triplot

LDA Functions:
-------------
perform_lda_analysis : Train and evaluate LDA model
perform_monte_carlo_cv : Monte Carlo cross-validation
plot_lda_cv_results : Visualize CV results
plot_lda_triplot : LDA triplot with site scores and habitat vectors
plot_cluster_comparison : Compare habitat/taxa patterns across clusters
compare_lda_rda_axes : Compare LDA and RDA axes
predict_nonreference_sites : Classify non-reference sites
update_data_with_predictions : Update datasets with predictions

Core Module:
-----------
RDA : Redundancy Analysis class (low-level API)

Examples
--------
>>> from zci.env_driven_taxa_clusters import perform_env_taxa_analysis
>>> 
>>> # Run complete pipeline
>>> results = perform_env_taxa_analysis(
...     raw_data=raw_data_m_by_p3,
...     multiindex_data=multiindex_data_m_by_p3,
...     env_variables=['Depth', 'Velocity', 'DO', 'Temp', 'MPS', 'LOI'],
...     taxa_transformation='hellinger',
...     lda_n_iterations=1000,
...     verbose=True
... )
>>> 
>>> # Access all results
>>> print(f"RDA R²: {results['rda_results']['rda_model'].fit_.r2:.3f}")
>>> print(f"LDA Accuracy: {results['lda_results']['accuracy']:.2%}")
>>> results['rda_triplot']
>>> results['lda_cv_figure']
"""

# Import core RDA module
from .rda import RDA

# Import comprehensive pipeline
from .pipeline import perform_env_taxa_analysis

# Import RDA pipeline components
from .rda_pipeline import (
    perform_rda_analysis,
    create_rda_axes_summary_table,
    create_rda_terms_summary_table,
    plot_rda_triplot
)

# Import LDA pipeline components
from .lda_pipeline import (
    perform_lda_analysis,
    perform_monte_carlo_cv,
    plot_lda_cv_results,
    plot_lda_triplot,
    plot_cluster_comparison,
    compare_lda_rda_axes,
    predict_nonreference_sites,
    update_data_with_predictions,
    compute_lda_variable_importance,
    create_lda_excel_table,
    # LDA table formatting functions
    create_lda_confusion_matrix_table,
    create_lda_classification_report_table,
    create_mccv_confusion_matrix_table,
    create_mccv_classification_report_table,
    save_lda_tables_to_excel
)

__all__ = [
    # Main pipeline
    'perform_env_taxa_analysis',
    
    # Core RDA module
    'RDA',
    
    # RDA pipeline functions
    'perform_rda_analysis',
    'create_rda_axes_summary_table',
    'create_rda_terms_summary_table',
    'plot_rda_triplot',
    
    # LDA pipeline functions
    'perform_lda_analysis',
    'perform_monte_carlo_cv',
    'plot_lda_cv_results',
    'plot_lda_triplot',
    'plot_cluster_comparison',
    'compare_lda_rda_axes',
    'predict_nonreference_sites',
    'update_data_with_predictions',
    'compute_lda_variable_importance',
    'create_lda_excel_table',
    
    # LDA table formatting functions
    'create_lda_confusion_matrix_table',
    'create_lda_classification_report_table',
    'create_mccv_confusion_matrix_table',
    'create_mccv_classification_report_table',
    'save_lda_tables_to_excel'
]
