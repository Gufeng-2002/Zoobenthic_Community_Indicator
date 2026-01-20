"""
ZCI (Zhigan Chemical Index) - Sediment Pollution Assessment Framework

This package provides tools for ecological data analysis, particularly focused on
sediment contamination assessment using PCA-based methods and statistical evaluation.

Subpackages:
- data_process: Data transformation and DataFrame operations
- contamination_assessment: PCA-based pollution scoring
- taxa_assemblage_in_refs: Reference site selection and clustering
- env_driven_taxa_clusters: RDA and LDA analysis
- community_composition_measures: ZCI calculation and visualization
- output_saver: Utilities for saving figures, tables, and parameters

Usage:
    Import from specific submodules:
    
    from zci.data_process import hellinger_transform
    from zci.contamination_assessment import contamination_assessment_pipeline
    from zci.output_saver import OutputSaver
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "ZCI (Zhigan Chemical Index) - Sediment Pollution Assessment Framework"

# Export output_saver for convenience
from .output_saver import OutputSaver, save_figure, save_figures_dict, save_table, save_tables_dict, save_parameters