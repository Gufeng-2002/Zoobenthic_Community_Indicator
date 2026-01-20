"""
Output Saver Utility Module

This module provides functions for saving figures, tables, and parameters
from the ZCI pipeline stages with consistent naming conventions.
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


def get_timestamp() -> str:
    """Get current timestamp in the format 'month_day_hour_minute'."""
    now = datetime.now()
    return now.strftime("%m_%d_%H_%M")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: plt.Figure,
    save_dir: Union[str, Path],
    filename: str,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Save a matplotlib figure in multiple formats.
    
    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save
    save_dir : str or Path
        Directory where to save the figure
    filename : str
        Base filename (without extension)
    formats : list of str
        List of formats to save (e.g., ['png', 'pdf', 'svg'])
    dpi : int
        Resolution for raster formats
    bbox_inches : str
        Bounding box specification
    verbose : bool
        Whether to print save messages
    
    Returns
    -------
    dict
        Dictionary mapping format to saved file path
    """
    save_dir = ensure_dir(save_dir)
    saved_paths = {}
    
    for fmt in formats:
        filepath = save_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        saved_paths[fmt] = filepath
        if verbose:
            print(f"  ✓ Saved: {filepath}")
    
    return saved_paths


def save_figures_dict(
    figures: Dict[str, plt.Figure],
    save_dir: Union[str, Path],
    prefix: str = "figure",
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    verbose: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Save multiple figures from a dictionary.
    
    Parameters
    ----------
    figures : dict
        Dictionary of {name: Figure}
    save_dir : str or Path
        Directory where to save figures
    prefix : str
        Prefix for figure numbering (e.g., 'figure' -> 'figure1', 'figure2')
    formats : list of str
        Formats to save
    dpi : int
        Resolution for raster formats
    verbose : bool
        Whether to print save messages
    
    Returns
    -------
    dict
        Dictionary mapping figure name to saved file paths
    """
    save_dir = ensure_dir(save_dir)
    saved_paths = {}
    
    if verbose:
        print(f"\n📁 Saving figures to: {save_dir}")
    
    for i, (name, fig) in enumerate(figures.items(), start=1):
        if fig is None:
            continue
            
        # Create numbered filename with descriptive suffix
        filename = f"{prefix}{i}_{name}"
        paths = save_figure(
            fig=fig,
            save_dir=save_dir,
            filename=filename,
            formats=formats,
            dpi=dpi,
            verbose=verbose
        )
        saved_paths[name] = paths
    
    return saved_paths


def save_table(
    df: pd.DataFrame,
    save_dir: Union[str, Path],
    filename: str,
    formats: List[str] = ['csv', 'xlsx'],
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Save a pandas DataFrame in multiple formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    save_dir : str or Path
        Directory where to save the table
    filename : str
        Base filename (without extension)
    formats : list of str
        List of formats to save (e.g., ['csv', 'xlsx', 'tex'])
    verbose : bool
        Whether to print save messages
    
    Returns
    -------
    dict
        Dictionary mapping format to saved file path
    """
    save_dir = ensure_dir(save_dir)
    saved_paths = {}
    
    for fmt in formats:
        filepath = save_dir / f"{filename}.{fmt}"
        
        if fmt == 'csv':
            df.to_csv(filepath)
        elif fmt == 'xlsx':
            df.to_excel(filepath, engine='openpyxl')
        elif fmt == 'tex':
            df.to_latex(filepath)
        elif fmt == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            continue
            
        saved_paths[fmt] = filepath
        if verbose:
            print(f"  ✓ Saved: {filepath}")
    
    return saved_paths


def save_tables_dict(
    tables: Dict[str, pd.DataFrame],
    save_dir: Union[str, Path],
    prefix: str = "table",
    formats: List[str] = ['csv', 'xlsx'],
    verbose: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Save multiple tables from a dictionary.
    
    Parameters
    ----------
    tables : dict
        Dictionary of {name: DataFrame}
    save_dir : str or Path
        Directory where to save tables
    prefix : str
        Prefix for table numbering
    formats : list of str
        Formats to save
    verbose : bool
        Whether to print save messages
    
    Returns
    -------
    dict
        Dictionary mapping table name to saved file paths
    """
    save_dir = ensure_dir(save_dir)
    saved_paths = {}
    
    if verbose:
        print(f"\n📁 Saving tables to: {save_dir}")
    
    for i, (name, df) in enumerate(tables.items(), start=1):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
            
        filename = f"{prefix}{i}_{name}"
        paths = save_table(
            df=df,
            save_dir=save_dir,
            filename=filename,
            formats=formats,
            verbose=verbose
        )
        saved_paths[name] = paths
    
    return saved_paths


def save_parameters(
    params: Dict[str, Any],
    save_dir: Union[str, Path],
    timestamp: Optional[str] = None,
    formats: List[str] = ['json', 'yaml'],
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Save pipeline parameters with datetime naming.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters to save
    save_dir : str or Path
        Directory where to save parameters
    timestamp : str, optional
        Timestamp string for filename. If None, generates current timestamp.
    formats : list of str
        Formats to save (e.g., ['json', 'yaml'])
    verbose : bool
        Whether to print save messages
    
    Returns
    -------
    dict
        Dictionary mapping format to saved file path
    """
    save_dir = ensure_dir(save_dir)
    
    if timestamp is None:
        timestamp = get_timestamp()
    
    saved_paths = {}
    
    if verbose:
        print(f"\n📁 Saving parameters to: {save_dir}")
    
    # Convert any non-serializable objects to strings
    serializable_params = _make_serializable(params)
    
    for fmt in formats:
        filename = f"pipeline_params_{timestamp}.{fmt}"
        filepath = save_dir / filename
        
        if fmt == 'json':
            with open(filepath, 'w') as f:
                json.dump(serializable_params, f, indent=2, default=str)
        elif fmt == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(serializable_params, f, default_flow_style=False, sort_keys=False)
        else:
            continue
        
        saved_paths[fmt] = filepath
        if verbose:
            print(f"  ✓ Saved: {filepath}")
    
    return saved_paths


def _make_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, (Path,)):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


class OutputSaver:
    """
    Context-aware output saver for pipeline stages.
    
    This class provides a convenient interface for saving outputs from
    each pipeline stage with consistent organization.
    
    Parameters
    ----------
    results_dir : str or Path
        Base results directory (e.g., '../results')
    timestamp : str, optional
        Timestamp for parameter files. If None, generates on first use.
    figure_formats : list of str
        Formats for saving figures
    table_formats : list of str
        Formats for saving tables
    verbose : bool
        Whether to print save messages
    
    Example
    -------
    >>> saver = OutputSaver('../results')
    >>> 
    >>> # Save Stage 1 outputs
    >>> saver.save_stage_figures(1, stage1_results['figures'], 'contamination_assessment')
    >>> saver.save_stage_tables(1, stage1_results['tables'], 'contamination_assessment')
    >>> 
    >>> # Save parameters at the end
    >>> saver.save_parameters(KEY_PARAMS)
    """
    
    STAGE_FOLDERS = {
        1: '01_contamination_assessment',
        2: '02_taxa_assemblage_in_refs',
        3: '03_env_driven_taxa_clusters',
        4: '04_community_composition_measures'
    }
    
    def __init__(
        self,
        results_dir: Union[str, Path],
        timestamp: Optional[str] = None,
        figure_formats: List[str] = ['png', 'pdf'],
        table_formats: List[str] = ['csv', 'xlsx'],
        verbose: bool = True
    ):
        self.results_dir = Path(results_dir)
        self.timestamp = timestamp or get_timestamp()
        self.figure_formats = figure_formats
        self.table_formats = table_formats
        self.verbose = verbose
        
        # Track saved outputs
        self.saved_figures = {}
        self.saved_tables = {}
        self.saved_parameters = {}
        
        if verbose:
            print(f"📂 OutputSaver initialized")
            print(f"   Results directory: {self.results_dir}")
            print(f"   Timestamp: {self.timestamp}")
    
    def get_figure_dir(self, stage: int) -> Path:
        """Get figure directory for a stage."""
        folder = self.STAGE_FOLDERS.get(stage, f'{stage:02d}_stage')
        return self.results_dir / 'figures' / folder
    
    def get_table_dir(self, stage: int) -> Path:
        """Get table directory for a stage."""
        folder = self.STAGE_FOLDERS.get(stage, f'{stage:02d}_stage')
        return self.results_dir / 'tables' / folder
    
    def get_params_dir(self) -> Path:
        """Get parameters directory."""
        return self.results_dir / 'parameters'
    
    def save_stage_figures(
        self,
        stage: int,
        figures: Dict[str, plt.Figure],
        stage_name: Optional[str] = None,
        dpi: int = 300
    ) -> Dict[str, Dict[str, Path]]:
        """
        Save figures for a specific pipeline stage.
        
        Parameters
        ----------
        stage : int
            Stage number (1-4)
        figures : dict
            Dictionary of {figure_name: Figure}
        stage_name : str, optional
            Name for the stage (used in messages)
        dpi : int
            Resolution for saved figures
        
        Returns
        -------
        dict
            Paths to saved figures
        """
        if stage_name is None:
            stage_name = f"Stage {stage}"
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Saving figures for {stage_name}")
            print(f"{'='*60}")
        
        save_dir = self.get_figure_dir(stage)
        
        saved = save_figures_dict(
            figures=figures,
            save_dir=save_dir,
            prefix="figure",
            formats=self.figure_formats,
            dpi=dpi,
            verbose=self.verbose
        )
        
        self.saved_figures[stage] = saved
        
        if self.verbose:
            print(f"\n✓ Saved {len(saved)} figures for {stage_name}")
        
        return saved
    
    def save_stage_tables(
        self,
        stage: int,
        tables: Dict[str, pd.DataFrame],
        stage_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Path]]:
        """
        Save tables for a specific pipeline stage.
        
        Parameters
        ----------
        stage : int
            Stage number (1-4)
        tables : dict
            Dictionary of {table_name: DataFrame}
        stage_name : str, optional
            Name for the stage (used in messages)
        
        Returns
        -------
        dict
            Paths to saved tables
        """
        if stage_name is None:
            stage_name = f"Stage {stage}"
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📋 Saving tables for {stage_name}")
            print(f"{'='*60}")
        
        save_dir = self.get_table_dir(stage)
        
        saved = save_tables_dict(
            tables=tables,
            save_dir=save_dir,
            prefix="table",
            formats=self.table_formats,
            verbose=self.verbose
        )
        
        self.saved_tables[stage] = saved
        
        if self.verbose:
            print(f"\n✓ Saved {len(saved)} tables for {stage_name}")
        
        return saved
    
    def save_all_parameters(
        self,
        params: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Save all pipeline parameters.
        
        Parameters
        ----------
        params : dict
            Main parameter dictionary (e.g., KEY_PARAMS)
        additional_info : dict, optional
            Additional information to include (e.g., data shapes, timestamps)
        
        Returns
        -------
        dict
            Paths to saved parameter files
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⚙️ Saving pipeline parameters")
            print(f"{'='*60}")
        
        # Combine params with additional info
        full_params = {
            'timestamp': self.timestamp,
            'parameters': params
        }
        
        if additional_info:
            full_params['run_info'] = additional_info
        
        saved = save_parameters(
            params=full_params,
            save_dir=self.get_params_dir(),
            timestamp=self.timestamp,
            formats=['json', 'yaml'],
            verbose=self.verbose
        )
        
        self.saved_parameters = saved
        
        return saved
    
    def save_single_figure(
        self,
        fig: plt.Figure,
        stage: int,
        name: str,
        dpi: int = 300
    ) -> Dict[str, Path]:
        """
        Save a single figure to the appropriate stage directory.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        stage : int
            Stage number (1-4)
        name : str
            Figure name (will be prefixed with 'figure_')
        dpi : int
            Resolution for saved figure
        
        Returns
        -------
        dict
            Paths to saved figure files
        """
        save_dir = self.get_figure_dir(stage)
        return save_figure(
            fig=fig,
            save_dir=save_dir,
            filename=f"figure_{name}",
            formats=self.figure_formats,
            dpi=dpi,
            verbose=self.verbose
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all saved outputs."""
        return {
            'timestamp': self.timestamp,
            'results_dir': str(self.results_dir),
            'figures_saved': {
                stage: list(figs.keys()) 
                for stage, figs in self.saved_figures.items()
            },
            'tables_saved': {
                stage: list(tables.keys()) 
                for stage, tables in self.saved_tables.items()
            },
            'parameters_saved': list(self.saved_parameters.keys()) if self.saved_parameters else []
        }
    
    def print_summary(self):
        """Print a summary of all saved outputs."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print("📦 OUTPUT SAVER SUMMARY")
        print(f"{'='*60}")
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Results directory: {summary['results_dir']}")
        
        print(f"\n📊 Figures saved:")
        for stage, figs in summary['figures_saved'].items():
            print(f"  Stage {stage}: {len(figs)} figures")
            for fig_name in figs:
                print(f"    - {fig_name}")
        
        print(f"\n📋 Tables saved:")
        for stage, tables in summary['tables_saved'].items():
            print(f"  Stage {stage}: {len(tables)} tables")
            for table_name in tables:
                print(f"    - {table_name}")
        
        if summary['parameters_saved']:
            print(f"\n⚙️ Parameters saved: {', '.join(summary['parameters_saved'])}")
        
        print(f"\n{'='*60}")
