"""Write tables and figures to disk.

Pure I/O — the *pipeline* decides directory layout; this module just writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd
import matplotlib.pyplot as plt


def save_table(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    formats: Sequence[str] = ("xlsx",),
    float_format: str = "%.6f",
    verbose: bool = True,
    header: bool = True,
) -> Dict[str, Path]:
    """Save a DataFrame to one or more file formats.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    path : str or Path
        **Base** file path *without extension*
        (e.g. ``results/01_pollution_assessment/tables/pc_loadings``).
    formats : sequence of str
        File extensions to produce, e.g. ``("xlsx",)`` or ``("xlsx", "csv")``.
    float_format : str
        C-style format string for floats (used by CSV).
    verbose : bool
        Print confirmation messages.

    Returns
    -------
    dict[str, Path]
        Mapping ``{format: written_path}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        if fmt == "xlsx":
            df.to_excel(out, header=header)
        elif fmt == "csv":
            df.to_csv(out, float_format=float_format, header=header)
        else:
            raise ValueError(f"Unsupported table format: {fmt!r}")
        saved[fmt] = out
        if verbose:
            print(f"  ✓ Saved table: {out}")

    return saved


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    bbox_inches: str = "tight",
    verbose: bool = True,
) -> Dict[str, Path]:
    """Save a matplotlib figure to one or more file formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        **Base** file path *without extension*
        (e.g. ``results/01_pollution_assessment/figures/variance_explained``).
    formats : sequence of str
        File extensions to produce, e.g. ``("png", "pdf")``.
    dpi : int
        Resolution for raster formats.
    bbox_inches : str
        Bounding-box setting passed to ``fig.savefig``.
    verbose : bool
        Print confirmation messages.

    Returns
    -------
    dict[str, Path]
        Mapping ``{format: written_path}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        saved[fmt] = out
        if verbose:
            print(f"  ✓ Saved figure: {out}")

    return saved
