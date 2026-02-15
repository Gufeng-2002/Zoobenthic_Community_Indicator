"""
ZCI v2 — Sediment Pollution Assessment Framework (refactored).

Layers
------
io/        Read raw files, write outputs.  No ecology math here.
core/      Pure functions: transforms, PCA, scoring.  No plotting, no file paths.
viz/       Plot functions that accept already-computed results and return fig/axes.
pipeline/  Orchestrates stages: load → transform → model → summarise → export.
models/    Lightweight dataclasses for structured results.
"""

__version__ = "2.0.0"
