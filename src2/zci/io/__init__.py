"""I/O layer — reading raw files, parsing, validation, writing outputs."""

from .readers import read_study_data, extract_block
from .writers import save_table, save_figure
