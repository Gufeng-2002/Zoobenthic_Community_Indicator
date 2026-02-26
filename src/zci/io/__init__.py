"""I/O layer — reading raw files, parsing, validation, writing outputs."""

from .readers import read_study_data, extract_block, wrap_columns, concat_blocks
from .writers import save_table, save_figure
