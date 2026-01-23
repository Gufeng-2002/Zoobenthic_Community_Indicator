"""
Excel to LaTeX Table Converter

This module converts all Excel tables in the tables/ folder to LaTeX format
and saves them in the table_latex/ folder with the same directory structure.
"""

import os
import pandas as pd
from pathlib import Path


def escape_latex(text):
    """Escape special LaTeX characters in text."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Escape special characters
    replacements = {
        '%': r'\%',
        '&': r'\&',
        '_': r'\_',
        '#': r'\#',
        '$': r'\$',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def format_number(val, decimals=4):
    """Format numbers with appropriate precision."""
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float)):
        if isinstance(val, float):
            # Check if it's a very small p-value
            if abs(val) < 0.0001 and val != 0:
                return f"{val:.2e}"
            # Regular float formatting
            return f"{val:.{decimals}f}".rstrip('0').rstrip('.')
        return str(val)
    return str(val)


def df_to_latex(df, caption=None, label=None, float_format=4):
    """
    Convert a pandas DataFrame to a well-formatted LaTeX table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    float_format : int
        Number of decimal places for floats
    
    Returns:
    --------
    str : LaTeX table code
    """
    # Clean column names - handle 'Unnamed' columns appropriately
    columns = list(df.columns)
    if columns[0].startswith('Unnamed'):
        # Check if the first column contains meaningful data (not just indices)
        first_col_values = df.iloc[:, 0].dropna()
        if len(first_col_values) > 0 and not all(isinstance(v, (int, float)) for v in first_col_values):
            # Contains text data - rename the column based on content or leave empty
            df = df.copy()
            df.columns = [''] + list(df.columns[1:])
            columns = list(df.columns)
        else:
            # Just an index column - remove it
            df = df.iloc[:, 1:]
            columns = list(df.columns)
    
    # Escape column names
    escaped_columns = [escape_latex(col) for col in columns]
    
    # Determine column alignment
    n_cols = len(columns)
    # First column left-aligned, rest centered
    col_spec = 'l' + 'c' * (n_cols - 1)
    
    # Build the LaTeX table
    lines = []
    
    # Document structure for standalone table file
    lines.append(r"% LaTeX table generated automatically")
    lines.append(r"% Use \input{filename.tex} to include in your document")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    
    if caption:
        lines.append(f"    \\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(f"    \\label{{{label}}}")
    
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")
    
    # Header row
    header = " & ".join(escaped_columns) + r" \\"
    lines.append(f"        {header}")
    lines.append(r"        \midrule")
    
    # Data rows
    for idx, row in df.iterrows():
        cells = []
        for i, val in enumerate(row):
            if isinstance(val, (int, float)) and not pd.isna(val):
                cells.append(format_number(val, float_format))
            else:
                cells.append(escape_latex(val))
        row_str = " & ".join(cells) + r" \\"
        lines.append(f"        {row_str}")
    
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    
    return "\n".join(lines)


def excel_to_latex_file(excel_path, output_path, caption=None):
    """
    Convert an Excel file to a LaTeX table file.
    
    Parameters:
    -----------
    excel_path : str or Path
        Path to the Excel file
    output_path : str or Path
        Path for the output .tex file
    caption : str, optional
        Table caption (derived from filename if not provided)
    """
    excel_path = Path(excel_path)
    output_path = Path(output_path)
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Generate caption from filename if not provided
    if caption is None:
        # Convert filename to readable caption
        name = excel_path.stem
        # Remove 'table' prefix and number if present
        caption = name.replace('_', ' ').title()
    
    # Generate label from filename
    label = f"tab:{excel_path.stem}"
    
    # Convert to LaTeX
    latex_content = df_to_latex(df, caption=caption, label=label)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the LaTeX file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    return output_path


def convert_all_tables(tables_dir, output_dir, verbose=True):
    """
    Convert all Excel tables in the tables directory to LaTeX format.
    
    Parameters:
    -----------
    tables_dir : str or Path
        Path to the tables directory containing Excel files
    output_dir : str or Path
        Path to the output directory for LaTeX files
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    list : List of paths to created LaTeX files
    """
    tables_dir = Path(tables_dir)
    output_dir = Path(output_dir)
    
    created_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(tables_dir):
        # Skip hidden directories and files
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'Icon']
        
        root_path = Path(root)
        
        # Find all Excel files
        excel_files = [f for f in files if f.endswith('.xlsx') and not f.startswith('~')]
        
        for excel_file in excel_files:
            excel_path = root_path / excel_file
            
            # Calculate relative path from tables_dir
            rel_path = root_path.relative_to(tables_dir)
            
            # Create output path with same structure
            output_subdir = output_dir / rel_path
            output_file = output_subdir / (Path(excel_file).stem + '.tex')
            
            try:
                result = excel_to_latex_file(excel_path, output_file)
                created_files.append(result)
                if verbose:
                    print(f"✓ Converted: {excel_path.name} -> {output_file}")
            except Exception as e:
                if verbose:
                    print(f"✗ Error converting {excel_path.name}: {e}")
    
    return created_files


def main():
    """Main function to run the conversion."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define input and output directories
    tables_dir = script_dir / "tables"
    output_dir = script_dir / "table_latex"
    
    print("=" * 60)
    print("Excel to LaTeX Table Converter")
    print("=" * 60)
    print(f"Input directory:  {tables_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Convert all tables
    created_files = convert_all_tables(tables_dir, output_dir)
    
    print("-" * 60)
    print(f"Conversion complete! Created {len(created_files)} LaTeX files.")
    print("=" * 60)
    
    return created_files


if __name__ == "__main__":
    main()
