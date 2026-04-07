#!/usr/bin/env python3
"""
Convert all Excel tables under 02_taxa_assemblage/ into LaTeX format.

Walks every ``tables/`` subfolder (WardsClustering, LDA_Method, MRT_Method,
GridSweep, etc.), reads each ``.xlsx`` file, and writes a matching ``.tex``
file **in the same directory** so that both formats live side-by-side.

The converter preserves the layout of each spreadsheet as-is:
  - header row → column headers
  - body rows  → tabular rows
  - numeric values are formatted to a sensible precision
  - LaTeX special characters are escaped
  - wide tables (>10 cols) are resized; tall tables (>40 rows) use longtable

Usage (from project root):
    python results/02_taxa_assemblage/convert_tables_to_latex.py
"""

import os
import re
import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace('\\', r'\textbackslash{}')
    for ch, repl in [('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                      ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                      ('}', r'\}'), ('~', r'\textasciitilde{}'),
                      ('^', r'\textasciicircum{}')]:
        text = text.replace(ch, repl)
    text = text.replace('±', r'$\pm$')
    text = text.replace('°', r'$^{\circ}$')
    text = re.sub(r'\(oC\)', r'($^{\\circ}$C)', text)
    # em-dash / en-dash keep as-is (pdflatex handles them)
    return text


def mathify_cm_text(text: str) -> str:
    """Convert confusion-matrix specific symbols to LaTeX math expressions.

    Applied *after* normal LaTeX escaping for column headers and special
    cells in comparison confusion-matrix tables.
    """
    if text is None:
        return ""
    text = str(text)
    # Cluster labels: "Cluster C1" → "Cluster $C_1$"
    text = re.sub(r'Cluster C(\d+)', r'Cluster $C_{\1}$', text)
    # Standalone "C1" etc. (e.g. in header cells already escaped as C1)
    text = re.sub(r'\bC(\d+)\b', r'$C_{\1}$', text)
    # p_max=0.562 → $p_{\max}=0.562$
    text = re.sub(r'p[_\\]*max\s*=\s*([\d.]+)', r'$p_{\\max}=\1$', text)
    # Dp=0.284 → $\Delta p=0.284$
    text = re.sub(r'Dp\s*=\s*([\d.]+)', r'$\\Delta p=\1$', text)
    # "n=15" → "$n=15$"  (inside parentheses like Held-out (n=15))
    text = re.sub(r'\bn=(\d+)\b', r'$n=\1$', text)
    # "% Correct" header → "\% Correct"  (already handled by escape_latex)
    # "x" between numbers for cross-validation: "5-fold x 10" → "$5$-fold $\times$ $10$"
    text = re.sub(r'(\d+)-fold\s*x\s*(\d+)', r'$\1$-fold $\\times$ $\2$', text)
    return text


def format_number(val, decimals=4) -> str:
    """Format a numeric value for LaTeX with sensible precision."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return f"{val:,}".replace(",", r"{,}")
    if isinstance(val, float):
        abs_v = abs(val)
        if abs_v == 0:
            return "0"
        if abs_v < 0.001:
            return f"{val:.6f}"
        if abs_v < 1:
            return f"{val:.{decimals}f}"
        if abs_v < 100:
            return f"{val:.{min(decimals, 2)}f}"
        if abs_v < 10000:
            return f"{val:.{min(decimals, 1)}f}"
        return f"{val:.0f}"
    return str(val)


def format_cell(val) -> str:
    """Format a single cell value for LaTeX."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return ""
    if isinstance(val, (int, float)):
        return format_number(val)
    return escape_latex(str(val))


def is_index_column(ws) -> bool:
    """True if column A is a throwaway sequential integer index."""
    header_val = ws.cell(row=1, column=1).value
    if header_val is not None and str(header_val).strip() != "":
        return False
    vals = []
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        v = row[0]
        if v is not None:
            try:
                vals.append(int(v))
            except (ValueError, TypeError):
                return False
    if len(vals) < 2:
        return False
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return all(d == 1 for d in diffs)


# ──────────────────────────────────────────────────────────────────────────────
# Caption / label
# ──────────────────────────────────────────────────────────────────────────────

def _prettify(name: str) -> str:
    """Turn a filename stem into a readable caption."""
    return name.replace('_', ' ').title()


def get_caption(rel_path: str, filename: str) -> str:
    """Build a caption from the relative path and filename."""
    stem = os.path.splitext(filename)[0]
    parts = rel_path.replace(os.sep, '/').split('/')
    # e.g. LDA_Method/ModelA_CoreOnly → "LDA Method — Model A Core Only"
    ctx = ' — '.join(_prettify(p) for p in parts if p not in ('tables',))
    return f"{_prettify(stem)} ({ctx})"


def get_label(rel_path: str, filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    tag = rel_path.replace(os.sep, '_').replace('/', '_')
    return f"tab:{tag}_{stem}"


# ──────────────────────────────────────────────────────────────────────────────
# Row classification
# ──────────────────────────────────────────────────────────────────────────────

def _classify_rows(all_rows, start_col):
    """Separate meaningful data rows from note/empty rows.

    Keeps section-header rows (like unified CM part labels) inline
    to preserve the spreadsheet layout faithfully.
    """
    data_rows = []
    notes = []

    for i, row in enumerate(all_rows):
        if i == 0:
            data_rows.append(row)
            continue

        relevant = row[start_col:]
        non_none = [(j, v) for j, v in enumerate(relevant)
                    if v is not None and str(v).strip() != ""]

        if len(non_none) == 0:
            continue

        data_rows.append(row)

    return data_rows, notes


# ──────────────────────────────────────────────────────────────────────────────
# Confusion-matrix comparison tables (special handling)
# ──────────────────────────────────────────────────────────────────────────────

_CM_COMPARISON_FILES = {
    "confusion_matrix_all_refsites",
    "cv_confusion_matrices_combined",
}


def _is_cm_comparison(filename: str) -> bool:
    """Return True if this file is a multi-model confusion-matrix comparison."""
    stem = os.path.splitext(filename)[0]
    return stem in _CM_COMPARISON_FILES


def _format_cm_cell(val) -> str:
    """Format a cell in confusion-matrix comparison tables with math symbols."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return ""
    if isinstance(val, (int, float)):
        return format_number(val)
    text = str(val).strip()
    if text == "/":
        return "/"
    # Check for math-pattern cells and return LaTeX math directly
    m = re.match(r'^p_max=([\d.]+)$', text)
    if m:
        return f"$p_{{\\max}}={m.group(1)}$"
    m = re.match(r'^Dp=([\d.]+)$', text)
    if m:
        return f"$\\Delta p={m.group(1)}$"
    return escape_latex(text)


def convert_cm_comparison(xlsx_path: str, rel_path: str, filename: str) -> str | None:
    """Convert a multi-model confusion-matrix comparison table to LaTeX.

    Adds \\midrule between model sections, uses \\multicolumn for long
    header rows, and converts math symbols.
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    all_rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()

    if not all_rows:
        return None

    skip_first = is_index_column(ws)
    start_col = 1 if skip_first else 0
    num_cols = len(all_rows[0]) - start_col

    # Filter empty rows
    data_rows, _ = _classify_rows(all_rows, start_col)
    if len(data_rows) < 2:
        return None

    header = data_rows[0][start_col:]
    body = data_rows[1:]

    caption = get_caption(rel_path, filename)
    label = get_label(rel_path, filename)
    # First col left-aligned, rest right-aligned
    align = "l" + "r" * (num_cols - 1) if num_cols > 1 else "l"

    # Math-ify column headers: "% Correct" stays, "Cluster C1" → "Cluster $C_1$"
    header_cells = []
    for h in header:
        if h is None:
            header_cells.append("")
        else:
            t = escape_latex(str(h))
            t = mathify_cm_text(t)
            header_cells.append(t)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{align}}}")
    lines.append(r"    \toprule")
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    first_model = True
    for row in body:
        cells = row[start_col:]

        # The first cell of each row is the row label (text).
        # For these comparison tables is_index_column returns False,
        # so start_col=0 and cells[0] is the label text.
        first_cell = str(cells[0]).strip() if cells[0] is not None else ""

        is_model_header = first_cell.startswith("Model ")
        is_heldout = first_cell.startswith("Held-out")

        if is_model_header:
            if not first_model:
                lines.append(r"    \midrule")
            first_model = False

            header_text = escape_latex(first_cell)
            header_text = mathify_cm_text(header_text)
            lines.append(
                f"    \\multicolumn{{{num_cols}}}{{l}}"
                f"{{\\textbf{{{header_text}}}}} \\\\"
            )
            continue

        if is_heldout:
            heldout_label = escape_latex(first_cell)
            heldout_label = mathify_cm_text(heldout_label)
            formatted_cells = [heldout_label]
            for v in cells[1:]:
                formatted_cells.append(_format_cm_cell(v))
            lines.append("    " + " & ".join(formatted_cells) + r" \\")
            continue

        # Regular row (Cluster C1, Total, etc.)
        formatted = []
        for i, v in enumerate(cells):
            if i == 0 and isinstance(v, str):
                t = escape_latex(v)
                t = mathify_cm_text(t)
                formatted.append(t)
            else:
                formatted.append(_format_cm_cell(v))
        lines.append("    " + " & ".join(formatted) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_one(xlsx_path: str, rel_path: str, filename: str) -> str | None:
    """Return LaTeX source string for one Excel table, or None if empty."""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    all_rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()

    if not all_rows:
        return None

    skip_first = is_index_column(ws)
    start_col = 1 if skip_first else 0
    num_cols = len(all_rows[0]) - start_col

    data_rows, notes = _classify_rows(all_rows, start_col)
    if len(data_rows) < 2:
        return None

    header = data_rows[0][start_col:]
    header_cells = [escape_latex(str(h)) if h else "" for h in header]
    body = data_rows[1:]

    caption = get_caption(rel_path, filename)
    label = get_label(rel_path, filename)
    align = "l" + "r" * (num_cols - 1) if num_cols > 1 else "l"

    use_long = len(body) > 40
    use_resize = num_cols > 10
    use_small = 7 <= num_cols <= 10

    lines: list[str] = []

    if use_long:
        lines.append(r"% Requires: \usepackage{longtable, booktabs}")
        if use_small:
            lines.append(r"{\small")
        if use_resize:
            lines.append(r"{\footnotesize")

        lines.append(f"\\begin{{longtable}}{{{align}}}")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}} \\\\")
        lines.append(r"  \toprule")
        lines.append("  " + " & ".join(header_cells) + r" \\")
        lines.append(r"  \midrule")
        lines.append(r"  \endfirsthead")
        lines.append(f"  \\multicolumn{{{num_cols}}}{{l}}"
                      f"{{\\tablename\\ \\thetable\\ -- continued}} \\\\")
        lines.append(r"  \toprule")
        lines.append("  " + " & ".join(header_cells) + r" \\")
        lines.append(r"  \midrule")
        lines.append(r"  \endhead")
        lines.append(f"  \\midrule \\multicolumn{{{num_cols}}}{{r}}"
                      r"{{Continued on next page}} \\")
        lines.append(r"  \endfoot")
        lines.append(r"  \bottomrule")
        if notes:
            for note in notes:
                lines.append(
                    f"  \\multicolumn{{{num_cols}}}{{l}}"
                    f"{{\\footnotesize {escape_latex(note)}}} \\\\"
                )
        lines.append(r"  \endlastfoot")
    else:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}}")
        if use_resize:
            lines.append(r"  \resizebox{\textwidth}{!}{%")
        elif use_small:
            lines.append(r"  \small")
        lines.append(f"  \\begin{{tabular}}{{{align}}}")
        lines.append(r"    \toprule")
        lines.append("    " + " & ".join(header_cells) + r" \\")
        lines.append(r"    \midrule")

    # Body rows
    indent = "  " if use_long else "    "
    for row in body:
        cells = row[start_col:]
        formatted = [format_cell(v) for v in cells]
        lines.append(indent + " & ".join(formatted) + r" \\")

    # Closing
    if use_long:
        lines.append(r"\end{longtable}")
        if use_small or use_resize:
            lines.append("}")
    else:
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        if use_resize:
            lines.append(r"  }")
        if notes:
            lines.append(r"  \\[4pt]")
            for note in notes:
                lines.append(
                    f"  \\parbox{{\\textwidth}}"
                    f"{{\\footnotesize {escape_latex(note)}}}"
                )
        lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Directory walker
# ──────────────────────────────────────────────────────────────────────────────

def walk_and_convert(root_dir: str) -> int:
    """Convert all .xlsx in tables/ subdirs → .tex in a sibling table_latex/ directory."""
    converted = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) != "tables":
            continue

        xlsx_files = sorted(f for f in filenames if f.endswith(".xlsx"))
        if not xlsx_files:
            continue

        rel_path = os.path.relpath(dirpath, root_dir)
        # Sibling table_latex/ folder next to tables/
        latex_dir = os.path.join(os.path.dirname(dirpath), "table_latex")
        os.makedirs(latex_dir, exist_ok=True)

        for fname in xlsx_files:
            xlsx_path = os.path.join(dirpath, fname)
            tex_name = os.path.splitext(fname)[0] + ".tex"
            tex_path = os.path.join(latex_dir, tex_name)

            if _is_cm_comparison(fname):
                tex_src = convert_cm_comparison(xlsx_path, rel_path, fname)
            else:
                tex_src = convert_one(xlsx_path, rel_path, fname)
            if tex_src is None:
                print(f"  [skip] {rel_path}/{fname}  (empty)")
                continue

            with open(tex_path, "w", encoding="utf-8") as fh:
                fh.write(tex_src)
            converted += 1
            latex_rel = os.path.relpath(latex_dir, root_dir)
            print(f"  [ok]   {latex_rel}/{tex_name}")

    return converted


if __name__ == "__main__":
    print(f"Root: {SCRIPT_DIR}\n")
    n = walk_and_convert(SCRIPT_DIR)
    print(f"\nDone — {n} table(s) converted to LaTeX.")
