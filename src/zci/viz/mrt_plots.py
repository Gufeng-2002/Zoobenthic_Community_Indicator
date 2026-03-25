"""MRT figure generation using the live mvpart objects in R."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from rpy2 import robjects as ro

from ..core.mrt import MRTRuntime
from ..models.mrt import MRTResult


def save_mrt_cp_tree_figure(
    result: MRTResult,
    runtime: MRTRuntime,
    output_path: str | Path,
) -> Path:
    """Save the 2-panel CP/tree figure with the same plotting logic as R."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cp_selected = result.cp_table[result.cp_table["nsplit"] == result.pruned_nsplits]
    if cp_selected.empty:
        raise RuntimeError("Selected tree size was not found in the CP table")
    info = cp_selected.iloc[0]
    selected_size = result.pruned_nsplits + 1

    path_escaped = str(output_path).replace("\\", "/")
    se1_level = result.min_cv_error + result.min_cv_se
    n_ref = len(result.ref_stations)

    ro.globalenv["zci_py_mrt_fig_path"] = ro.StrVector([path_escaped])
    ro.globalenv["zci_py_mrt_selected_size"] = ro.IntVector([selected_size])
    ro.globalenv["zci_py_mrt_selected_xerror"] = ro.FloatVector([float(info["xerror"])])
    ro.globalenv["zci_py_mrt_se1"] = ro.FloatVector([float(se1_level)])
    ro.globalenv["zci_py_mrt_n_ref"] = ro.IntVector([n_ref])
    ro.globalenv["zci_py_mrt_rel_error"] = ro.FloatVector(result.cp_table["rel error"].astype(float).tolist())
    ro.globalenv["zci_py_mrt_xerror"] = ro.FloatVector(result.cp_table["xerror"].astype(float).tolist())
    ro.globalenv["zci_py_mrt_xstd"] = ro.FloatVector(result.cp_table["xstd"].astype(float).tolist())
    ro.globalenv["zci_py_mrt_sizes"] = ro.IntVector((result.cp_table["nsplit"].astype(int) + 1).tolist())
    ro.globalenv["zci_py_mrt_cp_labels"] = ro.StrVector([
        "Inf" if i == 0 else f"{cp:.3g}" for i, cp in enumerate(result.cp_table["CP"].astype(float).tolist())
    ])

    plot_code = dedent(
        f"""
        local({{
          full_obj <- {runtime.full_name}
          pruned_obj <- {runtime.pruned_name}
          old_par <- par(no.readonly = TRUE)
          old_pal <- palette()
          on.exit({{ palette(old_pal); par(old_par); try(dev.off(), silent = TRUE) }}, add = TRUE)

          png(zci_py_mrt_fig_path[1], width = 2400, height = 1000, res = 130, bg = "white")
          layout(matrix(c(1, 2), nrow = 1), widths = c(1, 1))

          # ── LEFT PANEL: CP plot ──
          par(mar = c(5.5, 5.5, 5.0, 1.0), bg = "white", xpd = NA)

          x  <- seq_along(zci_py_mrt_xerror)
          lo <- zci_py_mrt_xerror - zci_py_mrt_xstd
          hi <- zci_py_mrt_xerror + zci_py_mrt_xstd
          y_min <- min(lo, zci_py_mrt_rel_error, na.rm = TRUE)
          y_max <- max(hi, zci_py_mrt_rel_error, zci_py_mrt_se1, na.rm = TRUE)

          plot(x, zci_py_mrt_xerror,
               type = "n", axes = FALSE,
               xlab = "", ylab = "Relative Error",
               ylim = c(y_min - 0.03, y_max + 0.08),
               cex.lab = 1.35, cex.axis = 1.1)

          # Bottom axis – size of tree
          axis(1, at = x, labels = zci_py_mrt_sizes, cex.axis = 1.05)
          mtext("size of tree", side = 1, line = 3.4, cex = 1.20)
          # Left axis
          axis(2, cex.axis = 1.05, las = 1)
          # Top axis – complexity parameter
          axis(3, at = x, labels = zci_py_mrt_cp_labels, cex.axis = 0.92)
          mtext("complexity parameter", side = 3, line = 2.5, cex = 1.20)
          box(bty = "l", col = "#6D6A75")

          # Error bars: caps + vertical stems
          cap_hw <- 0.15
          segments(x, lo, x, hi, col = "#5B9BD5", lwd = 1.8)
          segments(x - cap_hw, lo, x + cap_hw, lo, col = "#5B9BD5", lwd = 1.5)
          segments(x - cap_hw, hi, x + cap_hw, hi, col = "#5B9BD5", lwd = 1.5)

          # CVRE median points (filled blue circles)
          points(x, zci_py_mrt_xerror, pch = 16, cex = 1.10, col = "#5B9BD5")

          # Full-data RE curve (red dashed with points)
          lines(x, zci_py_mrt_rel_error, lwd = 1.8, lty = 2, col = "#FF3B30")
          points(x, zci_py_mrt_rel_error, pch = 21, cex = 0.95, col = "#FF3B30", bg = "white")

          # Min-CVRE highlight (green open circle)
          i_min <- which.min(zci_py_mrt_xerror)
          points(i_min, min(zci_py_mrt_xerror),
                 pch = 21, bg = "#FFFFFF", col = "#2E7D32", cex = 1.55, lwd = 1.6)

          # Selected tree (red filled)
          points(zci_py_mrt_selected_size[1], zci_py_mrt_selected_xerror[1],
                 pch = 16, col = "#D62828", cex = 1.40)

          # +1SE horizontal line (clipped to left panel only)
          segments(min(x) - 0.5, zci_py_mrt_se1[1], max(x) + 0.5, zci_py_mrt_se1[1],
                   col = "#D62828", lwd = 1.6, lty = 2)
          text(1.3, zci_py_mrt_se1[1] + 0.015, "+1SE",
               col = "#D62828", adj = c(0, 0), cex = 1.10, font = 3)

          # "n = ... sites" annotation
          text(max(x) - 0.3, y_min + 0.01,
               sprintf("n = %d sites", zci_py_mrt_n_ref[1]),
               adj = c(1, 0), cex = 1.05, col = "#4B5563")

          # Legend
          legend("topleft",
                 legend = c(expression("median " %+-% " mean CI"),
                            "full-data", "min CVRE", "selected tree"),
                 col  = c("#5B9BD5", "#FF3B30", "#2E7D32", "#D62828"),
                 lty  = c(NA, 2, NA, NA),
                 pch  = c(16, NA, 21, 16),
                 pt.bg = c(NA, NA, "#FFFFFF", NA),
                 pt.cex = c(1.1, NA, 1.2, 1.1),
                 bty = "n", cex = 1.02, seg.len = 2.3)

          # ── RIGHT PANEL: Pruned tree (fallen-leaves layout) ──
          if ({result.pruned_nsplits} > 0) {{
            par(mar = c(6.0, 2.0, 5.5, 2.0), xpd = NA)

            # Draw tree structure (branches only)
            mvpart:::plot.rpart(pruned_obj, uniform = TRUE, branch = 1, margin = 0.25)

            # Title – bold italic
            title(main = sprintf("Tree of size %d", {selected_size}),
                  line = 3.0, cex.main = 1.40, font.main = 4)
            mtext(sprintf("RE : %.2f   CVRE : %.3f   SE : %.3f",
                          {float(info['rel error'])},
                          {float(info['xerror'])},
                          {float(info['xstd'])}),
                  side = 3, line = 1.6, cex = 1.05)

            # — Coordinates and frame info —
            rp_parms <- list(uniform = TRUE, branch = 1, nspace = -1L, minbranch = 0.3)
            xy  <- mvpart:::rpartco(pruned_obj, rp_parms)
            fr  <- pruned_obj$frame
            is_leaf   <- (fr$var == "<leaf>")
            node_nums <- as.integer(rownames(fr))
            bottom_y  <- min(xy$y[is_leaf])

            # Extend shallow leaf branches down to the bottom level
            for (i in which(is_leaf)) {{
              if (xy$y[i] > bottom_y + 0.01) {{
                segments(xy$x[i], xy$y[i], xy$x[i], bottom_y, lwd = 1.4)
              }}
            }}

            # — Split condition labels —
            labs <- labels(pruned_obj, pretty = 0)
            for (i in which(!is_leaf)) {{
              nd <- node_nums[i]
              li <- match(2 * nd, node_nums)
              ri <- match(2 * nd + 1, node_nums)
              if (!is.na(li) && !is.na(ri)) {{
                text(mean(c(xy$x[i], xy$x[li])), xy$y[i],
                     labs[li], cex = 1.02, pos = 3, offset = 0.3)
                text(mean(c(xy$x[i], xy$x[ri])), xy$y[i],
                     labs[ri], cex = 1.02, pos = 3, offset = 0.3)
              }}
            }}

            # — Barplots at the bottom for ALL leaves —
            n_taxa     <- ncol(zci_mrt_taxa_mat)
            taxa_means <- fr$yval2[, , drop = FALSE]
            max_val    <- max(abs(taxa_means[is_leaf, ]))
            usr        <- par("usr")
            bar_h      <- 0.16 * diff(usr[3:4])
            bar_w      <- 0.14 * diff(usr[1:2])
            bar_top_y  <- bottom_y - 0.03 * diff(usr[3:4])

            bar_cols <- colorRampPalette(c("#0D1B6F", "#1A3A8F", "#3F51B5",
                                           "#7986CB", "#C5CAE9"))(n_taxa)

            for (i in which(is_leaf)) {{
              cx <- xy$x[i]
              m  <- as.numeric(taxa_means[i, ])
              nb <- length(m)
              bw <- bar_w / nb
              bar_bot_y <- bar_top_y - bar_h
              for (j in seq_len(nb)) {{
                h  <- m[j] / max_val * bar_h
                x0 <- cx - bar_w / 2 + (j - 1) * bw
                x1 <- x0 + bw * 0.85
                rect(x0, bar_bot_y, x1, bar_bot_y + h,
                     col = bar_cols[j], border = NA)
              }}

              text(cx, bar_bot_y - 0.04 * diff(usr[3:4]),
                   sprintf("%.3g : n=%d", fr$yval[i], fr$n[i]),
                   cex = 1.02)
            }}
          }} else {{
            plot.new()
            text(0.5, 0.5, "No splits selected\\n(minimum CVRE selects root only)", cex = 1.2)
          }}
        }})
        """
    )
    ro.r(plot_code)

    return output_path