"""Piecewise quantile regression with wild-bootstrap confidence intervals.

Core math — no plotting, no file paths, no pipeline logic.

A *piecewise linear quantile regression* fits a model of the form:

    Q_τ(Y | X) = β₀ + β₁·X + β₂·(X − ψ)₊

where (X − ψ)₊ = max(0, X − ψ) is the "hinge" at the unknown breakpoint ψ.
The model is estimated by minimizing the check-loss (quantile loss) jointly
over (β₀, β₁, β₂, ψ) using a profile-likelihood grid search over ψ.

For confidence intervals, we use the **wild bootstrap** method adapted for
quantile regression (Feng, He & Hu 2011), which resamples the quantile
residual signs using a two-point distribution.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _fmt_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _progress_bar(current: int, total: int, t0: float, width: int = 30,
                  prefix: str = "") -> str:
    """Build a progress-bar string with ETA."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - t0
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = _fmt_eta(eta)
    else:
        eta_str = "--"
    elapsed_str = _fmt_eta(elapsed)
    return f"\r{prefix}|{bar}| {current}/{total}  elapsed {elapsed_str}  ETA {eta_str}  "


# ═════════════════════════════════════════════════════════════════════
# Check loss (quantile loss / pinball loss)
# ═════════════════════════════════════════════════════════════════════


def _check_loss(residuals: np.ndarray, tau: float) -> float:
    """Quantile check loss  ρ_τ(u) = u·(τ − 𝟙(u<0))."""
    return np.sum(residuals * (tau - (residuals < 0).astype(float)))


# ═════════════════════════════════════════════════════════════════════
# Design matrix construction
# ═════════════════════════════════════════════════════════════════════


def _design_matrix(x: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """Build piecewise-linear design matrix.

    For k breakpoints ψ₁, …, ψ_k the model is:

        Q_τ(Y | X) = β₀ + β₁·X + β₂·(X − ψ₁)₊ + … + β_{k+1}·(X − ψ_k)₊

    Returns X_design of shape (n, k+2): [1, x, (x − ψ₁)₊, …, (x − ψ_k)₊].
    """
    n = len(x)
    k = len(breakpoints)
    X = np.empty((n, k + 2))
    X[:, 0] = 1.0
    X[:, 1] = x
    for j, bp in enumerate(breakpoints):
        X[:, j + 2] = np.maximum(0.0, x - bp)
    return X


# ═════════════════════════════════════════════════════════════════════
# Linear quantile regression solver (interior-point simplex)
# ═════════════════════════════════════════════════════════════════════


def _fit_linear_qr(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Solve the linear quantile regression sub-problem for fixed design.

    Uses scipy's ``linprog`` to solve the LP reformulation of the check-loss
    minimisation problem.

    Parameters
    ----------
    X : (n, p)
        Design matrix.
    y : (n,)
        Response.
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    coefficients : (p,)
    """
    from scipy.optimize import linprog

    n, p = X.shape

    # LP formulation:
    #   min_{β, u, v}  τ·1ᵀu + (1−τ)·1ᵀv
    #   s.t.  Xβ + u − v = y,  u ≥ 0, v ≥ 0
    #
    # Decision variables: [β (p), u (n), v (n)]  →  p + 2n
    c = np.concatenate([
        np.zeros(p),
        tau * np.ones(n),
        (1 - tau) * np.ones(n),
    ])

    # Equality constraint: Xβ + I·u − I·v = y
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = y

    # Bounds: β free, u ≥ 0, v ≥ 0
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    return res.x[:p]


# ═════════════════════════════════════════════════════════════════════
# Profile grid search for breakpoints
# ═════════════════════════════════════════════════════════════════════


def _profile_search(
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    n_breakpoints: int = 1,
    grid_size: int = 50,
    search_range: Tuple[float, float] = (0.10, 0.90),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Find optimal breakpoint(s) via profile grid search.

    For each candidate breakpoint (combination), solve the linear QR
    sub-problem and pick the one with lowest check-loss.

    Parameters
    ----------
    x, y : arrays
        Predictor and response.
    tau : float
        Quantile level.
    n_breakpoints : int
        Number of breakpoints (default 1).
    grid_size : int
        Number of grid points per breakpoint dimension.
    search_range : (float, float)
        Quantile range for the grid, e.g. (0.10, 0.90) means the grid
        spans from the 10th to the 90th percentile of *x*.

    Returns
    -------
    best_breakpoints : (k,) ndarray
    best_coefficients : (k+2,) ndarray
    best_loss : float
    """
    lo = np.quantile(x, search_range[0])
    hi = np.quantile(x, search_range[1])
    grid = np.linspace(lo, hi, grid_size)

    if n_breakpoints == 1:
        candidates = grid.reshape(-1, 1)
    elif n_breakpoints == 2:
        candidates = np.array(
            [[g1, g2] for g1 in grid for g2 in grid if g1 < g2]
        )
    else:
        # Generalise to k breakpoints via itertools
        from itertools import combinations
        candidates = np.array(list(combinations(grid, n_breakpoints)))

    best_loss = np.inf
    best_bp = candidates[0]
    best_coef = None

    for bp_vec in candidates:
        X = _design_matrix(x, bp_vec)
        try:
            coef = _fit_linear_qr(X, y, tau)
        except RuntimeError:
            continue
        residuals = y - X @ coef
        loss = _check_loss(residuals, tau)
        if loss < best_loss:
            best_loss = loss
            best_bp = bp_vec
            best_coef = coef

    if best_coef is None:
        raise RuntimeError("All grid-search candidates failed.")

    return best_bp, best_coef, best_loss


# ═════════════════════════════════════════════════════════════════════
# Prediction helper
# ═════════════════════════════════════════════════════════════════════


def predict(
    x: np.ndarray,
    coefficients: np.ndarray,
    breakpoints: np.ndarray,
) -> np.ndarray:
    """Evaluate the piecewise-linear quantile regression at *x*.

    Parameters
    ----------
    x : (n,)  predictor values.
    coefficients : (k+2,)  [intercept, slope, Δ-slopes…].
    breakpoints : (k,)  breakpoint locations.

    Returns
    -------
    y_hat : (n,)
    """
    X = _design_matrix(x, breakpoints)
    return X @ coefficients


# ═════════════════════════════════════════════════════════════════════
# Wild bootstrap
# ═════════════════════════════════════════════════════════════════════


def wild_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    n_breakpoints: int = 1,
    n_boot: int = 500,
    confidence: float = 0.90,
    grid_size: int = 50,
    search_range: Tuple[float, float] = (0.10, 0.90),
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Wild-bootstrap confidence intervals for piecewise quantile regression.

    The wild bootstrap for quantile regression (Feng, He & Hu 2011):

    1. Fit the original model → β̂, ψ̂
    2. Compute residuals  ê = Y − X(ψ̂)β̂
    3. For b = 1, …, B:
       a. Draw w_i ~ two-point distribution:
          w_i = { −(√5−1)/2  w.p. (√5+1)/(2√5),
                   (√5+1)/2  w.p. (√5−1)/(2√5) }   (Mammen's weights)
       b. Construct Y* = X(ψ̂)β̂ + w ⊙ ê
       c. Re-fit the piecewise QR on (X, Y*) → β̂*, ψ̂*
    4. Build CI from quantiles of (β̂*, ψ̂*).

    Parameters
    ----------
    x, y : arrays (n,)
    tau : float
        Quantile level.
    n_breakpoints : int
        Number of breakpoints.
    n_boot : int
        Number of bootstrap replicates (default 500).
    confidence : float
        Confidence level (default 0.90 → 90 % CI).
    grid_size : int
        Grid size for the profile search.
    search_range : (float, float)
        Quantile range for breakpoint search grid.
    random_state : int, optional

    Returns
    -------
    dict with keys:
        "coefficients" : (k+2,) original coefficients
        "breakpoints"  : (k,)   original breakpoints
        "boot_coefs"   : (B, k+2) bootstrap coefficient matrix
        "boot_bps"     : (B, k)   bootstrap breakpoint matrix
        "ci_lower"     : (k+2+k,) lower CI bounds [coefs, bps]
        "ci_upper"     : (k+2+k,) upper CI bounds [coefs, bps]
        "param_names"  : list of str
    """
    rng = np.random.default_rng(random_state)

    # ── Original fit ─────────────────────────────────────────────────
    bp, coef, _ = _profile_search(
        x, y, tau,
        n_breakpoints=n_breakpoints,
        grid_size=grid_size,
        search_range=search_range,
    )
    X_orig = _design_matrix(x, bp)
    y_hat = X_orig @ coef
    resid = y - y_hat

    n = len(x)

    # Mammen two-point weights
    a1 = -(np.sqrt(5) - 1) / 2
    a2 = (np.sqrt(5) + 1) / 2
    p1 = (np.sqrt(5) + 1) / (2 * np.sqrt(5))

    # ── Bootstrap ────────────────────────────────────────────────────
    boot_coefs = np.empty((n_boot, len(coef)))
    boot_bps = np.empty((n_boot, n_breakpoints))
    t0_boot = time.time()

    for b in range(n_boot):
        # Mammen weights
        u = rng.random(n)
        w = np.where(u < p1, a1, a2)

        # Perturbed response
        y_star = y_hat + w * resid

        try:
            bp_b, coef_b, _ = _profile_search(
                x, y_star, tau,
                n_breakpoints=n_breakpoints,
                grid_size=grid_size,
                search_range=search_range,
            )
            boot_coefs[b] = coef_b
            boot_bps[b] = bp_b
        except RuntimeError:
            boot_coefs[b] = np.nan
            boot_bps[b] = np.nan

        # progress bar (update every 5% or at least every iteration for small n_boot)
        if (b + 1) % max(1, n_boot // 20) == 0 or b == n_boot - 1:
            sys.stdout.write(_progress_bar(
                b + 1, n_boot, t0_boot,
                prefix=f"    bootstrap τ={tau:.2f} ",
            ))
            sys.stdout.flush()

    sys.stdout.write("\n")

    # ── Confidence intervals ─────────────────────────────────────────
    alpha = 1 - confidence
    all_params = np.hstack([boot_coefs, boot_bps])  # (B, k+2+k)

    ci_lower = np.nanquantile(all_params, alpha / 2, axis=0)
    ci_upper = np.nanquantile(all_params, 1 - alpha / 2, axis=0)

    # Parameter names
    param_names = ["intercept", "slope_1"]
    for j in range(n_breakpoints):
        param_names.append(f"delta_slope_{j+1}")
    for j in range(n_breakpoints):
        param_names.append(f"breakpoint_{j+1}")

    return {
        "coefficients": coef,
        "breakpoints": bp,
        "boot_coefs": boot_coefs,
        "boot_bps": boot_bps,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "param_names": param_names,
    }


# ═════════════════════════════════════════════════════════════════════
# Multi-quantile wrapper
# ═════════════════════════════════════════════════════════════════════


@dataclass
class PiecewiseQRResult:
    """Holds piecewise quantile regression results for one quantile level."""
    tau: float
    coefficients: np.ndarray
    breakpoints: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    param_names: List[str]
    boot_coefs: np.ndarray
    boot_bps: np.ndarray

    @property
    def n_breakpoints(self) -> int:
        return len(self.breakpoints)

    @property
    def all_params(self) -> np.ndarray:
        """[coefficients, breakpoints] concatenated."""
        return np.concatenate([self.coefficients, self.breakpoints])


def fit_multi_quantile(
    x: np.ndarray,
    y: np.ndarray,
    taus: Sequence[float] = None,
    *,
    n_breakpoints: int = 1,
    n_boot: int = 500,
    confidence: float = 0.90,
    grid_size: int = 50,
    search_range: Tuple[float, float] = (0.10, 0.90),
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Dict[float, PiecewiseQRResult]:
    """Fit piecewise QR + wild-bootstrap CI at multiple quantile levels.

    Parameters
    ----------
    x, y : (n,) arrays
    taus : sequence of floats
        Quantile levels.  Default: ``np.arange(0.10, 0.91, 0.05)``.
    n_breakpoints : int
    n_boot : int
    confidence : float
    grid_size : int
    search_range : (float, float)
    random_state : int, optional
    verbose : bool

    Returns
    -------
    dict mapping ``tau → PiecewiseQRResult``
    """
    if taus is None:
        taus = np.round(np.arange(0.10, 0.91, 0.05), 2)

    n_taus = len(taus)
    t0_all = time.time()

    results: Dict[float, PiecewiseQRResult] = {}
    for i_tau, tau in enumerate(taus):
        if verbose:
            print(f"\n  [{i_tau+1}/{n_taus}] τ = {tau:.2f}  "
                  f"(n_boot={n_boot}) …", flush=True)
        res = wild_bootstrap_ci(
            x, y, tau,
            n_breakpoints=n_breakpoints,
            n_boot=n_boot,
            confidence=confidence,
            grid_size=grid_size,
            search_range=search_range,
            random_state=random_state,
        )
        results[tau] = PiecewiseQRResult(
            tau=tau,
            coefficients=res["coefficients"],
            breakpoints=res["breakpoints"],
            ci_lower=res["ci_lower"],
            ci_upper=res["ci_upper"],
            param_names=res["param_names"],
            boot_coefs=res["boot_coefs"],
            boot_bps=res["boot_bps"],
        )
        if verbose:
            bp_str = ", ".join(f"{b:.3f}" for b in res["breakpoints"])
            elapsed = time.time() - t0_all
            eta = elapsed / (i_tau + 1) * (n_taus - i_tau - 1)
            print(f"    → breakpoint(s) = [{bp_str}]  "
                  f"(total elapsed {_fmt_eta(elapsed)}, "
                  f"ETA remaining {_fmt_eta(eta)})")

    return results


# ═════════════════════════════════════════════════════════════════════
# Sample-size sensitivity analysis
# ═════════════════════════════════════════════════════════════════════


@dataclass
class SubsampleResult:
    """Results for one subsample fraction, aggregated over repeats."""
    frac: float
    n_samples: int
    n_repeats: int
    # shape (n_repeats, n_params) — each row is point-estimates from one repeat
    param_estimates: np.ndarray
    # shape (n_repeats, n_params) — lower CI
    ci_lowers: np.ndarray
    # shape (n_repeats, n_params) — upper CI
    ci_uppers: np.ndarray
    param_names: List[str]


def subsample_sensitivity(
    x: np.ndarray,
    y: np.ndarray,
    tau: float = 0.50,
    *,
    n_breakpoints: int = 1,
    fracs: Sequence[float] = None,
    n_repeats: int = 30,
    n_boot: int = 200,
    confidence: float = 0.90,
    grid_size: int = 40,
    search_range: Tuple[float, float] = (0.10, 0.90),
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[float, SubsampleResult], np.ndarray]:
    """Sample-size sensitivity analysis.

    For each subsample fraction ``k%``, repeatedly draw ``n_repeats``
    random subsets of size ``k% × n``, re-fit the piecewise QR with
    wild-bootstrap CIs, and record all parameter estimates + CIs.

    Parameters
    ----------
    x, y : (n,) arrays
    tau : float
        Quantile level for the sensitivity analysis.
    n_breakpoints : int
    fracs : sequence of float
        Subsample fractions (default 0.15 to 1.00 step 0.05).
    n_repeats : int
        Number of random subsets per fraction (default 30).
    n_boot : int
        Bootstrap replicates per fit (default 200).
    confidence : float
    grid_size : int
    search_range : (float, float)
    random_state : int, optional
    verbose : bool

    Returns
    -------
    (results_dict, true_params)
        ``results_dict``: mapping ``frac → SubsampleResult``
        ``true_params``: parameter vector from the full-data fit.
    """
    if fracs is None:
        fracs = np.round(np.arange(0.15, 1.01, 0.05), 2)

    rng = np.random.default_rng(random_state)
    n = len(x)

    # ── Full-data fit (the "truth") ──────────────────────────────────
    if verbose:
        print(f"  Full-data fit (100 %, n_boot={n_boot}) …")
    full_res = wild_bootstrap_ci(
        x, y, tau,
        n_breakpoints=n_breakpoints,
        n_boot=n_boot,
        confidence=confidence,
        grid_size=grid_size,
        search_range=search_range,
        random_state=random_state,
    )
    true_params = np.concatenate([full_res["coefficients"], full_res["breakpoints"]])
    param_names = full_res["param_names"]
    n_params = len(true_params)

    # ── Subsample loop ───────────────────────────────────────────────
    n_fracs = len(fracs)
    total_fits = n_fracs * n_repeats
    fit_count = 0
    t0_sens = time.time()

    results: Dict[float, SubsampleResult] = {}
    for i_frac, frac in enumerate(fracs):
        k = max(int(round(frac * n)), n_breakpoints + 3)  # absolute minimum
        if verbose:
            print(f"\n  [{i_frac+1}/{n_fracs}] frac = {frac:.0%} "
                  f"({k} sites) × {n_repeats} repeats  "
                  f"(n_boot={n_boot} each) …",
                  flush=True)

        estimates = np.full((n_repeats, n_params), np.nan)
        ci_lo = np.full((n_repeats, n_params), np.nan)
        ci_hi = np.full((n_repeats, n_params), np.nan)

        t0_frac = time.time()
        for rep in range(n_repeats):
            idx = rng.choice(n, size=k, replace=False)
            x_sub = x[idx]
            y_sub = y[idx]

            try:
                res = wild_bootstrap_ci(
                    x_sub, y_sub, tau,
                    n_breakpoints=n_breakpoints,
                    n_boot=n_boot,
                    confidence=confidence,
                    grid_size=grid_size,
                    search_range=search_range,
                    random_state=None,  # differ across repeats
                )
                estimates[rep] = np.concatenate([
                    res["coefficients"], res["breakpoints"],
                ])
                ci_lo[rep] = res["ci_lower"]
                ci_hi[rep] = res["ci_upper"]
            except Exception:
                pass  # leave as NaN

            fit_count += 1
            # progress bar per repeat within this fraction
            if (rep + 1) % max(1, n_repeats // 10) == 0 or rep == n_repeats - 1:
                sys.stdout.write(_progress_bar(
                    rep + 1, n_repeats, t0_frac,
                    prefix=f"    repeat ",
                ))
                sys.stdout.flush()

        # summary line for this fraction
        if verbose:
            elapsed_total = time.time() - t0_sens
            eta_total = (elapsed_total / fit_count
                         * (total_fits - fit_count)) if fit_count else 0
            sys.stdout.write(
                f"\n    ✓ frac {frac:.0%} done  "
                f"(overall {fit_count}/{total_fits} fits, "
                f"elapsed {_fmt_eta(elapsed_total)}, "
                f"ETA {_fmt_eta(eta_total)})\n"
            )
            sys.stdout.flush()

        results[frac] = SubsampleResult(
            frac=frac,
            n_samples=k,
            n_repeats=n_repeats,
            param_estimates=estimates,
            ci_lowers=ci_lo,
            ci_uppers=ci_hi,
            param_names=param_names,
        )

    return results, true_params
