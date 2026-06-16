#!/usr/bin/env python3
"""
Brainage / epiage analysis — Python conversion of the original R script.

R → Python equivalents
  tidyverse / dplyr   → pandas
  ggplot2             → matplotlib + seaborn
  metafor::rma.mv     → statsmodels MixedLM (approximation; see note in _pool_group)

Install dependencies:
  pip install pandas numpy matplotlib seaborn statsmodels
"""

import math
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
results_dir = "/Users/vb506/Documents/dashboard/data"
plots_dir   = "/Users/vb506/Documents/dashboard/plots"
os.makedirs(plots_dir, exist_ok=True)

# ── Model colour palette ──────────────────────────────────────────────────────
# Defined upfront so both plots can use it (in the original R script it appeared
# mid-way through, after the first plot block).
MODEL_PALETTE = {
    "PCBrainAge":    "#222222",
    "PCGrimAge":     "#f3c300",
    "CorticalClock": "#875692",
    "cAge":          "#8db600",
    "AltumAge":      "#a1caf1",
    "Horvath2013":   "#be0032",
    "DamAge":        "#c2b280",
    "AdaptAge":      "#848482",
    "Hannum":        "#008856",
    "PhenoAge":      "#e68fac",
    "skinHorvath":   "#0067a5",
    "PedBE":         "#f99379",
    "Wu":            "#604e97",
    "ZhangBLUP":     "#f6a600",
    "ZhangEN":       "#b3446c",
    "Bohlin":        "#dcd300",
    "Knight":        "#654522",
    "EPIC":          "#2b3d26",
    "DunedinPACE":   "#e25822",
}

BRAIN_MODEL_PALETTE = {
    "Pyment":      "#222222",
    "DevBrainAge": "#f3c300",
    "Centile2":    "#875692",
    "DBN":         "#8db600",
    "Kaufmann":    "#a1caf1",
    "PyBrainAge":  "#be0032",
    "ENIGMA":      "#c2b280",
}

COHORT_PALETTE = {
    "ALSPAC":           "#222222",
    "BHRC":             "#f3c300",
    "CannTeen":         "#875692",
    "DCHS":             "#f38400",
    "FinnBrain":        "#a1caf1",
    "GenR":             "#be0032",
    "GUSTO":            "#c2b280",
    "K2H Childhood":    "#848482",
    "K2H Infancy":      "#008856",
    "MTwiNS":           "#e68fac",
    "NICAP":            "#0067a5",
    "Oregon ADHD-1000": "#f99379",
    "FFCWS":            "#604e97",
    "TAG":              "#f6a600",
    "UCI Echo":         "#b3446c",
}

# ── Load data ─────────────────────────────────────────────────────────────────
perf_brain = pd.read_csv(os.path.join(results_dir, "perf_brain.csv"))
perf_epi   = pd.read_csv(os.path.join(results_dir, "perf_epi.csv"))
mod2_eff2  = pd.read_csv(os.path.join(results_dir, "mod2_eff2.csv"))

# ── Preprocessing ─────────────────────────────────────────────────────────────
RENAME = {"Bohlin_conv": "Bohlin", "Knight_conv": "Knight", "EPIC_conv": "EPIC"}

perf_epi["model"] = perf_epi["model"].replace(RENAME)

perf_brain_sub = perf_brain[~perf_brain["model"].isin(["DunedinPACNI"])].copy()
perf_epi_sub   = perf_epi[~perf_epi["model"].isin(["DunedinPACE", "DNAmTL"])].copy()

# log(wMAE) and delta-method variance:  Var(log X) ≈ (SE_X / X)^2
perf_epi_sub["log_wMAE"]     = np.log(perf_epi_sub["wMAE_test"])
perf_epi_sub["log_wMAE_var"] = (perf_epi_sub["wMAE_SE_boot"] / perf_epi_sub["wMAE_test"]) ** 2

# log(MAE) — unweighted equivalent
perf_epi_sub["log_MAE"]     = np.log(perf_epi_sub["MAE"])
perf_epi_sub["log_MAE_var"] = (perf_epi_sub["MAE_SE_boot"] / perf_epi_sub["MAE"]) ** 2

# rename gestational clocks (already done for perf_epi above; redo for sub)
perf_epi_sub["model"] = perf_epi_sub["model"].replace(RENAME)

perf_epi_sub_nogest = perf_epi_sub[
    ~perf_epi_sub["model"].isin(["Bohlin", "Knight", "EPIC"])
].copy()

# Brain: variance columns for rma.mv(V = SE^2) equivalent
perf_brain_sub["MAE_var"]  = perf_brain_sub["MAE_SE_boot"] ** 2
perf_brain_sub["wMAE_var"] = perf_brain_sub["wMAE_SE_boot"] ** 2

# Brain 5 age bins
BRAIN_BIN5_LEVELS = [
    "Early childhood\n4–5y",
    "Middle childhood\n6–9y",
    "Late childhood\n10–14y",
    "Adolescence\n15–17y",
    "Young adulthood\n18+y",
]

def _bin_brain5(age: float) -> str:
    if age < 6:  return "Early childhood\n4–5y"
    if age < 10: return "Middle childhood\n6–9y"
    if age < 15: return "Late childhood\n10–14y"
    if age < 18: return "Adolescence\n15–17y"
    return "Young adulthood\n18+y"


# ── Meta-analysis ─────────────────────────────────────────────────────────────
# R equivalent:
#   rma.mv(yi=log_wMAE, V=log_wMAE_var,
#          random = ~1 | cohort/timepoint, data=dat, method="REML")
#
# Python has no direct equivalent for a 3-level meta-analytic model with
# *known* sampling variances.  We use:
#   • statsmodels MixedLM (REML, grouped by cohort) when ≥2 cohorts exist
#   • inverse-variance weighted mean + 95 % CI as a fallback
#
# For exact numerical equivalence with metafor, use rpy2 to call R from Python.

def _pool_group(dat: pd.DataFrame, yi_col: str = "log_wMAE", vi_col: str = "log_wMAE_var"):
    """
    Compute pooled estimate and 95 % CI for one (group) slice.

    Four-step convergence cascade — mirrors the R fallback loop used in the
    unweighted-MAE and Pearson violin analyses (R script lines ~2761 / ~3880):
      1) MixedLM + L-BFGS-B  (fast default)
      2) MixedLM + Nelder-Mead  (more robust, slower)
      3) MixedLM without explicit random-slope term  (simpler RE structure)
      4) IV-weighted fixed-effect mean  (always converges)

    The same fallback order applies everywhere: violin age-bin metas,
    modelwise metas, LOO sensitivity loops, etc.

    Returns (mu, lower, upper, method_label).
    """
    yi = dat[yi_col].values
    vi = dat[vi_col].values

    if len(dat) == 1:
        return yi[0], np.nan, np.nan, "single estimate"

    if dat["cohort"].nunique() >= 2:
        d = dat.copy()
        d["const"] = 1.0

        # Steps 1 & 2 — full RE model, different optimisers
        for method, label in [("lbfgs", "RE lbfgs"), ("nm", "RE nm")]:
            try:
                fit = MixedLM(
                    endog=d[yi_col],
                    exog=d[["const"]],
                    groups=d["cohort"],
                    exog_re=d[["const"]],
                ).fit(reml=True, method=method, disp=False)
                if not fit.converged:
                    raise RuntimeError("not converged")
                mu = float(fit.fe_params["const"])
                se = float(fit.bse_fe["const"])
                if np.isnan(mu) or np.isnan(se) or se <= 0:
                    raise RuntimeError("invalid params")
                return mu, mu - 1.96 * se, mu + 1.96 * se, label
            except Exception:
                continue

        # Step 3 — simpler RE: no explicit exog_re (cohort intercepts only)
        try:
            fit = MixedLM(
                endog=d[yi_col],
                exog=d[["const"]],
                groups=d["cohort"],
            ).fit(reml=True, disp=False)
            mu = float(fit.fe_params["const"])
            se = float(fit.bse_fe["const"])
            if not (np.isnan(mu) or np.isnan(se) or se <= 0):
                return mu, mu - 1.96 * se, mu + 1.96 * se, "RE simple"
        except Exception:
            pass

    # Step 4 — IV-weighted fixed-effect (always converges)
    w  = 1.0 / vi
    mu = float((w * yi).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    return mu, mu - 1.96 * se, mu + 1.96 * se, "IV-weighted (FE)"


def compute_meta(
    df: pd.DataFrame,
    yi_col: str = "log_wMAE",
    vi_col: str = "log_wMAE_var",
) -> pd.DataFrame:
    """Run meta-analysis across all (age_bin, model) combinations."""
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]
    rows  = []

    for (age_bin, model), dat in valid.groupby(["age_bin", "model"], observed=True):
        mu, lb, ub, meth = _pool_group(dat, yi_col=yi_col, vi_col=vi_col)
        rows.append({
            "age_bin":        age_bin,
            "model":          model,
            "pooled_log_mae": mu,
            "ci_lb_log":      lb,
            "ci_ub_log":      ub,
            "pooled_mae":     np.exp(mu),
            "ci_lb":          np.exp(lb) if not np.isnan(lb) else np.nan,
            "ci_ub":          np.exp(ub) if not np.isnan(ub) else np.nan,
            "k":              len(dat),
            "meta_model":     meth,
        })

    out = pd.DataFrame(rows)
    # preserve ordered-categorical dtype for age_bin if present in source
    if not out.empty and hasattr(df["age_bin"], "cat"):
        out["age_bin"] = pd.Categorical(
            out["age_bin"],
            categories=df["age_bin"].cat.categories,
            ordered=True,
        )
    return out


# ── Modelwise meta-analysis ───────────────────────────────────────────────────

def compute_modelwise_meta(
    df: pd.DataFrame,
    yi_col: str,
    vi_col: str,
    back_transform: bool = False,
) -> pd.DataFrame:
    """
    Per-model pooled estimate + one 'Pooled' row across all models.
    Equivalent to rma.mv with ~model - 1 as moderator.

    back_transform : apply exp() to results — use when yi_col is log-scale
                     (e.g. log_wMAE) and you want raw-scale output.
    """
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]

    def _make_row(model_name, dat):
        mu, lb, ub, meth = _pool_group(dat, yi_col=yi_col, vi_col=vi_col)
        if back_transform:
            mu = np.exp(mu)
            lb = np.exp(lb) if not np.isnan(lb) else np.nan
            ub = np.exp(ub) if not np.isnan(ub) else np.nan
        return {"model": model_name, "pooled_val": mu, "ci_lb": lb,
                "ci_ub": ub, "k": len(dat), "meta_model": meth}

    rows = [_make_row("Pooled", valid)]
    for model, dat in valid.groupby("model", observed=True):
        rows.append(_make_row(model, dat))

    return pd.DataFrame(rows)


# ── Forest / caterpillar plot ─────────────────────────────────────────────────

_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', 'h', '<', '>', 'H', '8', '+']


def forest_plot(
    raw_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    model_order: list,
    x_col: str,
    filename: Optional[str] = None,
    title: str = "Model Performance",
    x_label: str = "Weighted MAE",
    # categorical colour: pass a dict {value: hex}
    # continuous colour:  pass None → uses color_cmap over color_col values
    color_col: str = "cohort",
    color_palette: Optional[dict] = None,
    color_cmap: str = "magma",
    color_label: str = "Mean age\nat time point",
    shape_col: Optional[str] = None,
    x_log: bool = False,
    x_limits: Optional[tuple] = None,
    x_breaks: Optional[list] = None,
    x_break_labels: Optional[list] = None,
    # list of (y_position, linestyle) e.g. [(1.5, "solid"), (5.5, "dashed")]
    dividers: Optional[list] = None,
    # vertical reference line on x-axis (e.g. x=0 for Pearson r plots)
    x_refline: Optional[float] = None,
    meta_label: str = "Meta-analysis",
    figsize: tuple = (11, 7),
    dpi: int = 300,
    jitter_height: float = 0.12,
) -> None:
    """
    Horizontal caterpillar plot: jittered raw observations + meta CI + diamond.
    Equivalent to the ggplot2 forest-plot blocks in the original R script.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_map = {m: i for i, m in enumerate(model_order)}
    rng   = np.random.default_rng(42)

    # shape mapping (cohort → marker)
    if shape_col is not None:
        shape_vals = sorted(raw_df[shape_col].dropna().unique())
        shape_map  = {s: _MARKERS[i % len(_MARKERS)] for i, s in enumerate(shape_vals)}
    else:
        shape_map, shape_vals = {}, []

    categorical = color_palette is not None

    # ── raw jittered points ──────────────────────────────────────────────────
    if categorical:
        for _, row in raw_df.iterrows():
            yi = y_map.get(row["model"])
            if yi is None or pd.isna(row[x_col]):
                continue
            col = color_palette.get(row[color_col], "#aaaaaa")
            mkr = shape_map.get(row[shape_col], "o") if shape_col else "o"
            ax.scatter(row[x_col],
                       yi + rng.uniform(-jitter_height, jitter_height),
                       color=col, marker=mkr, alpha=0.45, s=18, zorder=3)
    else:
        # continuous colour (e.g. mean_age) optionally paired with marker per cohort
        c_vals = raw_df[color_col].dropna().values
        norm   = plt.Normalize(vmin=c_vals.min(), vmax=c_vals.max())
        cmap   = plt.get_cmap(color_cmap)

        for _, row in raw_df.iterrows():
            yi = y_map.get(row["model"])
            if yi is None or pd.isna(row[x_col]) or pd.isna(row[color_col]):
                continue
            col = cmap(norm(row[color_col]))
            mkr = shape_map.get(row[shape_col], "o") if shape_col else "o"
            ax.scatter(row[x_col],
                       yi + rng.uniform(-jitter_height, jitter_height),
                       color=col, marker=mkr, alpha=0.65, s=18, zorder=3)

        sm = plt.cm.ScalarMappable(cmap=color_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.35, aspect=15, pad=0.01)
        cbar.set_label(color_label, fontsize=10)

    # ── meta CI segments + diamond points ────────────────────────────────────
    for _, row in meta_df.iterrows():
        yi = y_map.get(row["model"])
        if yi is None:
            continue
        if not (np.isnan(row["ci_lb"]) or np.isnan(row["ci_ub"])):
            ax.hlines(yi, row["ci_lb"], row["ci_ub"],
                      color="black", linewidth=1.1, zorder=5)
        ax.scatter(row["pooled_val"], yi,
                   color="black", marker="D", s=45, zorder=6)

    # ── dividers ─────────────────────────────────────────────────────────────
    if dividers:
        for y_pos, lstyle in dividers:
            ax.axhline(y_pos, linestyle=lstyle, linewidth=0.4,
                       color="#666666", zorder=2)

    # ── axes ─────────────────────────────────────────────────────────────────
    if x_refline is not None:
        ax.axvline(x_refline, color="#d9d9d9", linewidth=0.8, zorder=1)

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=11)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(None)
    ax.set_title(title, fontsize=16, pad=12)

    if x_log:
        ax.set_xscale("log")
    if x_limits:
        ax.set_xlim(x_limits)
    if x_breaks is not None:
        ax.set_xticks(x_breaks)
        ax.set_xticklabels(x_break_labels if x_break_labels else [str(b) for b in x_breaks])

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.5)
    sns.despine(ax=ax, left=True)

    # ── legend ───────────────────────────────────────────────────────────────
    if categorical:
        present = set(raw_df[color_col].dropna())
        handles = [
            mpatches.Patch(color=color_palette.get(c, "#aaaaaa"), label=c, alpha=0.7)
            for c in sorted(color_palette.keys()) if c in present
        ]
        handles.append(
            plt.Line2D([0], [0], marker="D", color="black",
                       markerfacecolor="black", markersize=7,
                       linestyle="-", label=meta_label)
        )
        ax.legend(handles=handles, title="Source",
                  loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, fontsize=8, title_fontsize=10, frameon=False)
    elif shape_col is not None:
        present_shapes = set(raw_df[shape_col].dropna())
        handles = [
            plt.Line2D([0], [0], marker=shape_map.get(s, "o"), color="#555555",
                       markerfacecolor="#555555", markersize=6,
                       linestyle="None", label=s)
            for s in shape_vals if s in present_shapes
        ]
        ax.legend(handles=handles, title="Cohort",
                  loc="upper left", bbox_to_anchor=(1.15, 1),
                  borderaxespad=0, fontsize=8, title_fontsize=10, frameon=False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {filename}")
    else:
        plt.show()
    plt.close()


# ── Plotting helper ───────────────────────────────────────────────────────────

def violin_epi_plot(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    bin_levels: list,
    counts: Optional[pd.DataFrame] = None,
    filename: Optional[str] = None,
    title: str = "Epigenetic clock performance across age groups",
    y_col: str = "log_wMAE",
    y_label: str = "log(weighted MAE)",
    dpi: int = 300,
    figsize: tuple = (13, 6.5),
) -> None:
    """
    Violin + boxplot of log(MAE) per age bin, with dodged meta-analysis
    points and CI bars overlaid per model.

    Equivalent to the ggplot2 block with geom_violin + geom_boxplot +
    geom_errorbar + geom_point + geom_text in the original R script.

    Parameters
    ----------
    y_col   : column in df to use as the y axis (e.g. "log_wMAE" or "log_MAE")
    y_label : y-axis label string
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_map = {b: i for i, b in enumerate(bin_levels)}

    # reference line at 0  (geom_hline)
    ax.axhline(0, color="#ebebeb", linewidth=0.5, zorder=0)

    # per-bin data arrays (ordered)
    bin_arrays = [
        df.loc[df["age_bin"] == b, y_col].dropna().values
        for b in bin_levels
    ]

    # geom_violin
    vp = ax.violinplot(
        bin_arrays,
        positions=range(len(bin_levels)),
        widths=0.7,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#d9d9d9")   # ~grey85
        body.set_alpha(1.0)
        body.set_edgecolor("none")

    # geom_boxplot (no outliers, transparent fill)
    bp = ax.boxplot(
        bin_arrays,
        positions=range(len(bin_levels)),
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for elem in ("whiskers", "caps", "medians"):
        for ln in bp[elem]:
            ln.set_color("#333333")
            ln.set_linewidth(0.8)

    # compute dodge offsets matching position_dodge(width = 0.7)
    all_models = sorted(meta["model"].unique(), key=str.lower)
    n_m = len(all_models)
    offsets = np.linspace(-0.32, 0.32, n_m) if n_m > 1 else np.array([0.0])
    off_map = dict(zip(all_models, offsets))

    # geom_errorbar + geom_point (per model, dodged)
    for _, row in meta.iterrows():
        xi = x_map.get(row["age_bin"])
        if xi is None:
            continue
        col  = MODEL_PALETTE.get(row["model"], "#888888")
        xpos = xi + off_map.get(row["model"], 0.0)

        ax.scatter(xpos, row["pooled_log_mae"],
                   color=col, s=18, alpha=0.65, zorder=5)

        if not (np.isnan(row["ci_lb_log"]) or np.isnan(row["ci_ub_log"])):
            ax.vlines(xpos, row["ci_lb_log"], row["ci_ub_log"],
                      color=col, linewidth=0.8, alpha=0.45, zorder=4)
            cw = 0.035
            for ycap in (row["ci_lb_log"], row["ci_ub_log"]):
                ax.hlines(ycap, xpos - cw, xpos + cw,
                          color=col, linewidth=0.8, alpha=0.45, zorder=4)

    # y-axis limits + k= labels (geom_text + coord_cartesian)
    flat = [v for arr in bin_arrays for v in arr]
    y_min_data = min(flat) if flat else 0.0
    y_max_data = max(flat) if flat else 1.0
    y_min_meta = meta["ci_lb_log"].min(skipna=True) if not meta.empty else y_min_data
    y_max_meta = meta["ci_ub_log"].max(skipna=True) if not meta.empty else y_max_data
    y_min = min(y_min_data, y_min_meta)
    y_max = max(y_max_data, y_max_meta)
    y_range = y_max - y_min
    y_pad_bottom = 0.08 * y_range
    y_pad_top    = 0.15 * y_range  # extra headroom so top points aren't clipped
    y_pad        = y_pad_bottom     # used for k= label placement below

    if counts is not None:
        cnt_map = dict(zip(counts["age_bin"], counts["label"]))
        for b, xi in x_map.items():
            lbl = cnt_map.get(b)
            if lbl:
                ax.text(xi, y_min - 0.35 * y_pad, lbl,
                        ha="center", va="top", fontsize=9)

    ax.set_xlim(-0.6, len(bin_levels) - 0.4)
    ax.set_ylim(y_min - y_pad_bottom, y_max + y_pad_top)
    ax.set_xticks(range(len(bin_levels)))
    ax.set_xticklabels(bin_levels, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Developmental age group", fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, left=True, bottom=False)

    # legend — placed outside the axes to the right so it never overlaps data
    present = set(meta["model"])
    handles = [
        mpatches.Patch(color=MODEL_PALETTE.get(m, "#888888"), label=m)
        for m in sorted(MODEL_PALETTE, key=str.lower)
        if m in present
    ]
    ax.legend(
        handles=handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=9,
        title_fontsize=11,
        frameon=False,
    )

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {filename}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — 5 age bins  (first ggplot block in the original R script)
# ══════════════════════════════════════════════════════════════════════════════

BIN5_LEVELS = [
    "Infancy\n<2y",
    "Early childhood\n2–5y",
    "Middle childhood\n6–11y",
    "Adolescence\n12–17y",
    "Adult\n18+y",
]

def _bin5(age: float) -> str:
    if age < 2:  return "Infancy\n<2y"
    if age < 6:  return "Early childhood\n2–5y"
    if age < 12: return "Middle childhood\n6–11y"
    if age < 18: return "Adolescence\n12–17y"
    return "Adult\n18+y"

epi_b5 = perf_epi_sub.copy()
epi_b5["age_bin"] = pd.Categorical(
    epi_b5["mean_age"].map(_bin5), categories=BIN5_LEVELS, ordered=True
)

meta_b5 = compute_meta(epi_b5)

# count unique cohort-timepoints per bin (same as epi_age_bin_counts in R)
counts_b5 = (
    epi_b5
    .assign(ct=epi_b5["cohort"].astype(str) + " " + epi_b5["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_b5["label"] = "k=" + counts_b5["k"].astype(str)

violin_epi_plot(
    df=epi_b5,
    meta=meta_b5,
    bin_levels=BIN5_LEVELS,
    counts=None,       # original R plot 1 had no k labels
    filename=os.path.join(plots_dir, "plot_violin_perf_epi_age_brackets.png"),
    title="Epigenetic clock performance across age groups",
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — 7 age bins with k= labels  ("USED" version in the original R script)
# ══════════════════════════════════════════════════════════════════════════════

BIN7_LEVELS = [
    "birth\n<3m",
    "Infancy\n<3y",
    "Early childhood\n3–5y",
    "Middle childhood\n6–9y",
    "Late childhood\n10–14y",
    "Adolescence\n15–17y",
    "Young adulthood\n18+y",
]

def _bin7(age: float) -> str:
    if age < 0.25: return "birth\n<3m"
    if age < 3:    return "Infancy\n<3y"
    if age < 6:    return "Early childhood\n3–5y"
    if age < 10:   return "Middle childhood\n6–9y"
    if age < 15:   return "Late childhood\n10–14y"
    if age < 18:   return "Adolescence\n15–17y"
    return "Young adulthood\n18+y"

epi_b7 = perf_epi_sub.copy()
epi_b7["age_bin"] = pd.Categorical(
    epi_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
)

meta_b7 = compute_meta(epi_b7)

counts_b7 = (
    epi_b7
    .assign(ct=epi_b7["cohort"].astype(str) + " " + epi_b7["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_b7["label"] = "k=" + counts_b7["k"].astype(str)

violin_epi_plot(
    df=epi_b7,
    meta=meta_b7,
    bin_levels=BIN7_LEVELS,
    counts=counts_b7,
    filename=os.path.join(plots_dir, "plot_violin_perf_epi_age_brackets_morebrackets_l.png"),
    title="Epigenetic clock performance across age groups",
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — 7 age bins, gestational clocks excluded  ("nogest" version)
# ══════════════════════════════════════════════════════════════════════════════

epi_nogest_b7 = perf_epi_sub_nogest.copy()
epi_nogest_b7["age_bin"] = pd.Categorical(
    epi_nogest_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
)

meta_nogest_b7 = compute_meta(epi_nogest_b7)

counts_nogest_b7 = (
    epi_nogest_b7
    .assign(ct=epi_nogest_b7["cohort"].astype(str) + " " + epi_nogest_b7["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_nogest_b7["label"] = "k=" + counts_nogest_b7["k"].astype(str)

violin_epi_plot(
    df=epi_nogest_b7,
    meta=meta_nogest_b7,
    bin_levels=BIN7_LEVELS,
    counts=counts_nogest_b7,
    filename=os.path.join(plots_dir, "plot_violin_perf_epi_age_brackets_morebrackets_k_nogest.png"),
    title="Epigenetic clock performance across age groups",
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — 7 age bins, unweighted MAE  (log_MAE instead of log_wMAE)
# ══════════════════════════════════════════════════════════════════════════════

epi_unw_b7 = perf_epi_sub.copy()
epi_unw_b7["age_bin"] = pd.Categorical(
    epi_unw_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
)

meta_unw_b7 = compute_meta(epi_unw_b7, yi_col="log_MAE", vi_col="log_MAE_var")

counts_unw_b7 = (
    epi_unw_b7
    .assign(ct=epi_unw_b7["cohort"].astype(str) + " " + epi_unw_b7["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_unw_b7["label"] = "k=" + counts_unw_b7["k"].astype(str)

violin_epi_plot(
    df=epi_unw_b7,
    meta=meta_unw_b7,
    bin_levels=BIN7_LEVELS,
    counts=counts_unw_b7,
    y_col="log_MAE",
    y_label="log(MAE)",
    filename=os.path.join(plots_dir, "plot_violin_perf_epi_age_unweightedMAE_brackets_morebrackets_k.png"),
    title="Epigenetic clock performance across age groups",
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Brain violin, unweighted MAE, 5 age bins
# ══════════════════════════════════════════════════════════════════════════════

brain_mae_b5 = perf_brain_sub.copy()
brain_mae_b5["age_bin"] = pd.Categorical(
    brain_mae_b5["mean_age"].map(_bin_brain5),
    categories=BRAIN_BIN5_LEVELS, ordered=True,
)

meta_brain_mae_b5 = compute_meta(brain_mae_b5, yi_col="MAE", vi_col="MAE_var")

counts_brain_mae_b5 = (
    brain_mae_b5
    .assign(ct=brain_mae_b5["cohort"].astype(str) + " " + brain_mae_b5["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_brain_mae_b5["label"] = "k=" + counts_brain_mae_b5["k"].astype(str)

violin_epi_plot(
    df=brain_mae_b5,
    meta=meta_brain_mae_b5,
    bin_levels=BRAIN_BIN5_LEVELS,
    counts=counts_brain_mae_b5,
    y_col="MAE",
    y_label="MAE",
    filename=os.path.join(plots_dir, "plot_violin_perf_brain_age_unweightedMAE_brackets_morebrackets_k.png"),
    title="Brain age model performance across age groups",
    # override palette so legend reflects brain models only
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Brain violin, weighted MAE, 5 age bins
# ══════════════════════════════════════════════════════════════════════════════

brain_wmae_b5 = perf_brain_sub.copy()
brain_wmae_b5["age_bin"] = pd.Categorical(
    brain_wmae_b5["mean_age"].map(_bin_brain5),
    categories=BRAIN_BIN5_LEVELS, ordered=True,
)

meta_brain_wmae_b5 = compute_meta(brain_wmae_b5, yi_col="wMAE_test", vi_col="wMAE_var")

counts_brain_wmae_b5 = (
    brain_wmae_b5
    .assign(ct=brain_wmae_b5["cohort"].astype(str) + " " + brain_wmae_b5["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["ct"]
    .nunique()
    .reset_index(name="k")
)
counts_brain_wmae_b5["label"] = "k=" + counts_brain_wmae_b5["k"].astype(str)

# temporarily swap MODEL_PALETTE so the legend shows brain models
_orig_palette = MODEL_PALETTE.copy()
MODEL_PALETTE.clear()
MODEL_PALETTE.update(BRAIN_MODEL_PALETTE)

violin_epi_plot(
    df=brain_wmae_b5,
    meta=meta_brain_wmae_b5,
    bin_levels=BRAIN_BIN5_LEVELS,
    counts=counts_brain_wmae_b5,
    y_col="wMAE_test",
    y_label="weighted MAE",
    filename=os.path.join(plots_dir, "plot_violin_perf_brain_age_wMAE_brackets_morebrackets_k.png"),
    title="Brain age model performance across age groups",
)

MODEL_PALETTE.clear()
MODEL_PALETTE.update(_orig_palette)  # restore epi palette

# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Brain modelwise + overall, colour by age, shape by cohort, linear x
# ══════════════════════════════════════════════════════════════════════════════
# Note: R loads pre-computed .rds metafor objects; here we re-derive the same
# estimates in Python using compute_modelwise_meta.

brain_meta_wmae = compute_modelwise_meta(
    perf_brain_sub, yi_col="wMAE_test", vi_col="wMAE_var"
)

# order: Pooled at bottom, models sorted worst→best wMAE going up
_brain_order_raw = (
    brain_meta_wmae[brain_meta_wmae["model"] != "Pooled"]
    .sort_values("pooled_val", ascending=False)["model"].tolist()
)
brain_model_order = ["Pooled"] + _brain_order_raw

forest_plot(
    raw_df=perf_brain_sub,
    meta_df=brain_meta_wmae,
    model_order=brain_model_order,
    x_col="wMAE_test",
    filename=os.path.join(plots_dir, "plot_perf_brain_modelwise_plus_overall_meta_raw_combined.png"),
    title="Brain Age Model Performance",
    x_label="Weighted MAE (MAE ÷ age range)",
    color_col="mean_age",
    color_palette=None,
    color_cmap="magma",
    color_label="Mean age\nat time point",
    shape_col="cohort",
    x_limits=(0, 15),
    x_breaks=[0, 5, 10, 15],
    x_break_labels=["0", "5", "10", "15+"],
    dividers=[(1.5, "solid")],
    figsize=(11, 7),
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Brain modelwise + overall, colour by age, shape by cohort, log x
# ══════════════════════════════════════════════════════════════════════════════

forest_plot(
    raw_df=perf_brain_sub,
    meta_df=brain_meta_wmae,
    model_order=brain_model_order,
    x_col="wMAE_test",
    filename=os.path.join(plots_dir, "plot_perf_brain_modelwise_plus_overall_meta_raw_combined_alt.png"),
    title="Brain Age Model Performance",
    x_label="Weighted MAE (MAE ÷ age range)",
    color_col="mean_age",
    color_palette=None,
    color_cmap="magma",
    color_label="Timepoint\nyounger = darker",
    shape_col="cohort",
    x_log=True,
    x_breaks=[1, 2, 5, 10, 30],
    dividers=[(1.5, "solid")],
    figsize=(11, 7),
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 9 — Brain modelwise + overall, colour by cohort, weighted MAE
# ══════════════════════════════════════════════════════════════════════════════

forest_plot(
    raw_df=perf_brain_sub,
    meta_df=brain_meta_wmae,
    model_order=brain_model_order,
    x_col="wMAE_test",
    filename=os.path.join(plots_dir, "plot_perf_brain_modelwise_plus_overall_meta_raw_combined_cohortcolors.png"),
    title="Brain age model performance (weighted MAE)",
    x_label="Weighted MAE (MAE ÷ age range)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Overall pooled",
    dividers=[(1.5, "solid")],
    figsize=(11, 7),
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 10 — Brain modelwise + overall, colour by cohort, unweighted MAE
# ══════════════════════════════════════════════════════════════════════════════

brain_meta_mae = compute_modelwise_meta(
    perf_brain_sub, yi_col="MAE", vi_col="MAE_var"
)

_brain_mae_order_raw = (
    brain_meta_mae[brain_meta_mae["model"] != "Pooled"]
    .sort_values("pooled_val", ascending=False)["model"].tolist()
)
brain_mae_model_order = ["Pooled"] + _brain_mae_order_raw

forest_plot(
    raw_df=perf_brain_sub,
    meta_df=brain_meta_mae,
    model_order=brain_mae_model_order,
    x_col="MAE",
    filename=os.path.join(plots_dir, "plot_perf_brain_modelwise_plus_overall_meta_raw_combined_cohortcolors_unweightedMAE.png"),
    title="Brain Age Model Performance",
    x_label="MAE",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    dividers=[(1.5, "solid")],
    figsize=(11, 7),
)

# ══════════════════════════════════════════════════════════════════════════════
# Plots 11 & 12 — Epi modelwise + overall, colour by cohort
# Meta was fit on log_wMAE → back-transform with exp() for the forest plot
# ══════════════════════════════════════════════════════════════════════════════

epi_meta_wmae = compute_modelwise_meta(
    perf_epi_sub, yi_col="log_wMAE", vi_col="log_wMAE_var", back_transform=True
)

# Generation labels for ordering
GEN1_GEST = ["Bohlin", "Knight", "EPIC"]
GEN1      = ["ZhangEN", "ZhangBLUP", "Wu", "skinHorvath", "PedBE", "PCBrainAge",
             "Horvath2013", "Hannum", "CorticalClock", "cAge", "AltumAge"]
GEN2_4    = ["PCGrimAge", "PhenoAge", "DamAge", "AdaptAge"]

def _gen_order(models_in_gen, meta_df):
    present = set(meta_df["model"])
    sub = meta_df[meta_df["model"].isin(models_in_gen) & meta_df["model"].isin(present)]
    return sub.sort_values("pooled_val")["model"].tolist()

gen1_gest_ord = _gen_order(GEN1_GEST, epi_meta_wmae)
gen1_ord      = _gen_order(GEN1,      epi_meta_wmae)
gen2_ord      = _gen_order(GEN2_4,    epi_meta_wmae)

epi_model_order = (
    ["Pooled"]
    + list(reversed(gen2_ord))
    + list(reversed(gen1_ord))
    + list(reversed(gen1_gest_ord))
)

divider_overall   = 1.5
divider_gen2_gen1 = 1.5 + len(gen2_ord)
divider_gen1_gest = 1.5 + len(gen2_ord) + len(gen1_ord)

# Plot 11 — linear x
forest_plot(
    raw_df=perf_epi_sub,
    meta_df=epi_meta_wmae,
    model_order=epi_model_order,
    x_col="wMAE_test",
    filename=os.path.join(plots_dir, "plot_perf_epi_modelwise_plus_overall_meta_raw_combined_cohortcolors.png"),
    title="Epigenetic clock performance (weighted MAE)",
    x_label="Weighted MAE (MAE ÷ age range)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    dividers=[
        (divider_overall,   "solid"),
        (divider_gen2_gen1, "dashed"),
        (divider_gen1_gest, "dashed"),
    ],
    figsize=(11, 9),
)

# Plot 12 — log x
forest_plot(
    raw_df=perf_epi_sub,
    meta_df=epi_meta_wmae,
    model_order=epi_model_order,
    x_col="wMAE_test",
    filename=os.path.join(plots_dir, "plot_perf_epi_modelwise_plus_overall_meta_raw_combined_cohortcolors_log.png"),
    title="Epigenetic Clock Performance",
    x_label="Weighted MAE (MAE ÷ age range), log scale",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    x_log=True,
    x_breaks=[1, 2, 5, 10, 20, 50, 100],
    dividers=[
        (divider_overall,   "solid"),
        (divider_gen2_gen1, "dashed"),
        (divider_gen1_gest, "dashed"),
    ],
    figsize=(11, 9),
)

# ══════════════════════════════════════════════════════════════════════════════
# Pearson r utilities (Fisher z → r back-transformation)
# ══════════════════════════════════════════════════════════════════════════════

def transf_ztor(z):
    """Fisher z → Pearson r.  Equivalent to R's transf.ztor()."""
    return np.tanh(z)


def _compute_meta_z(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Meta-analysis on Fisher-z scale; back-transforms pooled estimates to
    Pearson r via tanh().  Requires 'pearson_z' and 'pearson_var' in df.
    Returns: group_cols + pooled_r, ci_lb, ci_ub, k, meta_model.
    """
    valid = df.dropna(subset=["pearson_z", "pearson_var"])
    valid = valid[valid["pearson_var"] > 0]
    rows  = []

    for keys, dat in valid.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mu, lb, ub, meth = _pool_group(dat, yi_col="pearson_z", vi_col="pearson_var")
        row = dict(zip(group_cols, keys))
        row.update({
            "pooled_r":   transf_ztor(mu),
            "ci_lb":      transf_ztor(lb) if not np.isnan(lb) else np.nan,
            "ci_ub":      transf_ztor(ub) if not np.isnan(ub) else np.nan,
            "k":          len(dat),
            "meta_model": meth,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in group_cols:
        if col in df.columns and hasattr(df[col], "cat"):
            out[col] = pd.Categorical(
                out[col], categories=df[col].cat.categories, ordered=True
            )
    return out


def compute_modelwise_pearson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model pooled Pearson r + one 'Pooled' row.
    Estimates are computed on Fisher-z scale and back-transformed via tanh().
    Returns columns: model, pooled_val (=r), ci_lb, ci_ub, k, meta_model.
    """
    out = _compute_meta_z(df, group_cols=["model"])
    out = out.rename(columns={"pooled_r": "pooled_val"})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Pearson preprocessing: age bins on the FULL datasets (includes DunedinPACE,
# DNAmTL for epi; DunedinPACNI for brain)
# ══════════════════════════════════════════════════════════════════════════════

# Epi full: reuse BIN7_LEVELS / _bin7 defined above
perf_epi["age_bin"] = pd.Categorical(
    perf_epi["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
)
perf_epi["model"] = perf_epi["model"].replace(RENAME)

# Brain Pearson bins — R uses "3–5y" lower bound (vs "4–5y" for MAE plots)
BRAIN_PEARSON_BIN5_LEVELS = [
    "Early childhood\n3–5y",
    "Middle childhood\n6–9y",
    "Late childhood\n10–14y",
    "Adolescence\n15–17y",
    "Young adulthood\n18+y",
]

def _bin_brain_pearson5(age: float) -> str:
    if age < 6:  return "Early childhood\n3–5y"
    if age < 10: return "Middle childhood\n6–9y"
    if age < 15: return "Late childhood\n10–14y"
    if age < 18: return "Adolescence\n15–17y"
    return "Young adulthood\n18+y"

perf_brain["age_bin"] = pd.Categorical(
    perf_brain["mean_age"].map(_bin_brain_pearson5),
    categories=BRAIN_PEARSON_BIN5_LEVELS, ordered=True
)

# Extended brain palette — includes DunedinPACNI (excluded from BRAIN_MODEL_PALETTE)
BRAIN_MODEL_PALETTE_FULL = {**BRAIN_MODEL_PALETTE, "DunedinPACNI": "#e25822"}

# Brain generation groups for Pearson modelwise ordering
BRAIN_GEN_BRAINAGE = ["Pyment", "DevBrainAge", "Centile2", "DBN",
                       "Kaufmann", "PyBrainAge", "ENIGMA"]
BRAIN_GEN_NEXTGEN  = ["DunedinPACNI"]

# ── Association preprocessing ─────────────────────────────────────────────────

mod2_eff2["assoc_var"] = mod2_eff2["RLM_SE_coeftest_HC3_scaled"] ** 2

# Age bins for association — reuse brain 5-bin scheme
mod2_eff2["age_bin"] = pd.Categorical(
    mod2_eff2["mean_age"].map(_bin_brain5),
    categories=BRAIN_BIN5_LEVELS, ordered=True
)

# Kelly (Polychrome) color sequence — 21 usable colors (omitting the white/cream)
_KELLY_COLORS = [
    "#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1",
    "#BE0032", "#C2B280", "#848482", "#008856", "#E68FAC",
    "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C",
    "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26",
]


# ══════════════════════════════════════════════════════════════════════════════
# Plot 13 — Epi unweighted MAE modelwise forest (2A)
# Re-estimated from data (R originally loads an .rds metafor object)
# ══════════════════════════════════════════════════════════════════════════════

epi_meta_mae = compute_modelwise_meta(
    perf_epi_sub, yi_col="log_MAE", vi_col="log_MAE_var", back_transform=True
)

gen1_gest_mae = _gen_order(GEN1_GEST, epi_meta_mae)
gen1_mae      = _gen_order(GEN1,      epi_meta_mae)
gen2_mae      = _gen_order(GEN2_4,    epi_meta_mae)

epi_mae_model_order = (
    ["Pooled"]
    + list(reversed(gen2_mae))
    + list(reversed(gen1_mae))
    + list(reversed(gen1_gest_mae))
)

div_overall_mae   = 1.5
div_gen2_gen1_mae = 1.5 + len(gen2_mae)
div_gen1_gest_mae = 1.5 + len(gen2_mae) + len(gen1_mae)

forest_plot(
    raw_df=perf_epi_sub,
    meta_df=epi_meta_mae,
    model_order=epi_mae_model_order,
    x_col="MAE",
    filename=os.path.join(plots_dir, "plot_perf_epi_unweightedMAE_modelwise_plus_overall_meta_raw_combined_cohortcolors.png"),
    title="Epigenetic Clock Performance (unweighted MAE)",
    x_label="MAE (years)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    dividers=[
        (div_overall_mae,   "solid"),
        (div_gen2_gen1_mae, "dashed"),
        (div_gen1_gest_mae, "dashed"),
    ],
    figsize=(11, 9),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3A — Brain unweighted MAE modelwise forest
# Mirrors 2A (Plot 13) but for brain models.  Brain MAE values span a much
# narrower range than epi clocks so we work on the raw (non-log) scale.
# Uses perf_brain_sub (excludes DunedinPACNI); generation grouping follows
# BRAIN_GEN_BRAINAGE defined above.
# ══════════════════════════════════════════════════════════════════════════════

brain_meta_mae_3a = compute_modelwise_meta(
    perf_brain_sub, yi_col="MAE", vi_col="MAE_var", back_transform=False
)

# Sort each generation's models ascending (worst→best) then reverse so worst
# sits just above Pooled and best is at the top — same convention as 2A.
brain_mae_3a_order_raw = _gen_order(BRAIN_GEN_BRAINAGE, brain_meta_mae_3a)
brain_mae_3a_model_order = ["Pooled"] + list(reversed(brain_mae_3a_order_raw))

div_overall_brain_mae = 1.5

forest_plot(
    raw_df=perf_brain_sub,
    meta_df=brain_meta_mae_3a,
    model_order=brain_mae_3a_model_order,
    x_col="MAE",
    filename=os.path.join(plots_dir, "plot_perf_brain_unweightedMAE_modelwise_plus_overall_meta_raw_combined_cohortcolors.png"),
    title="Brain Age Model Performance (unweighted MAE)",
    x_label="MAE (years)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    dividers=[(div_overall_brain_mae, "solid")],
    figsize=(11, 7),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 14 — Epi Pearson violin by age bin (2C)
# Uses full perf_epi (including DunedinPACE, DNAmTL); meta re-estimated on z-scale
# ══════════════════════════════════════════════════════════════════════════════

epi_pearson_meta_v = _compute_meta_z(perf_epi, group_cols=["age_bin", "model"])
# rename to match violin_epi_plot column convention
epi_pearson_meta_v = epi_pearson_meta_v.rename(columns={
    "pooled_r": "pooled_log_mae",
    "ci_lb":    "ci_lb_log",
    "ci_ub":    "ci_ub_log",
})

epi_pearson_counts = (
    perf_epi
    .assign(_ct=perf_epi["cohort"].astype(str) + "|" + perf_epi["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["_ct"]
    .nunique()
    .reset_index(name="k")
)
epi_pearson_counts["label"] = "k=" + epi_pearson_counts["k"].astype(str)

violin_epi_plot(
    df=perf_epi,
    meta=epi_pearson_meta_v,
    bin_levels=BIN7_LEVELS,
    counts=epi_pearson_counts,
    y_col="Pearson",
    y_label="Pearson r",
    filename=os.path.join(plots_dir, "plot_violin_perf_epi_age_pearson_brackets.png"),
    title="Epigenetic clock correlations across age groups",
    figsize=(13, 6.5),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 15 — Brain Pearson violin by age bin (3C)
# Uses full perf_brain (including DunedinPACNI)
# ══════════════════════════════════════════════════════════════════════════════

brain_pearson_meta_v = _compute_meta_z(perf_brain, group_cols=["age_bin", "model"])
brain_pearson_meta_v = brain_pearson_meta_v.rename(columns={
    "pooled_r": "pooled_log_mae",
    "ci_lb":    "ci_lb_log",
    "ci_ub":    "ci_ub_log",
})

brain_pearson_counts = (
    perf_brain
    .assign(_ct=perf_brain["cohort"].astype(str) + "|" + perf_brain["timepoint"].astype(str))
    .groupby("age_bin", observed=True)["_ct"]
    .nunique()
    .reset_index(name="k")
)
brain_pearson_counts["label"] = "k=" + brain_pearson_counts["k"].astype(str)

# temporarily swap MODEL_PALETTE to brain (including DunedinPACNI) for this plot
MODEL_PALETTE.clear()
MODEL_PALETTE.update(BRAIN_MODEL_PALETTE_FULL)

violin_epi_plot(
    df=perf_brain,
    meta=brain_pearson_meta_v,
    bin_levels=BRAIN_PEARSON_BIN5_LEVELS,
    counts=brain_pearson_counts,
    y_col="Pearson",
    y_label="Pearson r",
    filename=os.path.join(plots_dir, "plot_violin_perf_brain_age_pearson_brackets.png"),
    title="Brain age model correlations across age groups",
    figsize=(13, 6.5),
)

MODEL_PALETTE.clear()
MODEL_PALETTE.update(_orig_palette)  # restore epi palette


# ══════════════════════════════════════════════════════════════════════════════
# Plot 16 — Epi Pearson modelwise forest (2B)
# Uses full perf_epi; generation ordering without reversal (higher r = better)
# ══════════════════════════════════════════════════════════════════════════════

# GEN2_4 extended with DunedinPACE and DNAmTL (present in full perf_epi)
GEN2_4_PEARSON_EPI = ["PCGrimAge", "PhenoAge", "DamAge", "AdaptAge", "DunedinPACE", "DNAmTL"]

epi_pearson_modelwise = compute_modelwise_pearson(perf_epi)

def _gen_order_ascending(models_in_gen, meta_df):
    """Sort models within a generation by pooled_val ascending (worst → best).
    For Pearson r, this places worst at bottom and best at top without reversal."""
    present = set(meta_df["model"])
    sub = meta_df[meta_df["model"].isin(models_in_gen) & meta_df["model"].isin(present)]
    return sub.sort_values("pooled_val", ascending=True)["model"].tolist()

gen1_gest_p = _gen_order_ascending(GEN1_GEST,          epi_pearson_modelwise)
gen1_p      = _gen_order_ascending(GEN1,               epi_pearson_modelwise)
gen2_p      = _gen_order_ascending(GEN2_4_PEARSON_EPI, epi_pearson_modelwise)

epi_pearson_model_order = ["Pooled"] + gen2_p + gen1_p + gen1_gest_p

div_overall_epip   = 1.5
div_gen2_gen1_epip = 1.5 + len(gen2_p)
div_gen1_gest_epip = 1.5 + len(gen2_p) + len(gen1_p)

forest_plot(
    raw_df=perf_epi,
    meta_df=epi_pearson_modelwise,
    model_order=epi_pearson_model_order,
    x_col="Pearson",
    filename=os.path.join(plots_dir, "plot_perf_epi_modelwise_plus_overall_pearson_combined_cohortcolors.png"),
    title="Epigenetic clock performance (Pearson r)",
    x_label="Pearson correlation (epigenetic age vs chronological age)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    x_refline=0,
    dividers=[
        (div_overall_epip,   "solid"),
        (div_gen2_gen1_epip, "dashed"),
        (div_gen1_gest_epip, "dashed"),
    ],
    figsize=(11, 9),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 17 — Brain Pearson modelwise forest (3B)
# Uses full perf_brain (including DunedinPACNI); grouped by brain-age / next-gen
# ══════════════════════════════════════════════════════════════════════════════

brain_pearson_modelwise = compute_modelwise_pearson(perf_brain)

brain_brainage_p = _gen_order_ascending(BRAIN_GEN_BRAINAGE, brain_pearson_modelwise)
brain_nextgen_p  = _gen_order_ascending(BRAIN_GEN_NEXTGEN,  brain_pearson_modelwise)

brain_pearson_model_order = ["Pooled"] + brain_nextgen_p + brain_brainage_p

div_overall_bp  = 1.5
div_nextgen_bp  = 1.5 + len(brain_nextgen_p)

forest_plot(
    raw_df=perf_brain,
    meta_df=brain_pearson_modelwise,
    model_order=brain_pearson_model_order,
    x_col="Pearson",
    filename=os.path.join(plots_dir, "plot_perf_brain_modelwise_plus_overall_pearson_combined_cohortcolors.png"),
    title="Brain age model performance (Pearson r)",
    x_label="Pearson correlation (brain age vs chronological age)",
    color_col="cohort",
    color_palette=COHORT_PALETTE,
    meta_label="Meta-analysis",
    x_refline=0,
    dividers=[
        (div_overall_bp, "solid"),
        (div_nextgen_bp, "dashed"),
    ],
    figsize=(11, 7),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 18 — Association violin boxplot by brain age bin
# Violin (left) + meta points dodged (right) within each age bin
# Equivalent to ##### USED violin boxplot association beta by brain age bin
# ══════════════════════════════════════════════════════════════════════════════

def _compute_meta_assoc(
    df: pd.DataFrame,
    yi_col: str = "RLM_Estimate_scaled",
    vi_col: str = "assoc_var",
    group_cols: list = None,
) -> pd.DataFrame:
    """IV-weighted / MixedLM meta for association data."""
    if group_cols is None:
        group_cols = ["age_bin", "model_combi"]
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]
    rows  = []

    for keys, dat in valid.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mu, lb, ub, meth = _pool_group(dat, yi_col=yi_col, vi_col=vi_col)
        row = dict(zip(group_cols, keys))
        row.update({"pooled_beta": mu, "ci_lb": lb, "ci_ub": ub,
                    "k": len(dat), "meta_model": meth})
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in group_cols:
        if col in df.columns and hasattr(df[col], "cat"):
            out[col] = pd.Categorical(
                out[col], categories=df[col].cat.categories, ordered=True
            )
    return out


# create model_combi if not already a column in the CSV
if "model_combi" not in mod2_eff2.columns:
    mod2_eff2["model_combi"] = (
        mod2_eff2["brain_model"].astype(str) + "_" + mod2_eff2["epi_model"].astype(str)
    )

assoc_meta_violin = _compute_meta_assoc(
    mod2_eff2,
    group_cols=["age_bin", "model_combi"],
)

# Build model_combi color palette from Kelly colors
_all_combis = sorted(assoc_meta_violin["model_combi"].unique(), key=str.lower)
ASSOC_COMBI_PALETTE = {
    c: _KELLY_COLORS[i % len(_KELLY_COLORS)]
    for i, c in enumerate(_all_combis)
}

def assoc_violin_plot(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    bin_levels: list,
    y_col: str = "RLM_Estimate_scaled",
    group_col: str = "model_combi",
    filename: Optional[str] = None,
    title: str = "Association: brain age × epigenetic age",
    y_label: str = "Robust standardised β",
    model_palette: Optional[dict] = None,
    dpi: int = 300,
    figsize: tuple = (13, 6.5),
) -> None:
    """
    Side-by-side layout: violin at bin_x − 0.18 (grey, all data),
    meta points dodged at bin_x + 0.18 (coloured by model_combi).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.axhline(0, color="#d9d9d9", linewidth=0.8, zorder=0)

    x_map   = {b: i for i, b in enumerate(bin_levels)}
    violin_x = [i - 0.18 for i in range(len(bin_levels))]
    bin_arrays = [
        df.loc[df["age_bin"] == b, y_col].dropna().values
        for b in bin_levels
    ]

    # violin
    vp = ax.violinplot(
        bin_arrays, positions=violin_x,
        widths=0.28, showmedians=False, showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#d9d9d9")
        body.set_alpha(1.0)
        body.set_edgecolor("none")

    # boxplot overlay
    bp = ax.boxplot(
        bin_arrays, positions=violin_x,
        widths=0.08, patch_artist=True,
        showfliers=False, zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for elem in ("whiskers", "caps", "medians"):
        for ln in bp[elem]:
            ln.set_color("#333333")
            ln.set_linewidth(0.8)

    # meta points (position_dodge around bin_x + 0.18)
    combis   = sorted(meta[group_col].unique(), key=str.lower)
    n_c      = len(combis)
    dodge_w  = 0.45
    offsets  = np.linspace(-dodge_w / 2, dodge_w / 2, n_c) if n_c > 1 else np.array([0.0])
    off_map  = dict(zip(combis, offsets))

    if model_palette is None:
        model_palette = {c: _KELLY_COLORS[i % len(_KELLY_COLORS)]
                         for i, c in enumerate(combis)}

    for _, row in meta.iterrows():
        xi = x_map.get(row["age_bin"])
        if xi is None:
            continue
        col  = model_palette.get(row[group_col], "#888888")
        xpos = xi + 0.18 + off_map.get(row[group_col], 0.0)
        ax.scatter(xpos, row["pooled_beta"], color=col, s=18, alpha=0.5, zorder=5)
        if not (np.isnan(row["ci_lb"]) or np.isnan(row["ci_ub"])):
            ax.vlines(xpos, row["ci_lb"], row["ci_ub"],
                      color=col, linewidth=0.2, alpha=0.3, zorder=4)

    # y limits
    flat  = [v for arr in bin_arrays for v in arr]
    y_min = min(flat + list(meta["ci_lb"].dropna()) + list(meta["pooled_beta"].dropna()))
    y_max = max(flat + list(meta["ci_ub"].dropna()) + list(meta["pooled_beta"].dropna()))
    rng   = y_max - y_min
    ax.set_ylim(y_min - 0.08 * rng, y_max + 0.15 * rng)

    ax.set_xlim(-0.6, len(bin_levels) - 0.4)
    ax.set_xticks(range(len(bin_levels)))
    ax.set_xticklabels(bin_levels, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Brain developmental age group", fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    # no legend (legend.position = "none" in R)
    sns.despine(ax=ax, left=True, bottom=False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {filename}")
    else:
        plt.show()
    plt.close()


assoc_violin_plot(
    df=mod2_eff2,
    meta=assoc_meta_violin,
    bin_levels=BRAIN_BIN5_LEVELS,
    y_col="RLM_Estimate_scaled",
    group_col="model_combi",
    model_palette=ASSOC_COMBI_PALETTE,
    filename=os.path.join(plots_dir, "plot_violin_assoc_brain_mean_age_brackets_modelcombi_meta_new.png"),
    title="Association: brain age × epigenetic age across developmental groups",
    figsize=(13, 6.5),
)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 19 — Association modelwise faceted plot
# 9 panels: one per brain model + "Epi clocks (pooled)".
# x = standardised beta, y = epi_model (ordered by pooled estimate).
# Meta: pairwise (brain_model × epi_model); marginal (brain or epi); grand.
# Equivalent to ##### USED association modelwise
# ══════════════════════════════════════════════════════════════════════════════

# ── pairwise, marginal and grand meta estimates ───────────────────────────────

# Pairwise: one row per (brain_model, epi_model)
pair_meta = _compute_meta_assoc(
    mod2_eff2,
    group_cols=["brain_model", "epi_model"],
)

# Brain-marginal: pool across epi_model for each brain_model
brain_margin = _compute_meta_assoc(
    mod2_eff2,
    group_cols=["brain_model"],
)
brain_margin["epi_model"] = "Pooled"

# Epi-marginal: pool across brain_model for each epi_model
epi_margin = _compute_meta_assoc(
    mod2_eff2,
    group_cols=["epi_model"],
)
epi_margin["brain_model"] = "Pooled"

# Grand pooled
_grand_data = mod2_eff2.dropna(subset=["RLM_Estimate_scaled", "assoc_var"])
_grand_data = _grand_data[_grand_data["assoc_var"] > 0]
_gmu, _glb, _gub, _gmeth = _pool_group(
    _grand_data, yi_col="RLM_Estimate_scaled", vi_col="assoc_var"
)
grand_row = pd.DataFrame([{
    "brain_model":  "Pooled",
    "epi_model":    "Pooled",
    "pooled_beta":  _gmu,
    "ci_lb":        _glb,
    "ci_ub":        _gub,
    "k":            len(_grand_data),
    "meta_model":   _gmeth,
}])

# Combined estimates used in facet panels
plot_est = pd.concat([pair_meta, brain_margin, epi_margin, grand_row],
                      ignore_index=True)

# Facet column: actual brain models go in their own panel; "Pooled" → "Epi clocks (pooled)"
plot_est["brain_model_facet"] = plot_est["brain_model"].apply(
    lambda x: "Epi clocks (pooled)" if x == "Pooled" else x
)

# Raw data for per-panel scatter
pair_raw = mod2_eff2[["cohort", "brain_model", "epi_model",
                        "RLM_Estimate_scaled"]].dropna().copy()
pair_raw["brain_model_facet"] = pair_raw["brain_model"]  # no "Pooled" rows in raw

# epi_model ordering: grand pooled estimate across all brain, ascending (worst → best)
epi_model_grand_order = (
    epi_margin
    .sort_values("pooled_beta", ascending=True)["epi_model"]
    .tolist()
)
epi_y_order = ["Pooled"] + epi_model_grand_order  # Pooled at bottom

# facet panels — brain models + pooled
brain_models_ordered = sorted(
    [m for m in plot_est["brain_model_facet"].unique() if m != "Epi clocks (pooled)"],
    key=str.lower,
)
facet_panels = brain_models_ordered + ["Epi clocks (pooled)"]
n_panels     = len(facet_panels)
n_cols       = 5
n_rows       = math.ceil(n_panels / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 3.5 * n_rows),
    squeeze=False,
    sharey=True,
)
fig.suptitle(
    "Association: brain age × epigenetic age (model-level)",
    fontsize=14, y=1.01,
)

y_positions = {m: i for i, m in enumerate(epi_y_order)}

for idx, panel in enumerate(facet_panels):
    row_i, col_i = divmod(idx, n_cols)
    ax = axes[row_i][col_i]

    # vertical reference
    ax.axvline(0, color="#d9d9d9", linewidth=0.8, zorder=0)

    # ── raw data scatter ──────────────────────────────────────────────────────
    raw_sub = pair_raw[pair_raw["brain_model_facet"] == panel]
    if not raw_sub.empty:
        for _, pt in raw_sub.iterrows():
            yi = y_positions.get(pt["epi_model"])
            if yi is None:
                continue
            col = COHORT_PALETTE.get(pt["cohort"], "#888888")
            ax.scatter(pt["RLM_Estimate_scaled"], yi,
                       color=col, s=8, alpha=0.35, zorder=3)

    # ── meta estimates ────────────────────────────────────────────────────────
    est_sub = plot_est[plot_est["brain_model_facet"] == panel]
    for _, est in est_sub.iterrows():
        yi = y_positions.get(est["epi_model"])
        if yi is None:
            continue
        is_pooled = (est["epi_model"] == "Pooled")
        color = "black"
        size  = 30 if is_pooled else 18

        ax.scatter(est["pooled_beta"], yi, color=color, s=size,
                   marker="D" if is_pooled else "o", zorder=6)
        if not (np.isnan(est["ci_lb"]) or np.isnan(est["ci_ub"])):
            ax.hlines(yi, est["ci_lb"], est["ci_ub"],
                      color=color, linewidth=0.8 if is_pooled else 0.4, zorder=5)

    # ── axes ─────────────────────────────────────────────────────────────────
    ax.set_yticks(range(len(epi_y_order)))
    ax.set_yticklabels(epi_y_order if col_i == 0 else [], fontsize=7)
    ax.set_title(panel, fontsize=9, pad=4)
    ax.set_xlabel("Std. β" if row_i == n_rows - 1 else "", fontsize=8)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.4)
    ax.yaxis.grid(False)
    sns.despine(ax=ax, left=True)

    # divider between "Pooled" and model rows (at y = 0.5)
    ax.axhline(0.5, color="#666666", linewidth=0.4, linestyle="solid", zorder=2)

# hide any unused axes
for idx in range(n_panels, n_rows * n_cols):
    row_i, col_i = divmod(idx, n_cols)
    axes[row_i][col_i].set_visible(False)

plt.tight_layout()
assoc_facet_path = os.path.join(plots_dir, "plot_assoc_pairwise_with_margins_fixed_5cols_wide.png")
plt.savefig(assoc_facet_path, dpi=300, bbox_inches="tight")
print(f"Saved → {assoc_facet_path}")
plt.close()