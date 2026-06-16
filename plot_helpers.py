"""
Shared meta-analysis and plotting utilities for the dashboard.
Provides forest_plot() and violin_plot() returning matplotlib Figure objects,
plus all palettes, age-bin helpers, and meta-analysis functions.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("ignore")

# ── Colour palettes ────────────────────────────────────────────────────────────
EPI_MODEL_PALETTE = {
    "PCBrainAge":  "#222222", "PCGrimAge":   "#f3c300", "CorticalClock": "#875692",
    "cAge":        "#8db600", "AltumAge":    "#a1caf1", "Horvath2013":   "#be0032",
    "DamAge":      "#c2b280", "AdaptAge":    "#848482", "Hannum":        "#008856",
    "PhenoAge":    "#e68fac", "skinHorvath": "#0067a5", "PedBE":         "#f99379",
    "Wu":          "#604e97", "ZhangBLUP":   "#f6a600", "ZhangEN":       "#b3446c",
    "Bohlin":      "#dcd300", "Knight":      "#654522", "EPIC":          "#2b3d26",
    "DunedinPACE": "#e25822",
}

BRAIN_MODEL_PALETTE = {
    "Pyment":      "#222222", "DevBrainAge": "#f3c300", "Centile2":   "#875692",
    "DBN":         "#8db600", "Kaufmann":    "#a1caf1", "PyBrainAge": "#be0032",
    "ENIGMA":      "#c2b280",
}

COHORT_PALETTE = {
    "ALSPAC":           "#222222", "BHRC":             "#f3c300",
    "CannTeen":         "#875692", "DCHS":             "#f38400",
    "FinnBrain":        "#a1caf1", "GenR":             "#be0032",
    "GUSTO":            "#c2b280", "K2H Childhood":    "#848482",
    "K2H Infancy":      "#008856", "MTwiNS":           "#e68fac",
    "NICAP":            "#0067a5", "Oregon ADHD-1000": "#f99379",
    "FFCWS":            "#604e97", "TAG":              "#f6a600",
    "UCI Echo":         "#b3446c",
}

# Model name renames (applied to perf_epi only)
RENAME = {"Bohlin_conv": "Bohlin", "Knight_conv": "Knight", "EPIC_conv": "EPIC"}

# ── Age bins ───────────────────────────────────────────────────────────────────
BIN7_LEVELS = [
    "birth\n<3m", "Infancy\n<3y", "Early childhood\n3–5y",
    "Middle childhood\n6–9y", "Late childhood\n10–14y",
    "Adolescence\n15–17y", "Young adulthood\n18+y",
]

def _bin7(age: float) -> str:
    if age < 0.25: return "birth\n<3m"
    if age < 3:    return "Infancy\n<3y"
    if age < 6:    return "Early childhood\n3–5y"
    if age < 10:   return "Middle childhood\n6–9y"
    if age < 15:   return "Late childhood\n10–14y"
    if age < 18:   return "Adolescence\n15–17y"
    return "Young adulthood\n18+y"

BRAIN_BIN5_LEVELS = [
    "Early childhood\n4–5y", "Middle childhood\n6–9y",
    "Late childhood\n10–14y", "Adolescence\n15–17y", "Young adulthood\n18+y",
]

def _bin_brain5(age: float) -> str:
    if age < 6:  return "Early childhood\n4–5y"
    if age < 10: return "Middle childhood\n6–9y"
    if age < 15: return "Late childhood\n10–14y"
    if age < 18: return "Adolescence\n15–17y"
    return "Young adulthood\n18+y"

# ── Model generation groupings ─────────────────────────────────────────────────
GEN1_GEST = ["Bohlin", "Knight", "EPIC"]
GEN1 = [
    "ZhangEN", "ZhangBLUP", "Wu", "skinHorvath", "PedBE", "PCBrainAge",
    "Horvath2013", "Hannum", "CorticalClock", "cAge", "AltumAge",
]
GEN2_4 = ["PCGrimAge", "PhenoAge", "DamAge", "AdaptAge"]

BRAIN_GEN_BRAINAGE = ["Pyment", "DevBrainAge", "Centile2", "DBN", "Kaufmann", "PyBrainAge", "ENIGMA"]
BRAIN_GEN_NEXTGEN  = ["DunedinPACNI"]

# ── Fisher-z helper ────────────────────────────────────────────────────────────
def _add_fisher_z(df: pd.DataFrame) -> pd.DataFrame:
    """Add pearson_z and pearson_var columns if not already present."""
    df = df.copy()
    if "pearson_z" not in df.columns:
        df["pearson_z"] = np.arctanh(np.clip(df["Pearson"], -0.9999, 0.9999))
    if "pearson_var" not in df.columns:
        if "N" in df.columns:
            df["pearson_var"] = 1.0 / (df["N"] - 3)
        else:
            raise KeyError(
                "pearson_var not found and N column not available. "
                "Please add pearson_var (= 1/(N-3)) to the CSV."
            )
    return df


# ── Meta-analysis helpers ──────────────────────────────────────────────────────

def _pool_group(dat: pd.DataFrame, yi_col: str, vi_col: str):
    """
    Four-step convergence cascade:
      1) MixedLM + L-BFGS-B
      2) MixedLM + Nelder-Mead
      3) MixedLM without explicit random-slope term
      4) IV-weighted fixed-effect mean
    Returns (mu, lower, upper, method_label).
    """
    yi = dat[yi_col].values
    vi = dat[vi_col].values

    if len(dat) == 1:
        return yi[0], np.nan, np.nan, "single"

    if dat["cohort"].nunique() >= 2:
        d = dat.copy()
        d["const"] = 1.0
        for method, label in [("lbfgs", "RE lbfgs"), ("nm", "RE nm")]:
            try:
                fit = MixedLM(
                    endog=d[yi_col], exog=d[["const"]],
                    groups=d["cohort"], exog_re=d[["const"]],
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
        try:
            fit = MixedLM(
                endog=d[yi_col], exog=d[["const"]], groups=d["cohort"],
            ).fit(reml=True, disp=False)
            mu = float(fit.fe_params["const"])
            se = float(fit.bse_fe["const"])
            if not (np.isnan(mu) or np.isnan(se) or se <= 0):
                return mu, mu - 1.96 * se, mu + 1.96 * se, "RE simple"
        except Exception:
            pass

    w  = 1.0 / vi
    mu = float((w * yi).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    return mu, mu - 1.96 * se, mu + 1.96 * se, "IV-weighted"


def _compute_bin_meta(df: pd.DataFrame, yi_col: str, vi_col: str) -> pd.DataFrame:
    """Per (age_bin, model) pooled estimate for violin plots."""
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]
    rows = []
    for (age_bin, model), dat in valid.groupby(["age_bin", "model"], observed=True):
        mu, lb, ub, meth = _pool_group(dat, yi_col=yi_col, vi_col=vi_col)
        rows.append({
            "age_bin": age_bin, "model": model,
            "pooled_val": mu, "ci_lb_val": lb, "ci_ub_val": ub,
            "k": len(dat), "meta_model": meth,
        })
    out = pd.DataFrame(rows)
    if not out.empty and hasattr(df["age_bin"], "cat"):
        out["age_bin"] = pd.Categorical(
            out["age_bin"], categories=df["age_bin"].cat.categories, ordered=True
        )
    return out


def _compute_modelwise_meta(
    df: pd.DataFrame, yi_col: str, vi_col: str, back_transform: bool = False
) -> pd.DataFrame:
    """Per-model pooled estimate + one 'Pooled' row for forest plots."""
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]

    def _row(name, dat):
        mu, lb, ub, meth = _pool_group(dat, yi_col=yi_col, vi_col=vi_col)
        if back_transform:
            mu = np.exp(mu)
            lb = np.exp(lb) if not np.isnan(lb) else np.nan
            ub = np.exp(ub) if not np.isnan(ub) else np.nan
        return {"model": name, "pooled_val": mu, "ci_lb": lb,
                "ci_ub": ub, "k": len(dat), "meta_model": meth}

    rows = [_row("Pooled", valid)]
    for model, dat in valid.groupby("model", observed=True):
        rows.append(_row(model, dat))
    return pd.DataFrame(rows)


def _compute_meta_z(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Meta-analysis on Fisher-z scale, back-transformed to Pearson r via tanh()."""
    valid = df.dropna(subset=["pearson_z", "pearson_var"])
    valid = valid[valid["pearson_var"] > 0]
    rows = []
    for keys, dat in valid.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mu, lb, ub, meth = _pool_group(dat, yi_col="pearson_z", vi_col="pearson_var")
        row = dict(zip(group_cols, keys))
        row.update({
            "pooled_val": np.tanh(mu),
            "ci_lb":      np.tanh(lb) if not np.isnan(lb) else np.nan,
            "ci_ub":      np.tanh(ub) if not np.isnan(ub) else np.nan,
            "k":          len(dat),
            "meta_model": meth,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ── Ordering helpers ───────────────────────────────────────────────────────────

def _gen_order(models_in_gen: list, meta_df: pd.DataFrame) -> list:
    """Ascending by pooled_val (reverse for worst-near-Pooled MAE display)."""
    present = set(meta_df["model"])
    sub = meta_df[meta_df["model"].isin(models_in_gen) & meta_df["model"].isin(present)]
    return sub.sort_values("pooled_val")["model"].tolist()

def _gen_order_ascending(models_in_gen: list, meta_df: pd.DataFrame) -> list:
    """Ascending by pooled_val (worst near Pooled, best at top for Pearson r)."""
    present = set(meta_df["model"])
    sub = meta_df[meta_df["model"].isin(models_in_gen) & meta_df["model"].isin(present)]
    return sub.sort_values("pooled_val", ascending=True)["model"].tolist()

def _k_counts(df: pd.DataFrame, bin_col: str = "age_bin") -> pd.DataFrame:
    """Count unique cohort-timepoint combinations per age bin."""
    counts = (
        df.assign(_ct=df["cohort"].astype(str) + "|" + df["timepoint"].astype(str))
        .groupby(bin_col, observed=True)["_ct"]
        .nunique()
        .reset_index(name="k")
    )
    counts["label"] = "k=" + counts["k"].astype(str)
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting functions — both return matplotlib Figure objects
# ═══════════════════════════════════════════════════════════════════════════════

def forest_plot(
    raw_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    model_order: list,
    x_col: str,
    title: str = "",
    x_label: str = "",
    color_col: str = "cohort",
    color_palette: Optional[dict] = None,
    meta_label: str = "Meta-analysis",
    x_refline: Optional[float] = None,
    dividers: Optional[list] = None,
    x_log: bool = False,
    x_limits: Optional[tuple] = None,
    x_breaks: Optional[list] = None,
    x_break_labels: Optional[list] = None,
    figsize: tuple = (11, 7),
    dpi: int = 150,
    jitter_height: float = 0.12,
    gen_labels: Optional[list] = None,
) -> plt.Figure:
    """
    Horizontal caterpillar / forest plot.
    Returns the matplotlib Figure (caller is responsible for plt.close).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_map = {m: i for i, m in enumerate(model_order)}
    rng   = np.random.default_rng(42)

    for _, row in raw_df.iterrows():
        yi = y_map.get(row["model"])
        if yi is None or pd.isna(row[x_col]):
            continue
        col = (color_palette or {}).get(row[color_col], "#aaaaaa")
        ax.scatter(
            row[x_col],
            yi + rng.uniform(-jitter_height, jitter_height),
            color=col, alpha=0.45, s=18, zorder=3,
        )

    for _, row in meta_df.iterrows():
        yi = y_map.get(row["model"])
        if yi is None:
            continue
        if not (np.isnan(row["ci_lb"]) or np.isnan(row["ci_ub"])):
            ax.hlines(yi, row["ci_lb"], row["ci_ub"], color="black", linewidth=1.1, zorder=5)
        ax.scatter(row["pooled_val"], yi, color="black", marker="D", s=45, zorder=6)

    if x_refline is not None:
        ax.axvline(x_refline, color="#d9d9d9", linewidth=0.8, zorder=1)
    if dividers:
        for y_pos, lstyle in dividers:
            ax.axhline(y_pos, linestyle=lstyle, linewidth=0.4, color="#666666", zorder=2)

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

    present = set(raw_df[color_col].dropna())
    handles = [
        mpatches.Patch(color=(color_palette or {}).get(c, "#aaaaaa"), label=c, alpha=0.7)
        for c in sorted((color_palette or {}).keys()) if c in present
    ]
    handles.append(
        plt.Line2D([0], [0], marker="D", color="black", markerfacecolor="black",
                   markersize=7, linestyle="-", label=meta_label)
    )
    ax.legend(handles=handles, title="Source", loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=8, title_fontsize=10, frameon=False)

    if gen_labels:
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        box_x = -0.24
        box_w =  0.055
        for info in gen_labels:
            pad   = info.get("pad", 0.45)
            present_m = [m for m in info["models"] if m in y_map]
            if not present_m:
                continue
            y_lo  = min(y_map[m] for m in present_m) - pad
            y_hi  = max(y_map[m] for m in present_m) + pad
            y_ctr = (y_lo + y_hi) / 2
            ax.add_patch(plt.Rectangle(
                (box_x - box_w, y_lo), box_w, y_hi - y_lo,
                transform=trans, clip_on=False,
                facecolor="white", edgecolor=info["color"], linewidth=2, zorder=10,
            ))
            ax.text(
                box_x - box_w / 2, y_ctr, info["label"],
                transform=trans, clip_on=False,
                ha="center", va="center", rotation=90,
                fontsize=9, color="black", fontweight="bold", zorder=11,
            )

    if gen_labels:
        plt.tight_layout(rect=[0.14, 0, 1, 1])
    else:
        plt.tight_layout()

    return fig


def violin_plot(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    bin_levels: list,
    y_col: str,
    y_label: str,
    model_palette: dict,
    counts: Optional[pd.DataFrame] = None,
    title: str = "",
    dpi: int = 150,
    figsize: tuple = (13, 6.5),
) -> plt.Figure:
    """
    Violin + boxplot of outcome per age bin with dodged meta points and CI bars.
    Returns the matplotlib Figure (caller is responsible for plt.close).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_map = {b: i for i, b in enumerate(bin_levels)}

    ax.axhline(0, color="#ebebeb", linewidth=0.5, zorder=0)

    bin_arrays = [df.loc[df["age_bin"] == b, y_col].dropna().values for b in bin_levels]

    vp = ax.violinplot(bin_arrays, positions=range(len(bin_levels)),
                       widths=0.7, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("#d9d9d9")
        body.set_alpha(1.0)
        body.set_edgecolor("none")

    bp = ax.boxplot(bin_arrays, positions=range(len(bin_levels)),
                    widths=0.12, patch_artist=True, showfliers=False, zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for elem in ("whiskers", "caps", "medians"):
        for ln in bp[elem]:
            ln.set_color("#333333")
            ln.set_linewidth(0.8)

    all_models = sorted(meta["model"].unique(), key=str.lower)
    n_m     = len(all_models)
    offsets = np.linspace(-0.32, 0.32, n_m) if n_m > 1 else np.array([0.0])
    off_map = dict(zip(all_models, offsets))

    for _, row in meta.iterrows():
        xi = x_map.get(row["age_bin"])
        if xi is None:
            continue
        col  = model_palette.get(row["model"], "#888888")
        xpos = xi + off_map.get(row["model"], 0.0)
        ax.scatter(xpos, row["pooled_val"], color=col, s=18, alpha=0.65, zorder=5)
        if not (np.isnan(row["ci_lb_val"]) or np.isnan(row["ci_ub_val"])):
            ax.vlines(xpos, row["ci_lb_val"], row["ci_ub_val"],
                      color=col, linewidth=0.8, alpha=0.45, zorder=4)
            cw = 0.035
            for ycap in (row["ci_lb_val"], row["ci_ub_val"]):
                ax.hlines(ycap, xpos - cw, xpos + cw,
                          color=col, linewidth=0.8, alpha=0.45, zorder=4)

    flat   = [v for arr in bin_arrays for v in arr]
    y_min_d = min(flat) if flat else 0.0
    y_max_d = max(flat) if flat else 1.0
    y_min_m = meta["ci_lb_val"].min(skipna=True) if not meta.empty else y_min_d
    y_max_m = meta["ci_ub_val"].max(skipna=True) if not meta.empty else y_max_d
    y_min   = min(y_min_d, y_min_m)
    y_max   = max(y_max_d, y_max_m)
    y_rng   = y_max - y_min
    y_pad_b = 0.08 * y_rng
    y_pad_t = 0.15 * y_rng

    if counts is not None:
        cnt_map = dict(zip(counts["age_bin"], counts["label"]))
        for b, xi in x_map.items():
            lbl = cnt_map.get(b)
            if lbl:
                ax.text(xi, y_min - 0.35 * y_pad_b, lbl,
                        ha="center", va="top", fontsize=9)

    ax.set_xlim(-0.6, len(bin_levels) - 0.4)
    ax.set_ylim(y_min - y_pad_b, y_max + y_pad_t)
    ax.set_xticks(range(len(bin_levels)))
    ax.set_xticklabels(bin_levels, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Developmental age group", fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, left=True, bottom=False)

    present = set(meta["model"])
    handles = [
        mpatches.Patch(color=model_palette.get(m, "#888888"), label=m)
        for m in sorted(model_palette, key=str.lower) if m in present
    ]
    ax.legend(handles=handles, title="Model", loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=9, title_fontsize=11, frameon=False)

    plt.tight_layout()
    return fig
