"""
Shared meta-analysis and plotting utilities for the dashboard.
Provides forest_plot(), violin_plot(), assoc_violin_plot(), and
assoc_forest_plot() returning matplotlib Figure objects, plus all palettes,
age-bin helpers, and meta-analysis functions.
"""

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go

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
    DerSimonian-Laird random-effects meta-analysis using known within-study variances.
    Weights each study by 1/(vi + tau²), where tau² is the DL method-of-moments
    estimate — the same variance-weighted approach used by R's metafor::rma().
    Always converges; no optimiser needed.
    Returns (mu, lower, upper, method_label).
    """
    yi = dat[yi_col].values.astype(float)
    vi = dat[vi_col].values.astype(float)
    k  = len(dat)

    if k == 1:
        return float(yi[0]), np.nan, np.nan, "single"

    # ── Fixed-effect (inverse-variance) weights ────────────────────────────
    w     = 1.0 / vi
    mu_fe = float((w * yi).sum() / w.sum())

    # ── Cochran's Q and DerSimonian-Laird tau² ─────────────────────────────
    Q    = float((w * (yi - mu_fe) ** 2).sum())
    c    = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / c)

    # ── Random-effects pooled estimate ─────────────────────────────────────
    w_re  = 1.0 / (vi + tau2)
    mu_re = float((w_re * yi).sum() / w_re.sum())
    se_re = float(np.sqrt(1.0 / w_re.sum()))

    method = "DL-RE" if tau2 > 0 else "IV-FE"
    return mu_re, mu_re - 1.96 * se_re, mu_re + 1.96 * se_re, method


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


def _compute_bin_meta_z(df: pd.DataFrame) -> pd.DataFrame:
    """Per (age_bin, model) pooled Pearson r via Fisher-z meta, back-transformed via tanh."""
    valid = df.dropna(subset=["pearson_z", "pearson_var"])
    valid = valid[valid["pearson_var"] > 0]
    rows = []
    for (age_bin, model), dat in valid.groupby(["age_bin", "model"], observed=True):
        mu, lb, ub, meth = _pool_group(dat, yi_col="pearson_z", vi_col="pearson_var")
        rows.append({
            "age_bin": age_bin, "model": model,
            "pooled_val":  np.tanh(mu),
            "ci_lb_val":   np.tanh(lb) if not np.isnan(lb) else np.nan,
            "ci_ub_val":   np.tanh(ub) if not np.isnan(ub) else np.nan,
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
    """Count unique cohort-timepoint-array combinations per age bin."""
    arr = df["cpg_array"].astype(str) if "cpg_array" in df.columns else ""
    counts = (
        df.assign(_ct=df["cohort"].astype(str) + "|" + df["timepoint"].astype(str) + "|" + arr)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive Plotly versions of forest_plot and violin_plot
# ═══════════════════════════════════════════════════════════════════════════════

def forest_plot_plotly(
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
    jitter_height: float = 0.12,
    gen_labels: Optional[list] = None,
    font_size: int = 11,
    marker_size: int = 5,
    row_height: int = 45,
) -> go.Figure:
    """
    Interactive Plotly caterpillar / forest plot.
    Hover on raw dots → cohort name + value.
    Hover on meta diamonds → model + pooled estimate + 95 % CI.
    """
    from plotly.subplots import make_subplots

    rng  = np.random.default_rng(42)
    y_map = {m: i for i, m in enumerate(model_order)}
    pal  = color_palette or {}
    n    = len(model_order)
    _fig_h = max(420, 80 + row_height * n)

    # ── Figure: two-column subplot when gen_labels present ────────────────────
    # Left col (7 %): gen-label boxes only.  Right col (93 %): data.
    # With horizontal_spacing=0 the cols are adjacent; yaxis2 tick labels
    # (model names) appear between the two columns automatically.
    _use_sub = bool(gen_labels)
    if _use_sub:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.06, 0.94],
            shared_yaxes=False,
            horizontal_spacing=0.16,
        )
        _dc = dict(row=1, col=2)   # route data traces to right column
    else:
        fig = go.Figure()
        _dc = {}

    # ── Raw cohort dots (one trace per cohort so legend colours work) ─────────
    present_cohorts = sorted(raw_df[color_col].dropna().unique())
    for ci, cohort in enumerate(present_cohorts):
        grp = raw_df[raw_df[color_col] == cohort]
        xs, ys, texts = [], [], []
        for _, row in grp.iterrows():
            yi = y_map.get(row["model"])
            if yi is None or pd.isna(row[x_col]):
                continue
            xs.append(row[x_col])
            ys.append(yi + rng.uniform(-jitter_height, jitter_height))
            texts.append(f"<b>{cohort}</b><br>{x_label or x_col}: {row[x_col]:.3f}")
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=pal.get(cohort, "#aaaaaa"), size=marker_size, opacity=0.25),
            name=str(cohort),
            legendgroup="cohort",
            legendgrouptitle_text="Source" if ci == 0 else None,
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ), **_dc)

    # ── Meta CI lines + diamond markers ──────────────────────────────────────
    for _, row in meta_df.iterrows():
        yi = y_map.get(row["model"])
        if yi is None:
            continue
        pooled = row["pooled_val"]
        lb = row.get("ci_lb", np.nan)
        ub = row.get("ci_ub", np.nan)
        has_ci = not (pd.isna(lb) or pd.isna(ub))
        ci_str = f"{lb:.3f} – {ub:.3f}" if has_ci else "n/a"
        is_pooled = row["model"] == "Pooled"
        if has_ci:
            fig.add_trace(go.Scatter(
                x=[lb, ub], y=[yi, yi], mode="lines",
                line=dict(color="black", width=1.5),
                showlegend=False, hoverinfo="skip",
            ), **_dc)
        fig.add_trace(go.Scatter(
            x=[pooled], y=[yi], mode="markers",
            marker=dict(symbol="diamond", size=marker_size + 6 if is_pooled else marker_size + 4, color="black"),
            name=meta_label if is_pooled else "",
            showlegend=is_pooled,
            legendgroup="meta",
            legendgrouptitle_text="Pooled" if is_pooled else None,
            hovertemplate=(
                f"<b>{row['model']}</b><br>"
                f"Pooled {x_label or x_col}: {pooled:.3f}<br>"
                f"95% CI: [{ci_str}]<extra></extra>"
            ),
        ), **_dc)

    # ── Vertical reference line ───────────────────────────────────────────────
    if x_refline is not None:
        _xref = "x2" if _use_sub else "x"
        fig.add_shape(type="line",
                      x0=x_refline, x1=x_refline, xref=_xref,
                      y0=0, y1=1, yref=("y2 domain" if _use_sub else "y domain"),
                      line=dict(color="#d9d9d9", width=0.8))

    # ── Horizontal dividers (between row groups) ──────────────────────────────
    if dividers:
        _xref  = "x2 domain" if _use_sub else "x domain"
        _yref  = "y2"        if _use_sub else "y"
        for y_pos, lstyle in dividers:
            fig.add_shape(type="line",
                          x0=0, x1=1, xref=_xref,
                          y0=y_pos, y1=y_pos, yref=_yref,
                          line=dict(color="#666666", width=0.6,
                                    dash="solid" if lstyle == "solid" else "dash"))

    # ── Generation labels in left subplot ─────────────────────────────────────
    raw_boxes: list = []
    if _use_sub:
        _plot_h      = _fig_h - 60 - 50
        _PX_PER_UNIT = max(30, _plot_h / max(n + 0.5, 1))
        _PX_PER_CHAR = 7
        _BOX_GAP     = 0.05

        for info in gen_labels:
            present_m = [m for m in info["models"] if m in y_map]
            if not present_m:
                continue
            _y_bottom = min(y_map[m] for m in present_m)
            _y_top    = max(y_map[m] for m in present_m)
            _needed   = len(info["label"]) * _PX_PER_CHAR / _PX_PER_UNIT
            _half     = info.get("pad", 0.30)
            raw_boxes.append({
                "info": info, "present_m": present_m,
                "y_lo": _y_bottom - _half,
                "y_hi": _y_top    + _half,
                "height": (_y_top - _y_bottom) + 2 * _half,
            })

        raw_boxes.sort(key=lambda b: b["y_lo"])
        for i in range(len(raw_boxes) - 1):
            j = i + 1
            if raw_boxes[i]["y_hi"] > raw_boxes[j]["y_lo"] - _BOX_GAP:
                raw_boxes[i]["y_hi"] = raw_boxes[j]["y_lo"] - _BOX_GAP
                raw_boxes[i]["y_lo"] = raw_boxes[i]["y_hi"] - raw_boxes[i]["height"]

        for bx in raw_boxes:
            info = bx["info"]
            y_lo = bx["y_lo"]
            y_hi = bx["y_hi"]
            fig.add_trace(go.Scatter(
                x=[0.1, 0.1, 0.9, 0.9, 0.1],
                y=[y_lo, y_hi, y_hi, y_lo, y_lo],
                mode="lines", fill="toself", fillcolor="white",
                line=dict(color=info["color"], width=2),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            # Annotation xref="x" → col-1 x-axis, yref="y" → col-1 y-axis
            fig.add_annotation(
                xref="x", yref="y",
                x=0.5, y=(y_lo + y_hi) / 2,
                text=f"<b>{info['label']}</b>",
                showarrow=False, textangle=-90,
                font=dict(size=max(8, font_size - 2), color="black"),
                xanchor="center", yanchor="middle",
            )

    # ── Y-axis range (must encompass all boxes) ───────────────────────────────
    _y_lo = min((b["y_lo"] for b in raw_boxes), default=-0.5)
    _y_lo = min(_y_lo, -0.5)
    _y_hi = max((b["y_hi"] for b in raw_boxes), default=n - 0.5)
    _y_hi = max(_y_hi, n - 0.5)

    # ── Axis configuration ────────────────────────────────────────────────────
    _xd_vals = raw_df[x_col].dropna()
    _xd_min  = float(_xd_vals.min()) if len(_xd_vals) else 0.0
    _xd_max  = float(_xd_vals.max()) if len(_xd_vals) else 1.0
    _xd_span = max(_xd_max - _xd_min, 1e-6)

    _xdata_kw: dict = dict(
        title_text=x_label, gridcolor="#ebebeb",
        zeroline=False, showline=True, linecolor="#cccccc",
        title_font=dict(size=font_size), tickfont=dict(size=font_size),
    )
    if x_log:
        _xdata_kw["type"] = "log"
    if x_limits:
        _xdata_kw["range"] = list(x_limits)
    if x_breaks is not None:
        _xdata_kw["tickvals"] = x_breaks
        _xdata_kw["ticktext"] = x_break_labels or [str(b) for b in x_breaks]

    _ydata_kw: dict = dict(
        tickmode="array", tickvals=list(range(n)), ticktext=model_order,
        showgrid=False, showline=False, range=[_y_lo, _y_hi],
        tickfont=dict(size=font_size),
    )

    if _use_sub:
        # Col-1: gen-label panel — no axes visible
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         zeroline=False, showline=False,
                         range=[0, 1], row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False,
                         zeroline=False, showline=False,
                         range=[_y_lo, _y_hi], row=1, col=1)
        # Col-2: data panel
        fig.update_xaxes(**_xdata_kw, row=1, col=2)
        fig.update_yaxes(**_ydata_kw, row=1, col=2)
    else:
        fig.update_layout(xaxis=_xdata_kw, yaxis=_ydata_kw)
    _title_x = 0.5

    fig.update_layout(
        height=_fig_h,
        title=dict(text=title, font=dict(size=font_size + 4),
                   x=_title_x, xref="paper", xanchor="center"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=1.02, y=1, xanchor="left",
                    font=dict(size=max(8, font_size - 2)), tracegroupgap=6),
        margin=dict(l=10, r=180, t=60, b=50),
        hovermode="closest",
    )
    return fig


def violin_plot_plotly(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    bin_levels: list,
    y_col: str,
    y_label: str,
    model_palette: dict,
    counts: Optional[pd.DataFrame] = None,
    title: str = "",
    font_size: int = 11,
    marker_size: int = 6,
) -> go.Figure:
    """
    Interactive Plotly violin + dodged meta-dot plot.
    Hover on meta dots → model name + age bin + value + 95 % CI.
    Hover is disabled on the violin/box background.
    """
    x_map   = {b: i for i, b in enumerate(bin_levels)}
    n_bins  = len(bin_levels)
    fig     = go.Figure()

    all_models = sorted(meta["model"].unique(), key=str.lower)
    n_m        = len(all_models)
    offsets    = np.linspace(-0.32, 0.32, n_m) if n_m > 1 else np.array([0.0])
    off_map    = dict(zip(all_models, offsets))

    # ── Violin distributions (background, no hover) ───────────────────────────
    for b in bin_levels:
        arr = df.loc[df["age_bin"] == b, y_col].dropna().values
        if len(arr) == 0:
            continue
        fig.add_trace(go.Violin(
            x=[x_map[b]] * len(arr), y=arr.tolist(),
            name=str(b), showlegend=False,
            fillcolor="#d9d9d9", line_color="#bbbbbb",
            opacity=1.0,
            box=dict(
                visible=True,
                fillcolor="rgba(255,255,255,0.8)",
                line=dict(color="black", width=1.5),
            ),
            meanline_visible=False,
            hoverinfo="skip",
            points=False,
            width=0.65,
            spanmode="hard",
        ))

    # ── Meta dots per model with CI error bars ────────────────────────────────
    for mi, model in enumerate(all_models):
        col = model_palette.get(model, "#888888")
        sub = meta[meta["model"] == model]
        xs, ys, texts, ep, em = [], [], [], [], []
        for _, row in sub.iterrows():
            xi = x_map.get(row["age_bin"])
            if xi is None:
                continue
            xpos = xi + off_map.get(model, 0.0)
            pv   = row["pooled_val"]
            lb   = row.get("ci_lb_val", np.nan)
            ub   = row.get("ci_ub_val", np.nan)
            has_ci = not (pd.isna(lb) or pd.isna(ub))
            ci_str = f"{lb:.3f} – {ub:.3f}" if has_ci else "n/a"
            xs.append(xpos)
            ys.append(pv)
            texts.append(
                f"<b>{model}</b><br>{row['age_bin']}<br>"
                f"{y_label}: {pv:.3f}<br>95% CI: [{ci_str}]"
            )
            ep.append(ub - pv if has_ci else 0)
            em.append(pv - lb if has_ci else 0)
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=col, size=marker_size, opacity=0.75),
            name=model,
            legendgroup="model",
            legendgrouptitle_text="Model" if mi == 0 else None,
            error_y=dict(
                type="data", symmetric=False,
                array=ep, arrayminus=em,
                color=col, thickness=0.9, width=4,
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    # ── k-count annotations in data coords just below y=0 ────────────────────
    all_vals   = df[y_col].dropna().values
    y_data_min = float(np.nanmin(all_vals)) if len(all_vals) else 0.0
    y_data_max = float(np.nanmax(all_vals)) if len(all_vals) else 1.0
    y_span     = (y_data_max - y_data_min) if y_data_max > y_data_min else 1.0
    # k= sits just below the lowest violin shading, scaled by span
    k_y  = y_data_min - y_span * 0.06
    y_min = y_data_min - y_span * 0.15

    cohort_col = "cohort" if "cohort" in df.columns else None

    for b, xi in x_map.items():
        bin_df = df.loc[df["age_bin"] == b]
        if cohort_col:
            bin_df = bin_df.dropna(subset=[cohort_col])
        if bin_df.empty:
            continue
        # Compute k directly from data to avoid Categorical/StringDtype issues on Cloud
        if cohort_col and "timepoint" in bin_df.columns:
            _k = int(bin_df[["cohort", "timepoint"]].drop_duplicates().shape[0])
        elif cohort_col:
            _k = int(bin_df[cohort_col].nunique())
        else:
            _k = len(bin_df)
        lbl = f"k={_k}"
        if cohort_col:
            if "timepoint" in bin_df.columns:
                ct_counts = (
                    bin_df.groupby(cohort_col, observed=True)["timepoint"]
                    .nunique()
                    .sort_index()
                )
                cohort_str = "<br>".join(f"{c} (n={n})" for c, n in ct_counts.items())
            else:
                cohort_str = "<br>".join(sorted(bin_df[cohort_col].unique()))
            hover_text = f"<b>{lbl}</b><br><br>Cohorts:<br>{cohort_str}"
        else:
            hover_text = lbl
        fig.add_trace(go.Scatter(
            x=[xi], y=[k_y],
            mode="text",
            text=[lbl],
            textfont=dict(size=font_size, color="black"),
            textposition="middle center",
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=540,
        title=dict(text=title, font=dict(size=font_size + 4), x=0.5, xanchor="center"),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_bins)),
            ticktext=[b.replace("\n", "<br>") for b in bin_levels],
            tickangle=30, showgrid=False,
            title=dict(text="Developmental age group", font=dict(size=font_size + 1)),
            tickfont=dict(size=font_size),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=font_size + 1)),
            tickfont=dict(size=font_size),
            showgrid=False,
            zeroline=True, zerolinecolor="#cccccc", zerolinewidth=0.8,
            range=[y_min, None],
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        violinmode="overlay",
        legend=dict(x=1.02, y=1, xanchor="left",
                    font=dict(size=max(8, font_size - 2)), tracegroupgap=4),
        margin=dict(l=60, r=180, t=60, b=120),
        hovermode="closest",
    )
    return fig


def age_slope_plot_plotly(
    plot_df: pd.DataFrame,
    x_label: str,
    title: str = "",
    model_palette: dict = None,
    gen_labels: list = None,
    font_size: int = 14,
    row_height: int = 36,
    x_range: list = None,
) -> go.Figure:
    """
    Horizontal CI plot of model-specific age slopes.

    plot_df must have columns:
        model     – model name (str)
        estimate  – point estimate
        ci_lb     – lower 95 % CI bound
        ci_ub     – upper 95 % CI bound
        generation – e.g. "Gen1", "Gen2-4", "Pooled"

    Rows must be in display order (bottom → top on y axis).
    The "Pooled" row is drawn in black; all others use model_palette.

    gen_labels: list of dicts with keys "label", "models", "color"
    """
    from plotly.subplots import make_subplots

    pal = model_palette or {}
    n   = len(plot_df)
    y_map = {row["model"]: i for i, row in plot_df.iterrows()}

    _use_sub = bool(gen_labels)
    if _use_sub:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.08, 0.92],
            shared_yaxes=False,
            horizontal_spacing=0.22,
        )
        _dc = dict(row=1, col=2)
    else:
        fig = go.Figure()
        _dc = {}

    # ── Vertical reference line at x=0 ───────────────────────────────────────
    _xref = "x2" if _use_sub else "x"
    fig.add_shape(
        type="line",
        x0=0, x1=0, xref=_xref,
        y0=-0.5, y1=n - 0.5, yref=("y2" if _use_sub else "y"),
        line=dict(color="#888888", dash="dash", width=1),
    )

    # ── CI lines + diamond markers ────────────────────────────────────────────
    for _, row in plot_df.iterrows():
        yi       = y_map[row["model"]]
        is_pooled = row["model"] == "Pooled"
        color    = "black" if is_pooled else pal.get(row["model"], "#aaaaaa")
        size     = 12 if is_pooled else 10
        lw       = 2.0 if is_pooled else 1.5

        hover = (
            f"<b>{row['model']}</b><br>"
            f"Estimate: {row['estimate']:.3f}<br>"
            f"95% CI: [{row['ci_lb']:.3f}, {row['ci_ub']:.3f}]"
        )

        # CI line
        if not (pd.isna(row["ci_lb"]) or pd.isna(row["ci_ub"])):
            fig.add_trace(go.Scatter(
                x=[row["ci_lb"], row["ci_ub"]], y=[yi, yi],
                mode="lines",
                line=dict(color=color, width=lw),
                showlegend=False, hoverinfo="skip",
            ), **_dc)

        # Diamond point
        fig.add_trace(go.Scatter(
            x=[row["estimate"]], y=[yi],
            mode="markers",
            marker=dict(symbol="diamond", size=size, color=color),
            showlegend=False,
            hovertemplate=hover + "<extra></extra>",
        ), **_dc)

    # ── Horizontal dividers ───────────────────────────────────────────────────
    # Solid line separating Pooled from model rows
    _xdref = "x2 domain" if _use_sub else "x domain"
    _ydref = "y2"        if _use_sub else "y"
    fig.add_shape(
        type="line", x0=0, x1=1, xref=_xdref,
        y0=0.5, y1=0.5, yref=_ydref,
        line=dict(color="#666666", width=0.8, dash="solid"),
    )
    # Dashed lines between generation groups
    if gen_labels:
        _seen = 1  # skip Pooled row (index 0)
        for info in reversed(gen_labels):
            present = [m for m in info["models"] if m in y_map]
            if not present:
                continue
            _seen += len(present)
            if _seen < n:
                fig.add_shape(
                    type="line", x0=0, x1=1, xref=_xdref,
                    y0=_seen - 0.5, y1=_seen - 0.5, yref=_ydref,
                    line=dict(color="#666666", width=0.6, dash="dash"),
                )

    # ── Generation label boxes in left subplot ────────────────────────────────
    if _use_sub:
        _fig_h = max(350, 80 + row_height * n)
        _plot_h = _fig_h - 60 - 50
        _PX_PER_UNIT = max(30, _plot_h / max(n + 0.5, 1))

        raw_boxes = []
        for info in gen_labels:
            present = [m for m in info["models"] if m in y_map]
            if not present:
                continue
            y_lo = min(y_map[m] for m in present) - 0.35
            y_hi = max(y_map[m] for m in present) + 0.35
            raw_boxes.append({"info": info, "y_lo": y_lo, "y_hi": y_hi,
                               "height": y_hi - y_lo})

        raw_boxes.sort(key=lambda b: b["y_lo"])
        for i in range(len(raw_boxes) - 1):
            if raw_boxes[i]["y_hi"] > raw_boxes[i + 1]["y_lo"] - 0.05:
                raw_boxes[i]["y_hi"] = raw_boxes[i + 1]["y_lo"] - 0.05
                raw_boxes[i]["y_lo"] = raw_boxes[i]["y_hi"] - raw_boxes[i]["height"]

        for bx in raw_boxes:
            info = bx["info"]
            fig.add_trace(go.Scatter(
                x=[0.1, 0.1, 0.9, 0.9, 0.1],
                y=[bx["y_lo"], bx["y_hi"], bx["y_hi"], bx["y_lo"], bx["y_lo"]],
                mode="lines", fill="toself", fillcolor="white",
                line=dict(color=info["color"], width=2),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            fig.add_annotation(
                xref="x", yref="y",
                x=0.5, y=(bx["y_lo"] + bx["y_hi"]) / 2,
                text=f"<b>{info['label']}</b>",
                showarrow=False, textangle=-90,
                font=dict(size=max(8, font_size - 2), color="black"),
                xanchor="center", yanchor="middle",
            )

    # ── Axis config ───────────────────────────────────────────────────────────
    _fig_h = max(350, 80 + row_height * n)
    _ydata_kw = dict(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=plot_df["model"].tolist(),
        tickfont=dict(size=font_size),
        showgrid=False,
        range=[-0.7, n - 0.3],
    )
    _xdata_kw = dict(
        title_text=x_label,
        gridcolor="#ebebeb", zeroline=False,
        showline=True, linecolor="#cccccc",
        title_font=dict(size=font_size),
        tickfont=dict(size=font_size),
        **({"range": x_range} if x_range is not None else {}),
    )
    if _use_sub:
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         zeroline=False, showline=False,
                         range=[0, 1], row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False,
                         zeroline=False, showline=False,
                         range=[-0.7, n - 0.3], row=1, col=1)
        fig.update_xaxes(**_xdata_kw, row=1, col=2)
        fig.update_yaxes(**_ydata_kw, row=1, col=2)
    else:
        fig.update_layout(xaxis=_xdata_kw, yaxis=_ydata_kw)

    fig.update_layout(
        height=_fig_h,
        title=dict(text=title, font=dict(size=font_size + 2),
                   x=0.5, xref="paper", xanchor="center"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=40, t=60, b=50),
        hovermode="closest",
        showlegend=False,
    )
    return fig


# ── Kelly colours for model combinations (many levels) ────────────────────────
KELLY_COLORS = [
    "#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1",
    "#BE0032", "#C2B280", "#848482", "#008856", "#E68FAC",
    "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C",
    "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26",
]


def _compute_meta_assoc(
    df: pd.DataFrame,
    yi_col: str = "RLM_Estimate_scaled",
    vi_col: str = "assoc_var",
    group_cols: list = None,
) -> pd.DataFrame:
    """
    IV-weighted / MixedLM meta for association data grouped by group_cols.
    Uses the same 4-step convergence cascade as _pool_group.
    """
    if group_cols is None:
        group_cols = ["age_bin", "model_combi"]
    valid = df.dropna(subset=[yi_col, vi_col])
    valid = valid[valid[vi_col] > 0]
    rows = []
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


def assoc_violin_plot(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    bin_levels: list,
    counts: Optional[pd.DataFrame] = None,
    y_col: str = "RLM_Estimate_scaled",
    group_col: str = "model_combi",
    title: str = "Association: brain age × epigenetic age",
    y_label: str = "Robust standardised β",
    model_palette: Optional[dict] = None,
    dpi: int = 150,
    figsize: tuple = (13, 6.5),
) -> plt.Figure:
    """
    Violin + boxplot of association betas per age bin with dodged meta points.
    No legend (too many model combinations to show).
    Returns the matplotlib Figure (caller responsible for plt.close).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_map = {b: i for i, b in enumerate(bin_levels)}

    ax.axhline(0, color="#d9d9d9", linewidth=0.8, zorder=0)

    violin_x  = [i - 0.18 for i in range(len(bin_levels))]
    bin_arrays = [
        df.loc[df["age_bin"] == b, y_col].dropna().values
        for b in bin_levels
    ]

    # violin
    vp = ax.violinplot(bin_arrays, positions=violin_x,
                       widths=0.28, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("#d9d9d9")
        body.set_alpha(1.0)
        body.set_edgecolor("none")

    # boxplot overlay
    bp = ax.boxplot(bin_arrays, positions=violin_x,
                    widths=0.08, patch_artist=True, showfliers=False, zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    for elem in ("whiskers", "caps", "medians"):
        for ln in bp[elem]:
            ln.set_color("#333333")
            ln.set_linewidth(0.8)

    # meta points dodged to the right of each bin
    combis  = sorted(meta[group_col].dropna().unique(), key=str.lower)
    n_c     = len(combis)
    dodge_w = 0.45
    offsets = np.linspace(-dodge_w / 2, dodge_w / 2, n_c) if n_c > 1 else np.array([0.0])
    off_map = dict(zip(combis, offsets))

    if model_palette is None:
        model_palette = {c: KELLY_COLORS[i % len(KELLY_COLORS)]
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
    flat    = [v for arr in bin_arrays for v in arr]
    all_ys  = (flat
               + list(meta["ci_lb"].dropna())
               + list(meta["ci_ub"].dropna())
               + list(meta["pooled_beta"].dropna()))
    y_min   = min(all_ys) if all_ys else -0.5
    y_max   = max(all_ys) if all_ys else  0.5
    rng     = y_max - y_min

    if counts is not None:
        cnt_map = dict(zip(counts["age_bin"], counts["label"]))
        for b, xi in x_map.items():
            lbl = cnt_map.get(b)
            if lbl:
                ax.text(xi - 0.18, y_min - 0.035 * rng, lbl,
                        ha="center", va="top", fontsize=9)

    ax.set_xlim(-0.6, len(bin_levels) - 0.4)
    ax.set_ylim(y_min - 0.08 * rng, y_max + 0.15 * rng)
    ax.set_xticks(range(len(bin_levels)))
    ax.set_xticklabels(bin_levels, rotation=30, ha="right", fontsize=10)
    ax.set_xlabel("Developmental age group", fontsize=12, labelpad=8)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=16, pad=15)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, left=True, bottom=False)

    plt.tight_layout()
    return fig


def assoc_forest_plot(
    plot_est: pd.DataFrame,
    pair_raw: pd.DataFrame,
    epi_y_order: list,
    facet_panels: list,
    cohort_palette: dict,
    title: str = "Brain-PAR – Epi-PAR associations across model combinations",
    n_cols: int = 5,
    dpi: int = 150,
    figsize_per_panel: tuple = (4, 3.5),
) -> plt.Figure:
    """
    Multi-panel faceted forest plot (Fig 4A).
    One panel per brain model + one 'Epi clocks (pooled)' panel.
    Returns the matplotlib Figure (caller responsible for plt.close).
    """
    n_panels = len(facet_panels)
    n_rows   = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        sharey=True,
    )
    fig.suptitle(title, fontsize=14, y=1.01)

    y_positions = {m: i for i, m in enumerate(epi_y_order)}
    pooled_divider = y_positions.get("Pooled", 0) + 0.5
    rng = np.random.default_rng(42)

    for idx, panel in enumerate(facet_panels):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes[row_i][col_i]

        ax.axvline(0, color="#d9d9d9", linewidth=0.8, zorder=0)
        ax.axhline(pooled_divider, color="#666666", linewidth=0.4,
                   linestyle="solid", zorder=2)

        # raw cohort scatter
        raw_sub = pair_raw[pair_raw["brain_model_facet"] == panel]
        for _, pt in raw_sub.iterrows():
            yi = y_positions.get(pt["epi_model"])
            if yi is None:
                continue
            col = cohort_palette.get(pt["cohort"], "#888888")
            ax.scatter(pt["RLM_Estimate_scaled"],
                       yi + rng.uniform(-0.12, 0.12),
                       color=col, s=8, alpha=0.35, zorder=3)

        # meta estimates
        est_sub = plot_est[plot_est["brain_model_facet"] == panel]
        for _, est in est_sub.iterrows():
            yi = y_positions.get(est["epi_model"])
            if yi is None:
                continue
            is_pooled = (est["epi_model"] == "Pooled")
            ax.scatter(est["pooled_beta"], yi, color="black",
                       s=30 if is_pooled else 18,
                       marker="D" if is_pooled else "o", zorder=6)
            if not (np.isnan(est["ci_lb"]) or np.isnan(est["ci_ub"])):
                ax.hlines(yi, est["ci_lb"], est["ci_ub"],
                          color="black",
                          linewidth=0.8 if is_pooled else 0.4, zorder=5)

        ax.set_yticks(range(len(epi_y_order)))
        ax.set_yticklabels(epi_y_order if col_i == 0 else [], fontsize=7)
        ax.set_title(panel, fontsize=9, pad=4)
        ax.set_xlabel("Std. β" if row_i == n_rows - 1 else "", fontsize=8)
        ax.xaxis.grid(True, alpha=0.25, linewidth=0.4)
        ax.yaxis.grid(False)
        sns.despine(ax=ax, left=True)

    # hide unused panels
    for idx in range(n_panels, n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes[row_i][col_i].set_visible(False)

    # cohort legend outside the figure
    handles = [
        mpatches.Patch(color=cohort_palette.get(c, "#888888"), label=c, alpha=0.7)
        for c in sorted(cohort_palette)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="D", color="black", markerfacecolor="black",
                   markersize=7, linestyle="-", label="Overall pooled")
    )
    fig.legend(handles=handles, title="Source", loc="center right",
               bbox_to_anchor=(1.12, 0.5), fontsize=7,
               title_fontsize=9, frameon=False)

    plt.tight_layout()
    return fig
