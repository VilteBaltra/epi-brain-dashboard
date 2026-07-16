# need to add:
# pooled estimate per model
# meta-analysed overall estimate as per Marlene R script
# estimate shapes matching Marlene's R plot
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from plot_helpers import (
    COHORT_PALETTE, BRAIN_MODEL_PALETTE,
    BRAIN_BIN5_LEVELS, _bin_brain5, BRAIN_GEN_BRAINAGE, BRAIN_GEN_NEXTGEN,
    _add_fisher_z, _compute_modelwise_meta, _compute_meta_z, _compute_bin_meta,
    _compute_bin_meta_z, _gen_order_ascending, _k_counts, forest_plot_plotly, violin_plot_plotly,
    age_slope_plot_plotly,
)

st.set_page_config(page_title="Brain Age Model Performance", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    d = pd.read_csv("data/perf_brain.csv")
    d.drop(columns=["overlap_age"], errors="ignore", inplace=True)
    d["age_bin"] = pd.Categorical(
        d["mean_age"].map(_bin_brain5),
        categories=BRAIN_BIN5_LEVELS, ordered=True,
    )
    return d

@st.cache_data
def load_ct():
    return pd.read_csv("data/cohorts_timepoints.csv")

# Mapping from perf_brain.csv cohort names → cohorts_timepoints.csv cohort names
_COHORT_CT_MAP = {
    "K2H Childhood":    "K2H_childhood",
    "K2H Infancy":      "K2H_infancy",
    "NICAP":            "NICAP_T1",
    "Oregon ADHD-1000": "Oregon_ADHD-1000",
    "UCI Echo":         "UCI_ECHO",
}

df = load_data()
ct = load_ct()

st.title("Brain Age Model Performance")

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

cohort_f = st.sidebar.multiselect(
    "Cohort",
    df["cohort"].dropna().unique(),
    df["cohort"].dropna().unique()
)

_bin_display = [b.replace("\n", " ") for b in BRAIN_BIN5_LEVELS]
_display_to_bin = dict(zip(_bin_display, BRAIN_BIN5_LEVELS))
age_group_display_f = st.sidebar.multiselect(
    "Age group",
    _bin_display,
    _bin_display,
)
age_group_f = [_display_to_bin[d] for d in age_group_display_f]

model_f = st.sidebar.multiselect(
    "Model",
    df["model"].dropna().unique(),
    df["model"].dropna().unique()
)

# -------------------------
# FILTER DATA
# -------------------------
filtered = df[
    (df["cohort"].isin(cohort_f)) &
    (df["age_bin"].isin(age_group_f)) &
    (df["model"].isin(model_f))
].copy()

# -------------------------
# KPIs
# -------------------------
st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)

_ct_cohorts = [_COHORT_CT_MAP.get(c, c) for c in cohort_f]
_ct_brain = ct[(ct["cohort"].isin(_ct_cohorts)) & (ct["modality"] == "brain")]
_n_brain = int(_ct_brain["total n"].sum())
c1.metric(
    "Brain samples",
    f"{_n_brain:,}",
    help="Number of brain age samples across selected cohorts, including repeated measures across timepoints and arrays",
)
_ph_mae  = c2.empty()
_ph_wmae = c3.empty()
_ph_r2   = c4.empty()
_ph_mae.metric("MAE", "…",        help="Mean Absolute Error — average absolute difference between predicted and chronological age (years)")
_ph_wmae.metric("wMAE_test", "…", help="MAE weighted by the age range of the test sample (MAE ÷ age range); allows comparison across cohorts with different age spans")
_ph_r2.metric("R²", "…",          help="Coefficient of determination — proportion of variance in chronological age explained by predicted age")

st.dataframe(filtered, use_container_width=True)

with st.expander("Column glossary"):
    st.markdown("""
| Column | Description |
|---|---|
| `wMAE_test` | MAE weighted by the age range of the test sample (MAE ÷ age range) |
| `MAE` | Mean Absolute Error — average absolute difference between predicted and chronological age (years) |
| `RMSE` | Root Mean Square Error — similar to MAE but penalises larger errors more heavily |
| `nRMSE` | Normalised RMSE — RMSE divided by the age range of the sample |
| `R2` | Coefficient of determination — proportion of variance in chronological age explained by predicted age |
| `Pearson` | Pearson correlation between predicted and chronological age |
| `Spearman` | Spearman rank correlation between predicted and chronological age |
| `MAE_SE_boot` | Bootstrapped standard error of MAE |
| `wMAE_SE_boot` | Bootstrapped standard error of wMAE |
| `RMSE_SE_boot` | Bootstrapped standard error of RMSE |
| `nRMSE_SE_boot` | Bootstrapped standard error of nRMSE |
| `pearson_se` | Standard error of the Pearson correlation |
""")

# =========================================================
# 📊 SIMPLE MODEL PLOT
# =========================================================
_col_title, _col_metric = st.columns([4, 1])
_col_title.subheader("Model Performance")
metric = _col_metric.selectbox(
    "Metric",
    ["MAE", "RMSE", "R2", "Pearson", "Spearman", "wMAE_test", "nRMSE"],
    key="brain_metric",
    label_visibility="collapsed",
)

fig_box = px.box(
    filtered,
    x="model",
    y=metric,
    color="model",
    points="all"
)

st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# FULL MODEL PERFORMANCE PLOT
# =========================================================

def plot_model_performance(df, est_df, metric):

    df = df.copy()
    est_df = est_df.copy()

    # -------------------------
    # MODEL ORDER
    # -------------------------
    model_order = (
        df.groupby("model")[metric]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    model_order = [m for m in model_order if m != "Pooled"][::-1] + ["Pooled"]

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    est_df["model"] = pd.Categorical(est_df["model"], categories=model_order, ordered=True)

    y_map = {m: i for i, m in enumerate(model_order)}

    # -------------------------
    # JITTER (FIXED)
    # -------------------------
    np.random.seed(42)
    df["y_base"] = df["model"].map(y_map).astype(float)
    df["y_jitter"] = df["y_base"] + np.random.normal(0, 0.12, len(df))

    # -------------------------
    # COHORT SHAPES (14 cohorts)
    # -------------------------
    cohorts = sorted(df["cohort"].dropna().unique())

    symbols = [
        "circle", "square", "diamond", "triangle-up",
        "triangle-down", "cross", "x", "star",
        "hexagram", "pentagon", "hourglass", "bowtie",
        "triangle-left", "triangle-right"
    ]

    cohort_symbol = {c: symbols[i % len(symbols)] for i, c in enumerate(cohorts)}

    # -------------------------
    # MEAN AGE SCALE (NO NORMALISATION)
    # -------------------------
    age_min = df["mean_age"].min()
    age_max = df["mean_age"].max()

    colorscale = [[0, "black"], [1, "#FFD700"]]
    colorscale = [
                    [0.0, "black"],
                    [0.2, "#3b0f70"],
                    [0.4, "#8c2981"],
                    [0.6, "#de4968"],
                    [0.8, "#fbbf45"],
                    [1.0, "#fcffa4"]
                ]

    fig = go.Figure()

    # =========================================================
    # SCATTER (cohort = shape, mean_age = color)
    # =========================================================
    for cohort in cohorts:
        sub = df[df["cohort"] == cohort]

        # -------------------------
        # LEGEND TRACE (neutral color ONLY)
        # -------------------------
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=str(cohort),
            legendgroup="cohort",
            showlegend=True,
            marker=dict(
                size=8,
                symbol=cohort_symbol[cohort],
                color="black"   # <- forces neutral legend color
            )
        ))

        # -------------------------
        # DATA TRACE (real plot)
        # -------------------------
        fig.add_trace(go.Scatter(
            x=sub[metric],
            y=sub["y_jitter"],
            mode="markers",
            legendgroup="cohort",
            showlegend=False,
            marker=dict(
                size=8,
                symbol=cohort_symbol[cohort],

                # keep original color mapping
                color=sub["mean_age"],
                colorscale=colorscale,
                cmin=age_min,
                cmax=age_max,
                showscale=False
            ),
            text=sub["cohort"]
        ))

    # =========================================================
    # MODEL ESTIMATES
    # =========================================================
    for _, row in est_df.iterrows():
        y = y_map[row["model"]]

        fig.add_trace(go.Scatter(
            x=[row["ci.lb"], row["ci.ub"]],
            y=[y, y],
            mode="lines",
            line=dict(color="black", width=3),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            #x=[row["wMAE"]],
            x=[row["estimate"]],
            y=[y],
            mode="markers",
            marker=dict(size=11, color="black", symbol="diamond"),
            showlegend=False
        ))

    # divider
    pooled_y = y_map.get("Pooled", 0)

    fig.add_shape(
        type="line",
        x0=df[metric].min(),
        x1=df[metric].max(),
        y0=pooled_y + 0.5,
        y1=pooled_y + 0.5,
        line=dict(color="grey", width=1)
    )

    # =========================================================
    # LAYOUT (LEGENDS CLEAN)
    # =========================================================
    fig.update_layout(
        height=850,
        title="Brain Age Model Performance",
        xaxis_title=metric,


        yaxis=dict(
            tickmode="array",
            tickvals=list(y_map.values()),
            ticktext=list(y_map.keys())
        ),

        # cohort legend (upper right)
        legend=dict(
            title="Cohort",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.4)"
        )
    )

    fig.update_xaxes(autorange=True)

    # =========================================================
    # GLOBAL COLORBAR (lower right)
    # =========================================================
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            color=[age_min, age_max],
            cmin=age_min,
            cmax=age_max,
            colorscale=colorscale,
            colorbar=dict(
                title="Mean Age",
                x=1.02,
                y=0.25,
                len=0.45
            )
        ),
        showlegend=False,
        hoverinfo="none"
    ))

    return fig


import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import pandas as pd

se_map = {
    "MAE": "MAE_SE_boot",
    "RMSE": "RMSE_SE_boot",
    "wMAE_test": "wMAE_SE_boot",
    "nRMSE": "nRMSE_SE_boot",
    "RAE": "RAE_SE_boot",
    "Pearson": "pearson_se",
    "Spearman": None,  # maybe no SE available
    "R2": None         # usually not meta-analysed like this
}


def rma_mv_exact(df, metric):

    #df = df.dropna(subset=[metric, "wMAE_SE_boot"]).copy()
    df = df.dropna(subset=[metric, se_col]).copy()

    yi = df[metric].values
    #vi = df["wMAE_SE_boot"].values ** 2
    #vi = df[se_col] ** 2
    se = df[se_col].values
    vi = se ** 2


    wi = 1 / vi

    # fixed effect mean
    mu_fixed = np.sum(wi * yi) / np.sum(wi)

    # heterogeneity (Q statistic)
    Q = np.sum(wi * (yi - mu_fixed) ** 2)
    C = np.sum(wi) - np.sum(wi**2) / np.sum(wi)

    # DerSimonian-Laird tau² (approx rma.mv behavior)
    tau2 = max(0, (Q - (len(df) - 1)) / C)

    # random-effects weights
    wi_star = 1 / (vi + tau2)

    mu = np.sum(wi_star * yi) / np.sum(wi_star)

    se = np.sqrt(1 / np.sum(wi_star))

    ci_lb = mu - 1.96 * se
    ci_ub = mu + 1.96 * se

    return pd.DataFrame({
        "model": ["Pooled"],
        "estimate": [mu],
        #"wMAE": [mu],
        "ci.lb": [ci_lb],
        "ci.ub": [ci_ub]
    })

se_col = se_map.get(metric)

if se_col is None or se_col not in filtered.columns:
    # fallback: simple pooled mean
    overall_est = pd.DataFrame({
        "model": ["Pooled"],
        "estimate": [filtered[metric].mean()],
        "ci.lb": [np.nan],
        "ci.ub": [np.nan]
    })
else:
    overall_est = rma_mv_exact(filtered, metric)


# =========================================================
# PUBLICATION FIGURES (Fig 3A · 3B · 3C)
# =========================================================

st.divider()
st.header("Publication Figures")
st.caption("Replicating paper figures — responds to sidebar filters on the left.")


@st.cache_data(show_spinner=False)
def _compute_brain_pub_data(raw_df: pd.DataFrame):
    """Run all meta-analyses for Figs 3A–3C. Cached after first load."""
    d = raw_df.copy()
    d = _add_fisher_z(d)

    # Sub-dataset (exclude DunedinPACNI)
    sub = d[~d["model"].isin(["DunedinPACNI"])].copy()
    sub["wMAE_var"] = sub["wMAE_SE_boot"] ** 2
    sub["MAE_var"]  = sub["MAE_SE_boot"] ** 2

    # 3A — wMAE modelwise meta (raw scale, no back-transform)
    wmae_meta = _compute_modelwise_meta(
        sub, yi_col="wMAE_test", vi_col="wMAE_var", back_transform=False
    )

    # MAE modelwise meta (raw scale)
    mae_meta = _compute_modelwise_meta(
        sub, yi_col="MAE", vi_col="MAE_var", back_transform=False
    )

    # R2 — derived from Pearson meta by squaring (preserves proper CIs)
    # pearson_meta is computed below and includes a Pooled row
    # We compute a per-model Pearson meta here first for R2 derivation
    _pm_for_r2 = _compute_meta_z(d, group_cols=["model"])
    _dp = d[~d["model"].isin(["DunedinPACNI"])].dropna(subset=["pearson_z", "pearson_var"])
    _dp = _dp[_dp["pearson_var"] > 0].copy()
    _dp["_grp"] = "Pooled"
    _pooled_r2_r = _compute_meta_z(_dp, group_cols=["_grp"]).rename(columns={"_grp": "model"})
    _pm_for_r2 = pd.concat([_pooled_r2_r, _pm_for_r2], ignore_index=True)
    r2_meta = _pm_for_r2[["model", "pooled_val", "ci_lb", "ci_ub"]].copy()
    r2_meta["pooled_val"] = r2_meta["pooled_val"] ** 2
    r2_meta["ci_lb"]      = r2_meta["ci_lb"] ** 2
    r2_meta["ci_ub"]      = r2_meta["ci_ub"] ** 2
    _lo = r2_meta[["ci_lb", "ci_ub"]].min(axis=1)
    _hi = r2_meta[["ci_lb", "ci_ub"]].max(axis=1)
    r2_meta["ci_lb"], r2_meta["ci_ub"] = _lo, _hi

    # 3B — Pearson meta on full dataset (per-model) + overall Pooled row
    pearson_meta = _compute_meta_z(d, group_cols=["model"])
    _d_pool = d[~d["model"].isin(["DunedinPACNI"])].dropna(subset=["pearson_z", "pearson_var"])
    _d_pool = _d_pool[_d_pool["pearson_var"] > 0].copy()
    _d_pool["_grp"] = "Pooled"
    _pooled_p = _compute_meta_z(_d_pool, group_cols=["_grp"]).rename(columns={"_grp": "model"})
    pearson_meta = pd.concat([_pooled_p, pearson_meta], ignore_index=True)

    # 3C — age-binned data
    brain_b5 = sub.copy()
    brain_b5["age_bin"] = pd.Categorical(
        brain_b5["mean_age"].map(_bin_brain5), categories=BRAIN_BIN5_LEVELS, ordered=True
    )
    brain_b5["MAE_var"] = brain_b5["MAE_SE_boot"] ** 2
    bin_meta_wmae = _compute_bin_meta(brain_b5, yi_col="wMAE_test", vi_col="wMAE_var")
    bin_meta_mae  = _compute_bin_meta(brain_b5, yi_col="MAE",       vi_col="MAE_var")
    counts = _k_counts(brain_b5)

    # Pearson by age bin (full dataset with age_bin added)
    brain_pearson_b5 = d.copy()
    brain_pearson_b5["age_bin"] = pd.Categorical(
        brain_pearson_b5["mean_age"].map(_bin_brain5), categories=BRAIN_BIN5_LEVELS, ordered=True
    )
    bin_meta_pearson = _compute_bin_meta_z(brain_pearson_b5)

    # R2 by age bin — derived from Pearson bin meta by squaring
    bin_meta_r2 = bin_meta_pearson[
        ["age_bin", "model", "pooled_val", "ci_lb_val", "ci_ub_val"]
    ].copy()
    bin_meta_r2["pooled_val"] = bin_meta_r2["pooled_val"] ** 2
    bin_meta_r2["ci_lb_val"]  = bin_meta_r2["ci_lb_val"]  ** 2
    bin_meta_r2["ci_ub_val"]  = bin_meta_r2["ci_ub_val"]  ** 2
    _lo = bin_meta_r2[["ci_lb_val", "ci_ub_val"]].min(axis=1)
    _hi = bin_meta_r2[["ci_lb_val", "ci_ub_val"]].max(axis=1)
    bin_meta_r2["ci_lb_val"], bin_meta_r2["ci_ub_val"] = _lo, _hi

    return (sub, wmae_meta, mae_meta, r2_meta, d, pearson_meta,
            brain_b5, brain_pearson_b5,
            bin_meta_wmae, bin_meta_mae, bin_meta_pearson, bin_meta_r2,
            counts)


@st.cache_data
def _load_r_brain_meta():
    """Load pre-computed R meta-analytic estimates (used when no filters applied)."""
    try:
        wmae = pd.concat([
            pd.read_csv("data/model_brain_wMAE_mv_global.csv").rename(columns={"wMAE": "pooled_val", "ci.lb": "ci_lb", "ci.ub": "ci_ub"}),
            pd.read_csv("data/model_brain_wMAE_mv_modelwise.csv").rename(columns={"wMAE": "pooled_val", "ci.lb": "ci_lb", "ci.ub": "ci_ub"}),
        ], ignore_index=True)
        pearson = pd.concat([
            pd.read_csv("data/model_brain_pearson_mv_global.csv").rename(columns={"pearson_r": "pooled_val", "ci.lb": "ci_lb", "ci.ub": "ci_ub"}),
            pd.read_csv("data/model_brain_pearson_mv_modelwise.csv").rename(columns={"pearson_r": "pooled_val", "ci.lb": "ci_lb", "ci.ub": "ci_ub"}),
        ], ignore_index=True)
        return wmae, pearson
    except FileNotFoundError:
        return None, None


with st.spinner("Computing meta-analyses…"):
    (brain_sub, brain_wmae_mw, brain_mae_mw, brain_r2_mw, brain_full, brain_pearson_mw,
     brain_b5, brain_pearson_b5,
     meta_brain_wmae, meta_brain_mae, meta_brain_pearson, meta_brain_r2,
     brain_counts) = _compute_brain_pub_data(filtered)

# Hybrid: use R estimates (incl. Pooled) for forest plots when no filters applied
if len(filtered) == len(df):
    _r_wmae, _r_pearson = _load_r_brain_meta()
    if _r_wmae is not None:
        brain_wmae_mw    = _r_wmae
        brain_pearson_mw = _r_pearson

# Fill top metrics with model-level pooled estimates
_pooled_mae  = brain_mae_mw.loc[brain_mae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_wmae = brain_wmae_mw.loc[brain_wmae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_r2   = brain_r2_mw.loc[brain_r2_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_ph_mae.metric("MAE", round(float(_pooled_mae), 3),        help="Mean Absolute Error — average absolute difference between predicted and chronological age (years)")
_ph_wmae.metric("wMAE_test", round(float(_pooled_wmae), 3), help="MAE weighted by the age range of the test sample (MAE ÷ age range); allows comparison across cohorts with different age spans")
_ph_r2.metric("R²", round(float(_pooled_r2), 3),           help="Coefficient of determination — proportion of variance in chronological age explained by predicted age")

_pub_col1, _pub_col2 = st.columns([3, 1])
_pub_col1.markdown("**Select metric for forest plots:**")
pub_metric = _pub_col2.selectbox(
    "Publication metric",
    ["wMAE_test", "MAE", "Pearson", "R2"],
    key="pub_metric",
    label_visibility="collapsed",
)

tab_3a, tab_3c, tab_3d = st.tabs([
    "Fig 3A/B — Forest plot",
    "Fig 3C — Weighted MAE by age",
    "Fig 3D — Performance stability over development",
])

# ── Fig 3A / 3B (metric-switchable forest plot) ───────────────────────────────
with tab_3a:
    _B_GEN1    = ["Pyment", "Centile2", "DevBrainAge", "Kaufmann", "DBN", "PyBrainAge", "ENIGMA"]
    _B_NEXTGEN = ["DunedinPACNI"]

    brain_mw_order_raw = (
        brain_wmae_mw[brain_wmae_mw["model"] != "Pooled"]
        .sort_values("pooled_val", ascending=False)["model"]
        .tolist()
    )
    order_3a = ["Pooled"] + brain_mw_order_raw
    order_3b = ["Pooled"] + _B_NEXTGEN + _B_GEN1[::-1]

    if pub_metric == "wMAE_test":
        fig_3ab = forest_plot_plotly(
            raw_df=brain_sub,
            meta_df=brain_wmae_mw,
            model_order=order_3a,
            x_col="wMAE_test",
            title="Brain age model performance (weighted MAE)",
            x_label="Weighted MAE (MAE ÷ age range)",
            color_palette=COHORT_PALETTE,
            meta_label="Overall pooled",
            x_refline=0,
            dividers=[(0.5, "solid")],
            gen_labels=[
                {"label": "1st Gen", "models": BRAIN_GEN_BRAINAGE, "color": "#2171b5"},
            ],
            font_size=14,
            marker_size=8,
            row_height=50,
        )
    elif pub_metric == "MAE":
        fig_3ab = forest_plot_plotly(
            raw_df=brain_sub,
            meta_df=brain_mae_mw,
            model_order=order_3a,
            x_col="MAE",
            title="Brain age model performance (MAE)",
            x_label="Mean Absolute Error (years)",
            color_palette=COHORT_PALETTE,
            meta_label="Overall pooled",
            x_refline=0,
            dividers=[(0.5, "solid")],
            gen_labels=[
                {"label": "1st Gen", "models": BRAIN_GEN_BRAINAGE, "color": "#2171b5"},
            ],
            font_size=14,
            marker_size=8,
            row_height=50,
        )
    elif pub_metric == "R2":
        fig_3ab = forest_plot_plotly(
            raw_df=brain_sub,
            meta_df=brain_r2_mw,
            model_order=order_3a,
            x_col="R2",
            title="Brain age model performance (R²)",
            x_label="R²",
            color_palette=COHORT_PALETTE,
            meta_label="Overall pooled",
            x_refline=0,
            dividers=[(0.5, "solid")],
            gen_labels=[
                {"label": "1st Gen", "models": BRAIN_GEN_BRAINAGE, "color": "#2171b5"},
            ],
            font_size=14,
            marker_size=8,
            row_height=50,
        )
    else:  # Pearson
        fig_3ab = forest_plot_plotly(
            raw_df=brain_full,
            meta_df=brain_pearson_mw,
            model_order=order_3b,
            x_col="Pearson",
            title="Brain age model performance (Pearson r)",
            x_label="Pearson correlation (brain age vs chronological age)",
            color_palette=COHORT_PALETTE,
            meta_label="Meta-analysis",
            x_refline=0,
            dividers=[(0.5, "solid"), (1.5, "dashed")],
            gen_labels=[
                {"label": "Next Gen", "models": _B_NEXTGEN, "color": "#08306b", "pad": 0.65},
                {"label": "1st Gen",  "models": _B_GEN1,    "color": "#2171b5"},
            ],
            font_size=14,
            marker_size=8,
            row_height=50,
        )
    st.caption("Each row shows individual cohort estimates (coloured dots) and the pooled meta-analytic estimate (diamond with CI). Models are grouped by generation (1st Gen / Next Gen).")
    if pub_metric in ("wMAE_test", "MAE"):
        _log = st.checkbox("Log scale", key="log_3a", value=False)
        if _log:
            # Remove the x=0 vertical refline (undefined in log space)
            fig_3ab.layout.shapes = [
                s for s in fig_3ab.layout.shapes
                if not (getattr(s, "x0", None) == 0 and getattr(s, "x1", None) == 0)
            ]
            # Apply log scale only to the data axis (xaxis2), not the gen-label left panel (xaxis)
            fig_3ab.update_layout(xaxis2=dict(type="log", autorange=True))
    st.plotly_chart(fig_3ab, use_container_width=True)
    if len(filtered) == len(df):
        st.caption("Pooled estimates from multilevel random-effects meta-analysis (rma.mv, REML; metafor R package).")
    else:
        st.caption("Pooled estimates from DerSimonian-Laird random-effects meta-analysis (approximate; applied to filtered subset).")

# ── Fig 3C ────────────────────────────────────────────────────────────────────
with tab_3c:
    st.caption("Pooled meta-analytic estimates of brain age model performance across developmental age groups (birth–24 years). Pooled estimates from DerSimonian-Laird random-effects meta-analysis (approximate).")
    _c_cfg = {
        "wMAE_test": ("wMAE_test", "Weighted MAE",  brain_b5,          meta_brain_wmae),
        "MAE":       ("MAE",       "MAE (years)",    brain_b5,          meta_brain_mae),
        "Pearson":   ("Pearson",   "Pearson r",      brain_pearson_b5,  meta_brain_pearson),
        "R2":        ("R2",        "R²",             brain_pearson_b5,  meta_brain_r2),
    }
    _y_col, _y_label, _df_c, _meta_c = _c_cfg[pub_metric]
    fig_3c = violin_plot_plotly(
        df=_df_c,
        meta=_meta_c,
        bin_levels=BRAIN_BIN5_LEVELS,
        y_col=_y_col,
        y_label=_y_label,
        model_palette=BRAIN_MODEL_PALETTE,
        counts=brain_counts,
        title="Brain age model performance across age groups",
        font_size=14,
        marker_size=8,
    )
    st.plotly_chart(fig_3c, use_container_width=True)

# ── Fig 3D ────────────────────────────────────────────────────────────────────
with tab_3d:
    st.caption(
        "Model-specific age slopes from a meta-regression of wMAE on mean age. "
        "Values show change in wMAE per 1-year increase in mean cohort age. "
        "Negative = better performance with age; positive = poorer performance with age."
    )

    @st.cache_data(show_spinner=False)
    def _load_brain_age_slopes():
        _global = pd.read_csv("data/model_brain_wMAE_mv_global_ageint.csv")
        _mw     = pd.read_csv("data/model_brain_wMAE_mv_modelwise_ageint.csv")

        # Extract age slopes
        _slopes = _mw[_mw["term"].str.contains("mean_age_c")].copy()
        _slopes["model"] = (
            _slopes["term"]
            .str.replace(r"^model", "", regex=True)
            .str.replace(r":mean_age_c$", "", regex=True)
            .str.replace(r"\.mean_age_c$", "", regex=True)
        )
        # Brain uses raw (log-scale) estimates — no back-transformation
        _slopes = _slopes.rename(columns={"ci.lb": "ci_lb", "ci.ub": "ci_ub"})

        _GEN1 = ["Pyment", "DevBrainAge", "Centile2", "DBN", "Kaufmann", "PyBrainAge", "ENIGMA"]
        _GEN2 = ["DunedinPACNI"]
        _slopes["generation"] = _slopes["model"].apply(
            lambda m: "Gen1" if m in _GEN1 else ("Gen2-4" if m in _GEN2 else "Other")
        )

        _g1  = _slopes[_slopes["generation"] == "Gen1" ].sort_values("estimate")["model"].tolist()
        _g2  = _slopes[_slopes["generation"] == "Gen2-4"].sort_values("estimate")["model"].tolist()
        _oth = _slopes[_slopes["generation"] == "Other" ].sort_values("estimate")["model"].tolist()
        _order = ["Pooled"] + list(reversed(_oth)) + list(reversed(_g2)) + list(reversed(_g1))

        _pr = _global[_global["term"] == "mean_age_c"].iloc[0]
        _pooled = pd.DataFrame([{
            "model": "Pooled", "generation": "Pooled",
            "estimate": _pr["estimate"],
            "ci_lb":    _pr["ci.lb"],
            "ci_ub":    _pr["ci.ub"],
        }])

        _present_order = [m for m in _order if m in _slopes["model"].tolist() + ["Pooled"]]
        _plot = (
            pd.concat([_slopes[["model", "estimate", "ci_lb", "ci_ub", "generation"]], _pooled])
            .set_index("model").loc[_present_order].reset_index()
        )
        return _plot, _GEN1, _GEN2

    if pub_metric != "wMAE_test":
        st.info("This figure is only available for the wMAE metric.")
    else:
        _brain_slope_df, _BRAIN_GEN1, _BRAIN_GEN2 = _load_brain_age_slopes()

        _brain_gen_labels = [
            {"label": "1st Gen",  "models": _BRAIN_GEN1, "color": "#2171b5"},
            {"label": "Next Gen", "models": _BRAIN_GEN2, "color": "#08306b"},
        ]

        fig_3d = age_slope_plot_plotly(
            plot_df=_brain_slope_df,
            x_label="Change in wMAE per 1-year increase in mean age",
            title="Brain age model performance stability over development",
            model_palette=BRAIN_MODEL_PALETTE,
            gen_labels=_brain_gen_labels,
            font_size=14,
            x_range=[-0.15, 0.2],
        )
        _col3d, _ = st.columns([2, 1])
        with _col3d:
            st.plotly_chart(fig_3d, use_container_width=True)
