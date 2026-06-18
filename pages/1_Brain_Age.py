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
    return pd.read_csv("data/perf_brain.csv")

df = load_data()

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

timepoint_f = st.sidebar.multiselect(
    "Timepoint",
    df["timepoint"].dropna().unique(),
    df["timepoint"].dropna().unique()
)

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
    (df["timepoint"].isin(timepoint_f)) &
    (df["model"].isin(model_f))
].copy()

# -------------------------
# KPIs
# -------------------------
st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)
#c1.metric("N", round(filtered["N"].sum(), 3))
#clean = filtered[['cohort', 'timepoint', 'N']].drop_duplicates() 
#c1.metric("N", clean['N'].sum())
c1.metric(
    "N",
    filtered[['cohort', 'timepoint', 'N']]
    .drop_duplicates()
    .groupby('cohort')['N']
    .max()
    .sum()
)
_ph_mae  = c2.empty()
_ph_wmae = c3.empty()
_ph_r2   = c4.empty()
_ph_mae.metric("MAE", "…")
_ph_wmae.metric("wMAE_test", "…")
_ph_r2.metric("R²", "…")

st.dataframe(filtered, use_container_width=True)

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
        
    # for cohort in cohorts:
    #     sub = df[df["cohort"] == cohort]

    #     fig.add_trace(go.Scatter(
    #         x=sub["wMAE_test"],
    #         y=sub["y_jitter"],
    #         mode="markers",
    #         name=str(cohort),
    #         legendgroup="cohort",
    #         showlegend=True,

    #         marker=dict(
    #             size=8,
    #             symbol=cohort_symbol[cohort],

    #             color=sub["mean_age"],
    #             colorscale=colorscale,
    #             cmin=age_min,
    #             cmax=age_max,

    #             showscale=False
    #         ),
    #         text=sub["cohort"]
    #     ))

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


# -------------------------
# OVERALL ESTIMATE
# -------------------------
# overall_est = pd.DataFrame({
#     "model": ["Pooled"],
#     "wMAE": [filtered["wMAE_test"].mean()],
#     "ci.lb": [filtered["wMAE_test"].mean() - filtered["wMAE_test"].std()],
#     "ci.ub": [filtered["wMAE_test"].mean() + filtered["wMAE_test"].std()],
# })

# def compute_pooled_estimate(df):
#     tmp = df.copy()

#     # keep only valid rows
#     tmp = tmp.dropna(subset=["wMAE_test", "wMAE_SE_boot"])
#     tmp = tmp[tmp["wMAE_SE_boot"] > 0]

#     if len(tmp) == 0:
#         return pd.DataFrame({
#             "model": ["Pooled"],
#             "wMAE": [np.nan],
#             "ci.lb": [np.nan],
#             "ci.ub": [np.nan]
#         })

#     # variance
#     tmp["vi"] = tmp["wMAE_SE_boot"] ** 2

#     # guard against extreme values
#     tmp = tmp[np.isfinite(tmp["vi"]) & (tmp["vi"] > 0)]

#     # inverse variance weights
#     tmp["wi"] = 1 / tmp["vi"]

#     # pooled estimate
#     wmae = np.sum(tmp["wi"] * tmp["wMAE_test"]) / np.sum(tmp["wi"])

#     # standard error (CRITICAL FIX)
#     se = np.sqrt(1 / np.sum(tmp["wi"]))

#     # CI
#     ci_lb = wmae - 1.96 * se
#     ci_ub = wmae + 1.96 * se

#     return pd.DataFrame({
#         "model": ["Pooled"],
#         "wMAE": [wmae],
#         "ci.lb": [ci_lb],
#         "ci.ub": [ci_ub]
#     })

#  overall_est = compute_pooled_estimate(filtered)

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import pandas as pd

# def rma_mv_python(df):

#     df = df.copy()

#     # -------------------------
#     # CLEAN DATA
#     # -------------------------
#     df = df.dropna(subset=["wMAE_test", "wMAE_SE_boot", "cohort", "timepoint", "model"])
#     df = df[df["wMAE_SE_boot"] > 0]

#     # -------------------------
#     # WEIGHTS (meta-analysis variance)
#     # -------------------------
#     df["vi"] = df["wMAE_SE_boot"] ** 2
#     df["weights"] = 1 / df["vi"]

#     # -------------------------
#     # GROUP STRUCTURE (random effects)
#     # -------------------------
#     df["cohort_tp"] = df["cohort"].astype(str) + "_" + df["timepoint"].astype(str)

#     # -------------------------
#     # DESIGN MATRICES
#     # -------------------------
#     endog = df["wMAE_test"]

#     exog = np.ones(len(df))  # intercept only

#     # random effects structure
#     groups = df["cohort_tp"]

#     re_group_model = df["model"]

#     # -------------------------
#     # MIXED MODEL (REML)
#     # -------------------------
#     md = MixedLM(
#         endog,
#         exog,
#         groups=groups,
#         exog_re=np.ones((len(df), 1))
#     )

#     mdf = md.fit(reml=True, weights=df["weights"])

#     # -------------------------
#     # POOLED ESTIMATE
#     # -------------------------
#     mu = mdf.params[0]
#     se = mdf.bse[0]

#     ci_lb = mu - 1.96 * se
#     ci_ub = mu + 1.96 * se

#     return pd.DataFrame({
#         "model": ["Pooled"],
#         "wMAE": [mu],
#         "ci.lb": [ci_lb],
#         "ci.ub": [ci_ub],
#         "se": [se]
#     })
#overall_est = rma_mv_python(filtered)

# def pooled_by_model(df, metric, se_col):
#     out = []

#     for m, sub in df.groupby("model"):
#         sub = sub.dropna(subset=[metric, se_col])

#         if len(sub) < 2:
#             continuea

#         yi = sub[metric].values
#         vi = sub[se_col].values ** 2
#         wi = 1 / vi

#         mu = np.sum(wi * yi) / np.sum(wi)
#         se = np.sqrt(1 / np.sum(wi))

#         out.append({
#             "model": m,
#             "estimate": mu,
#             "ci.lb": mu - 1.96 * se,
#             "ci.ub": mu + 1.96 * se
#         })

#     return pd.DataFrame(out)

# model_est = pooled_by_model(filtered, metric, se_col)

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
    
#overall_est = rma_mv_exact(filtered, metric)

#fig_adv = plot_model_performance(filtered, overall_est)
# fig_adv = plot_model_performance(filtered, overall_est, metric)
# st.plotly_chart(fig_adv, use_container_width=True)


# =========================================================
# PUBLICATION FIGURES (Fig 3A · 3B · 3C)
# =========================================================

st.divider()
st.header("Publication Figures")
st.caption("Replicating paper figures — responds to sidebar filters above.")


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


with st.spinner("Computing meta-analyses…"):
    (brain_sub, brain_wmae_mw, brain_mae_mw, brain_r2_mw, brain_full, brain_pearson_mw,
     brain_b5, brain_pearson_b5,
     meta_brain_wmae, meta_brain_mae, meta_brain_pearson, meta_brain_r2,
     brain_counts) = _compute_brain_pub_data(filtered)

# Fill top metrics with model-level pooled estimates
_pooled_mae  = brain_mae_mw.loc[brain_mae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_wmae = brain_wmae_mw.loc[brain_wmae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_r2   = brain_r2_mw.loc[brain_r2_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_ph_mae.metric("MAE", round(float(_pooled_mae), 3))
_ph_wmae.metric("wMAE_test", round(float(_pooled_wmae), 3))
_ph_r2.metric("R²", round(float(_pooled_r2), 3))

_pub_col1, _pub_col2 = st.columns([3, 1])
_pub_col1.markdown("**Select metric for forest plots:**")
pub_metric = _pub_col2.selectbox(
    "Publication metric",
    ["wMAE_test", "MAE", "Pearson", "R2"],
    key="pub_metric",
    label_visibility="collapsed",
)

tab_3a, tab_3c = st.tabs([
    "Fig 3A/B — Forest plot",
    "Fig 3C — Weighted MAE by age",
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
    st.plotly_chart(fig_3ab, use_container_width=True)

# ── Fig 3C ────────────────────────────────────────────────────────────────────
with tab_3c:
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