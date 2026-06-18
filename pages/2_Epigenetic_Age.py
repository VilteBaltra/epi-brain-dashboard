## todo: # just take one N per cohort and timepoint
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from plot_helpers import (
    COHORT_PALETTE, EPI_MODEL_PALETTE, RENAME,
    BIN7_LEVELS, _bin7, GEN1, GEN1_GEST, GEN2_4,
    _add_fisher_z, _compute_modelwise_meta, _compute_meta_z, _compute_bin_meta,
    _compute_bin_meta_z, _gen_order, _k_counts, forest_plot_plotly, violin_plot_plotly,
)

st.set_page_config(page_title="Epigenetic Clock Performance", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/perf_epi.csv")

df = load_data()

st.title("Epigenetic Clock Performance")

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

filtered = df[
    (df["cohort"].isin(cohort_f)) &
    (df["timepoint"].isin(timepoint_f)) &
    (df["model"].isin(model_f))
].copy()

c1, c2, c3, c4 = st.columns(4)

#c1.metric("N", round(filtered["N"].sum(), 3)) # just take one N per cohort and timepoint
c1.metric(
    "N",
    filtered[['cohort', 'timepoint', 'N']]
    .drop_duplicates()
    .groupby('cohort')['N']
    .max()
    .sum()
)
_ph_mae   = c2.empty()
_ph_wmae  = c3.empty()
_ph_r2    = c4.empty()
# placeholders filled after meta-analysis (model-level pooled means)
_ph_mae.metric("MAE", "…")
_ph_wmae.metric("wMAE_test", "…")
_ph_r2.metric("R2", "…")

st.dataframe(filtered, use_container_width=True)

_col_title, _col_metric = st.columns([4, 1])
_col_title.subheader("Model Performance")
metric = _col_metric.selectbox(
    "Metric",
    ["MAE", "RMSE", "R2", "Pearson", "Spearman", "wMAE_test", "nRMSE"],
    key="epi_metric",
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

se_map = {
    "MAE":      "MAE_SE_boot",
    "RMSE":     "RMSE_SE_boot",
    "wMAE_test":"wMAE_SE_boot",
    "nRMSE":    "nRMSE_SE_boot",
    "Pearson":  "pearson_se",
    "Spearman": None,
    "R2":       None,
}

def rma_mv_exact(df, metric, se_col):
    df = df.dropna(subset=[metric, se_col]).copy()
    yi = df[metric].values
    vi = df[se_col].values ** 2
    wi = 1 / vi
    mu_fixed = np.sum(wi * yi) / np.sum(wi)
    Q = np.sum(wi * (yi - mu_fixed) ** 2)
    C = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0, (Q - (len(df) - 1)) / C)
    wi_star = 1 / (vi + tau2)
    mu = np.sum(wi_star * yi) / np.sum(wi_star)
    se = np.sqrt(1 / np.sum(wi_star))
    return pd.DataFrame({
        "model": ["Pooled"],
        "estimate": [mu],
        "ci.lb": [mu - 1.96 * se],
        "ci.ub": [mu + 1.96 * se]
    })

def plot_model_performance(df, est_df, metric):
    df = df.copy()
    est_df = est_df.copy()
    model_order = df.groupby("model")[metric].mean().sort_values(ascending=True).index.tolist()
    model_order = [m for m in model_order if m != "Pooled"][::-1] + ["Pooled"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    est_df["model"] = pd.Categorical(est_df["model"], categories=model_order, ordered=True)
    y_map = {m: i for i, m in enumerate(model_order)}
    np.random.seed(42)
    df["y_base"] = df["model"].map(y_map).astype(float)
    df["y_jitter"] = df["y_base"] + np.random.normal(0, 0.12, len(df))
    cohorts = sorted(df["cohort"].dropna().unique())
    symbols = ["circle","square","diamond","triangle-up","triangle-down","cross","x","star","hexagram","pentagon","hourglass","bowtie","triangle-left","triangle-right"]
    cohort_symbol = {c: symbols[i % len(symbols)] for i, c in enumerate(cohorts)}
    age_min = df["mean_age"].min()
    age_max = df["mean_age"].max()
    colorscale = [[0.0,"black"],[0.2,"#3b0f70"],[0.4,"#8c2981"],[0.6,"#de4968"],[0.8,"#fbbf45"],[1.0,"#fcffa4"]]
    fig = go.Figure()
    for cohort in cohorts:
        sub = df[df["cohort"] == cohort]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=str(cohort),
            legendgroup="cohort",
            showlegend=True,
            marker=dict(size=8, symbol=cohort_symbol[cohort], color="black")
        ))
        fig.add_trace(go.Scatter(
            x=sub[metric],
            y=sub["y_jitter"],
            mode="markers",
            legendgroup="cohort",
            showlegend=False,
            marker=dict(
                size=8,
                symbol=cohort_symbol[cohort],
                color=sub["mean_age"],
                colorscale=colorscale,
                cmin=age_min,
                cmax=age_max,
                showscale=False
            ),
            text=sub["cohort"]
        ))
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
            x=[row["estimate"]],
            y=[y],
            mode="markers",
            marker=dict(size=11, color="black", symbol="diamond"),
            showlegend=False
        ))
    pooled_y = y_map.get("Pooled", 0)
    fig.add_shape(
        type="line",
        x0=df[metric].min(),
        x1=df[metric].max(),
        y0=pooled_y + 0.5,
        y1=pooled_y + 0.5,
        line=dict(color="grey", width=1)
    )
    fig.update_layout(
        height=850,
        title="Epigenetic Clock Performance",
        xaxis_title=metric,
        yaxis=dict(
            tickmode="array",
            tickvals=list(y_map.values()),
            ticktext=list(y_map.keys())
        ),
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
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            color=[age_min, age_max],
            cmin=age_min,
            cmax=age_max,
            colorscale=colorscale,
            colorbar=dict(title="Mean Age", x=1.02, y=0.25, len=0.45)
        ),
        showlegend=False,
        hoverinfo="none"
    ))
    return fig

se_col = se_map.get(metric)

if se_col is None or se_col not in filtered.columns:
    overall_est = pd.DataFrame({
        "model": ["Pooled"],
        "estimate": [filtered[metric].mean()],
        "ci.lb": [np.nan],
        "ci.ub": [np.nan]
    })
else:
    overall_est = rma_mv_exact(filtered, metric, se_col)

# fig_adv = plot_model_performance(filtered, overall_est, metric)
# st.plotly_chart(fig_adv, use_container_width=True)


# =========================================================
# PUBLICATION FIGURES (Fig 2A · 2B · 2C)
# =========================================================

st.divider()
st.header("Publication Figures")
st.caption("Replicating paper figures — responds to sidebar filters above.")


@st.cache_data(show_spinner=False)
def _compute_epi_pub_data(raw_df: pd.DataFrame):
    """Run all meta-analyses for Figs 2A–2C. Cached after first load."""
    # Apply renames and add Fisher-z columns
    d = raw_df.copy()
    d["model"] = d["model"].replace(RENAME)
    d = _add_fisher_z(d)

    # Sub-dataset (exclude clocks not used in performance analyses)
    sub = d[~d["model"].isin(["DunedinPACE", "DNAmTL"])].copy()
    sub["log_wMAE"]     = np.log(sub["wMAE_test"])
    sub["log_wMAE_var"] = (sub["wMAE_SE_boot"] / sub["wMAE_test"]) ** 2
    sub["log_MAE"]      = np.log(sub["MAE"].clip(lower=1e-9))
    sub["log_MAE_var"]  = (sub["MAE_SE_boot"] / sub["MAE"].clip(lower=1e-9)) ** 2

    # 2A — wMAE modelwise meta
    wmae_meta = _compute_modelwise_meta(
        sub, yi_col="log_wMAE", vi_col="log_wMAE_var", back_transform=True
    )

    # MAE modelwise meta
    mae_meta = _compute_modelwise_meta(
        sub, yi_col="log_MAE", vi_col="log_MAE_var", back_transform=True
    )

    # 2B — Pearson meta (sign-flip DNAmTL so all clocks point same direction)
    epi_pearson = d.copy()
    _mask = epi_pearson["model"] == "DNAmTL"
    epi_pearson.loc[_mask, "Pearson"]   = -epi_pearson.loc[_mask, "Pearson"]
    epi_pearson.loc[_mask, "pearson_z"] = -epi_pearson.loc[_mask, "pearson_z"]
    pearson_meta = _compute_meta_z(epi_pearson, group_cols=["model"])
    # Add overall pooled Pearson row — exclude DunedinPACE/DNAmTL (not chronological-age clocks)
    _excluded = ["DunedinPACE", "DNAmTL"]
    _ep = epi_pearson[~epi_pearson["model"].isin(_excluded)].dropna(subset=["pearson_z", "pearson_var"])
    _ep = _ep[_ep["pearson_var"] > 0].copy()
    _ep["_grp"] = "Pooled"
    _pooled_pr = _compute_meta_z(_ep, group_cols=["_grp"]).rename(columns={"_grp": "model"})
    pearson_meta = pd.concat([_pooled_pr, pearson_meta], ignore_index=True)

    # R2 — derived from Pearson meta by squaring (preserves proper CIs)
    r2_meta = pearson_meta[["model", "pooled_val", "ci_lb", "ci_ub"]].copy()
    r2_meta["pooled_val"] = r2_meta["pooled_val"] ** 2
    r2_meta["ci_lb"]      = r2_meta["ci_lb"] ** 2
    r2_meta["ci_ub"]      = r2_meta["ci_ub"] ** 2
    # Re-order bounds (squaring flips bounds for negative r)
    _lo = r2_meta[["ci_lb", "ci_ub"]].min(axis=1)
    _hi = r2_meta[["ci_lb", "ci_ub"]].max(axis=1)
    r2_meta["ci_lb"], r2_meta["ci_ub"] = _lo, _hi

    # 2C — age-binned data
    epi_b7 = sub.copy()
    epi_b7["age_bin"] = pd.Categorical(
        epi_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
    )
    epi_b7["MAE_var"] = epi_b7["MAE_SE_boot"] ** 2
    bin_meta_wmae = _compute_bin_meta(epi_b7, yi_col="log_wMAE", vi_col="log_wMAE_var")
    bin_meta_mae  = _compute_bin_meta(epi_b7, yi_col="MAE",      vi_col="MAE_var")
    counts = _k_counts(epi_b7)

    # Pearson by age bin (uses full epi_pearson dataset with age_bin added)
    epi_pearson_b7 = epi_pearson.copy()
    epi_pearson_b7["age_bin"] = pd.Categorical(
        epi_pearson_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
    )
    bin_meta_pearson = _compute_bin_meta_z(epi_pearson_b7)

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

    return (sub, wmae_meta, mae_meta, r2_meta, epi_pearson, pearson_meta,
            epi_b7, epi_pearson_b7,
            bin_meta_wmae, bin_meta_mae, bin_meta_pearson, bin_meta_r2,
            counts)


with st.spinner("Computing meta-analyses…"):
    (epi_sub, epi_wmae_mw, epi_mae_mw, epi_r2_mw, epi_pearson_df, epi_pearson_mw,
     epi_b7, epi_pearson_b7,
     meta_epi_wmae, meta_epi_mae, meta_epi_pearson, meta_epi_r2,
     epi_counts) = _compute_epi_pub_data(filtered)

# Fill top metrics with model-level pooled estimates
_pooled_mae  = epi_mae_mw.loc[epi_mae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_wmae = epi_wmae_mw.loc[epi_wmae_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_pooled_r2   = epi_r2_mw.loc[epi_r2_mw["model"] == "Pooled", "pooled_val"].iloc[0]
_ph_mae.metric("MAE", round(float(_pooled_mae), 3))
_ph_wmae.metric("wMAE_test", round(float(_pooled_wmae), 3))
_ph_r2.metric("R2", round(float(_pooled_r2), 3))

_pub_col1, _pub_col2 = st.columns([3, 1])
_pub_col1.markdown("**Select metric for forest plots:**")
pub_metric = _pub_col2.selectbox(
    "Publication metric",
    ["wMAE_test", "MAE", "Pearson", "R2"],
    key="pub_metric",
    label_visibility="collapsed",
)

tab_2a, tab_2c = st.tabs([
    "Fig 2A/B — Forest plot",
    "Fig 2C — Weighted MAE by age",
])

# ── Fig 2A / 2B (metric-switchable forest plot) ───────────────────────────────
with tab_2a:
    _A_NEXTGEN = ["DamAge", "AdaptAge", "PCGrimAge", "PhenoAge"]
    _A_GEN1    = ["CorticalClock", "PCBrainAge", "Hannum", "ZhangEN", "ZhangBLUP",
                  "Wu", "AltumAge", "cAge", "skinHorvath", "PedBE", "Horvath2013"]
    _A_GEST    = ["EPIC", "Knight", "Bohlin"]

    _P_NEXT_GEN = ["DNAmTL", "DunedinPACE", "AdaptAge", "DamAge", "PhenoAge", "PCGrimAge"]
    _P_GEN1     = ["PCBrainAge", "Wu", "Hannum", "PedBE", "Horvath2013",
                   "CorticalClock", "AltumAge", "ZhangBLUP", "ZhangEN", "cAge", "skinHorvath"]
    _P_GEST     = ["Knight", "EPIC", "Bohlin"]

    if pub_metric == "wMAE_test":
        order_2a = ["Pooled"] + _A_NEXTGEN + _A_GEN1 + _A_GEST
        fig_2ab = forest_plot_plotly(
            raw_df=epi_sub,
            meta_df=epi_wmae_mw,
            model_order=order_2a,
            x_col="wMAE_test",
            title="Epigenetic clock performance (weighted MAE)",
            x_label="Weighted MAE (MAE ÷ age range)",
            color_palette=COHORT_PALETTE,
            dividers=[
                (0.5,  "solid"),
                (4.5,  "dashed"),
                (15.5, "dashed"),
            ],
            x_refline=0,
            gen_labels=[
                {"label": "Next Gen", "models": _A_NEXTGEN, "color": "#08306b"},
                {"label": "1st Gen",  "models": _A_GEN1,    "color": "#2171b5"},
                {"label": "Gest",     "models": _A_GEST,    "color": "#6baed6"},
            ],
            font_size=14,
            marker_size=8,
            row_height=32,
        )
    elif pub_metric == "MAE":
        order_2a = ["Pooled"] + _A_NEXTGEN + _A_GEN1 + _A_GEST
        fig_2ab = forest_plot_plotly(
            raw_df=epi_sub,
            meta_df=epi_mae_mw,
            model_order=order_2a,
            x_col="MAE",
            title="Epigenetic clock performance (MAE)",
            x_label="Mean Absolute Error (years)",
            color_palette=COHORT_PALETTE,
            dividers=[
                (0.5,  "solid"),
                (4.5,  "dashed"),
                (15.5, "dashed"),
            ],
            x_refline=0,
            gen_labels=[
                {"label": "Next Gen", "models": _A_NEXTGEN, "color": "#08306b"},
                {"label": "1st Gen",  "models": _A_GEN1,    "color": "#2171b5"},
                {"label": "Gest",     "models": _A_GEST,    "color": "#6baed6"},
            ],
            font_size=14,
            marker_size=8,
            row_height=32,
        )
    elif pub_metric == "R2":
        order_2a = ["Pooled"] + _A_NEXTGEN + _A_GEN1 + _A_GEST
        fig_2ab = forest_plot_plotly(
            raw_df=epi_sub,
            meta_df=epi_r2_mw,
            model_order=order_2a,
            x_col="R2",
            title="Epigenetic clock performance (R²)",
            x_label="R²",
            color_palette=COHORT_PALETTE,
            dividers=[
                (0.5,  "solid"),
                (4.5,  "dashed"),
                (15.5, "dashed"),
            ],
            x_refline=0,
            gen_labels=[
                {"label": "Next Gen", "models": _A_NEXTGEN, "color": "#08306b"},
                {"label": "1st Gen",  "models": _A_GEN1,    "color": "#2171b5"},
                {"label": "Gest",     "models": _A_GEST,    "color": "#6baed6"},
            ],
            font_size=14,
            marker_size=8,
            row_height=32,
        )
    else:
        order_2b = ["Pooled"] + _P_NEXT_GEN + _P_GEN1 + _P_GEST
        div_ep2  = 0.5 + len(_P_NEXT_GEN)
        div_ep3  = 0.5 + len(_P_NEXT_GEN) + len(_P_GEN1)
        fig_2ab = forest_plot_plotly(
            raw_df=epi_pearson_df,
            meta_df=epi_pearson_mw,
            model_order=order_2b,
            x_col="Pearson",
            title="Epigenetic clock performance (Pearson r)",
            x_label="Pearson correlation (epigenetic age vs chronological age)",
            color_palette=COHORT_PALETTE,
            x_refline=0,
            dividers=[(0.5, "solid"), (div_ep2, "dashed"), (div_ep3, "dashed")],
            gen_labels=[
                {"label": "Next Gen", "models": _P_NEXT_GEN, "color": "#08306b"},
                {"label": "1st Gen",  "models": _P_GEN1,     "color": "#2171b5"},
                {"label": "Gest",     "models": _P_GEST,     "color": "#6baed6"},
            ],
            font_size=14,
            marker_size=8,
            row_height=32,
        )
    st.plotly_chart(fig_2ab, use_container_width=True)

# ── Fig 2C ────────────────────────────────────────────────────────────────────
with tab_2c:
    _c_cfg = {
        "wMAE_test": ("log_wMAE",  "log(weighted MAE)", epi_b7,          meta_epi_wmae),
        "MAE":       ("MAE",       "MAE (years)",        epi_b7,          meta_epi_mae),
        "Pearson":   ("Pearson",   "Pearson r",          epi_pearson_b7,  meta_epi_pearson),
        "R2":        ("R2",        "R²",                 epi_pearson_b7,  meta_epi_r2),
    }
    _y_col, _y_label, _df_c, _meta_c = _c_cfg[pub_metric]
    fig_2c = violin_plot_plotly(
        df=_df_c,
        meta=_meta_c,
        bin_levels=BIN7_LEVELS,
        y_col=_y_col,
        y_label=_y_label,
        model_palette=EPI_MODEL_PALETTE,
        counts=epi_counts,
        title="Epigenetic clock performance across age groups",
        font_size=14,
        marker_size=8,
    )
    st.plotly_chart(fig_2c, use_container_width=True)