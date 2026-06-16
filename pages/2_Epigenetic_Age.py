## todo: # just take one N per cohort and timepoint
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from plot_helpers import (
    COHORT_PALETTE, EPI_MODEL_PALETTE, RENAME,
    BIN7_LEVELS, _bin7, GEN1, GEN1_GEST, GEN2_4,
    _add_fisher_z, _compute_modelwise_meta, _compute_meta_z, _compute_bin_meta,
    _gen_order, _k_counts, forest_plot, violin_plot,
)

st.set_page_config(page_title="Epigenetic Clock Performance", layout="wide")

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

metric = st.sidebar.selectbox(
    "Metric",
    ["MAE", "RMSE", "R2", "Pearson", "Spearman", "wMAE_test", "nRMSE"]
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
c2.metric("MAE", round(filtered["MAE"].mean(), 3))
c3.metric("wMAE_test", round(filtered["wMAE_test"].mean(), 3))
c4.metric("R2", round(filtered["R2"].mean(), 3))

st.dataframe(filtered, use_container_width=True)

fig_box = px.box(
    filtered,
    x="model",
    y=metric,
    color="model",
    points="all"
)

st.plotly_chart(fig_box, use_container_width=True)

def rma_mv_exact(df):
    df = df.dropna(subset=["wMAE_test", "wMAE_SE_boot"]).copy()
    yi = df["wMAE_test"].values
    vi = df["wMAE_SE_boot"].values ** 2
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
        "wMAE": [mu],
        "ci.lb": [mu - 1.96 * se],
        "ci.ub": [mu + 1.96 * se]
    })

def plot_model_performance(df, est_df):
    df = df.copy()
    est_df = est_df.copy()
    model_order = df.groupby("model")["wMAE_test"].mean().sort_values(ascending=True).index.tolist()
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
            marker=dict(size=8, symbol=cohort_symbol[cohort], color="black")
        ))
        fig.add_trace(go.Scatter(
            x=sub["wMAE_test"],
            y=sub["y_jitter"],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=8,
                symbol=cohort_symbol[cohort],
                color=sub["mean_age"],
                colorscale=colorscale,
                cmin=age_min,
                cmax=age_max,
                showscale=False
            )
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
            x=[row["wMAE"]],
            y=[y],
            mode="markers",
            marker=dict(size=11, color="black", symbol="diamond"),
            showlegend=False
        ))
    pooled_y = y_map.get("Pooled", 0)
    fig.add_shape(
        type="line",
        x0=df["wMAE_test"].min(),
        x1=df["wMAE_test"].max(),
        y0=pooled_y + 0.5,
        y1=pooled_y + 0.5,
        line=dict(color="grey", width=1)
    )
    fig.update_layout(
        height=850,
        title="Epigenetic Clock Performance",
        xaxis_title="Weighted MAE",
        yaxis=dict(
            tickmode="array",
            tickvals=list(y_map.values()),
            ticktext=list(y_map.keys())
        ),
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top"
        )
    )
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
        showlegend=False
    ))
    return fig

overall_est = rma_mv_exact(filtered)

fig_adv = plot_model_performance(filtered, overall_est)

st.plotly_chart(fig_adv, use_container_width=True)


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

    # 2A — wMAE modelwise meta
    wmae_meta = _compute_modelwise_meta(
        sub, yi_col="log_wMAE", vi_col="log_wMAE_var", back_transform=True
    )

    # 2B — Pearson meta (sign-flip DNAmTL so all clocks point same direction)
    epi_pearson = d.copy()
    _mask = epi_pearson["model"] == "DNAmTL"
    epi_pearson.loc[_mask, "Pearson"]   = -epi_pearson.loc[_mask, "Pearson"]
    epi_pearson.loc[_mask, "pearson_z"] = -epi_pearson.loc[_mask, "pearson_z"]
    pearson_meta = _compute_meta_z(epi_pearson, group_cols=["model"])

    # 2C — wMAE by age bin
    epi_b7 = sub.copy()
    epi_b7["age_bin"] = pd.Categorical(
        epi_b7["mean_age"].map(_bin7), categories=BIN7_LEVELS, ordered=True
    )
    bin_meta = _compute_bin_meta(epi_b7, yi_col="log_wMAE", vi_col="log_wMAE_var")
    counts   = _k_counts(epi_b7)

    return sub, wmae_meta, epi_pearson, pearson_meta, epi_b7, bin_meta, counts


with st.spinner("Computing meta-analyses…"):
    (epi_sub, epi_wmae_mw, epi_pearson_df, epi_pearson_mw,
     epi_b7, meta_epi_b7, epi_counts) = _compute_epi_pub_data(filtered)

tab_2a, tab_2b, tab_2c = st.tabs([
    "Fig 2A — Weighted MAE forest",
    "Fig 2B — Pearson r forest",
    "Fig 2C — Weighted MAE by age",
])

# ── Fig 2A ────────────────────────────────────────────────────────────────────
with tab_2a:
    _A_NEXTGEN = ["DamAge", "AdaptAge", "PCGrimAge", "PhenoAge"]
    _A_GEN1    = ["CorticalClock", "PCBrainAge", "Hannum", "ZhangEN", "ZhangBLUP",
                  "Wu", "AltumAge", "cAge", "skinHorvath", "PedBE", "Horvath2013"]
    _A_GEST    = ["EPIC", "Knight", "Bohlin"]
    order_2a   = ["Pooled"] + _A_NEXTGEN + _A_GEN1 + _A_GEST

    fig_2a = forest_plot(
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
        figsize=(11, 9),
        gen_labels=[
            {"label": "Next Gen", "models": _A_NEXTGEN, "color": "#08306b"},
            {"label": "1st Gen",  "models": _A_GEN1,    "color": "#2171b5"},
            {"label": "Gest",     "models": _A_GEST,    "color": "#6baed6"},
        ],
    )
    st.pyplot(fig_2a, use_container_width=True)
    plt.close(fig_2a)

# ── Fig 2B ────────────────────────────────────────────────────────────────────
with tab_2b:
    _P_NEXT_GEN = ["DNAmTL", "DunedinPACE", "AdaptAge", "DamAge", "PhenoAge", "PCGrimAge"]
    _P_GEN1     = ["PCBrainAge", "Wu", "Hannum", "PedBE", "Horvath2013",
                   "CorticalClock", "AltumAge", "ZhangBLUP", "ZhangEN", "cAge", "skinHorvath"]
    _P_GEST     = ["Knight", "EPIC", "Bohlin"]
    order_2b    = ["Pooled"] + _P_NEXT_GEN + _P_GEN1 + _P_GEST
    div_ep2     = 0.5 + len(_P_NEXT_GEN)
    div_ep3     = 0.5 + len(_P_NEXT_GEN) + len(_P_GEN1)

    fig_2b = forest_plot(
        raw_df=epi_pearson_df,
        meta_df=epi_pearson_mw,
        model_order=order_2b,
        x_col="Pearson",
        title="Epigenetic clock performance (Pearson r)",
        x_label="Pearson correlation (epigenetic age vs chronological age)",
        color_palette=COHORT_PALETTE,
        x_refline=0,
        dividers=[(0.5, "solid"), (div_ep2, "dashed"), (div_ep3, "dashed")],
        figsize=(11, 9),
        gen_labels=[
            {"label": "Next Gen", "models": _P_NEXT_GEN, "color": "#08306b"},
            {"label": "1st Gen",  "models": _P_GEN1,     "color": "#2171b5"},
            {"label": "Gest",     "models": _P_GEST,     "color": "#6baed6"},
        ],
    )
    st.pyplot(fig_2b, use_container_width=True)
    plt.close(fig_2b)

# ── Fig 2C ────────────────────────────────────────────────────────────────────
with tab_2c:
    fig_2c = violin_plot(
        df=epi_b7,
        meta=meta_epi_b7,
        bin_levels=BIN7_LEVELS,
        y_col="log_wMAE",
        y_label="log(weighted MAE)",
        model_palette=EPI_MODEL_PALETTE,
        counts=epi_counts,
        title="Epigenetic clock performance across age groups",
    )
    st.pyplot(fig_2c, use_container_width=True)
    plt.close(fig_2c)