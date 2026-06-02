## todo: # just take one N per cohort and timepoint
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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