import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui_helpers import render_sidebar_logo, render_footer
from plot_helpers import (
    COHORT_PALETTE, KELLY_COLORS,
    BRAIN_BIN5_LEVELS, _bin_brain5,
    _compute_meta_assoc, _pool_group,
    _k_counts,
)

st.set_page_config(page_title="Brain–Epi Associations", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stSidebar"] button {
    font-size: 12px !important;
    padding: 2px 8px !important;
    height: 26px !important;
    min-height: 0 !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/mod2_eff2.csv")
    df["assoc_var"] = df["RLM_SE_coeftest_HC3_scaled"] ** 2
    df["age_bin"] = pd.Categorical(
        df["mean_age"].map(_bin_brain5),
        categories=BRAIN_BIN5_LEVELS, ordered=True,
    )
    return df

@st.cache_data
def load_ct():
    return pd.read_csv("data/cohorts_timepoints.csv")

_COHORT_CT_MAP = {
    "K2H Childhood":    "K2H_childhood",
    "K2H Infancy":      "K2H_infancy",
    "NICAP":            "NICAP_T1",
    "Oregon ADHD-1000": "Oregon_ADHD-1000",
    "UCI Echo":         "UCI_ECHO",
}

df = load_data()
ct = load_ct()

st.title("Brain–Epigenetic Age Associations")

# ── Sidebar filters ────────────────────────────────────────────────────────────
render_sidebar_logo()
st.sidebar.header("Filters")

_all_cohorts      = list(df["cohort"].dropna().unique())
_bin_display      = [b.replace("\n", " ") for b in BRAIN_BIN5_LEVELS]
_display_to_bin   = dict(zip(_bin_display, BRAIN_BIN5_LEVELS))
_all_brain_models = sorted(df["brain_model"].dropna().unique())
_all_epi_models   = sorted(df["epi_model"].dropna().unique())

if "assoc_cohort" not in st.session_state:
    st.session_state["assoc_cohort"]      = _all_cohorts
if "assoc_age" not in st.session_state:
    st.session_state["assoc_age"]         = _bin_display
if "assoc_brain_model" not in st.session_state:
    st.session_state["assoc_brain_model"] = _all_brain_models
if "assoc_epi_model" not in st.session_state:
    st.session_state["assoc_epi_model"]   = _all_epi_models

if st.sidebar.button("↺ Reset filters"):
    st.session_state["assoc_cohort"]      = _all_cohorts
    st.session_state["assoc_age"]         = _bin_display
    st.session_state["assoc_brain_model"] = _all_brain_models
    st.session_state["assoc_epi_model"]   = _all_epi_models

st.sidebar.markdown("**Cohort**")
_c1, _c2 = st.sidebar.columns(2)
if _c1.button("Select all", key="assoc_cohort_all"):
    st.session_state["assoc_cohort"] = _all_cohorts
if _c2.button("Clear", key="assoc_cohort_clear"):
    st.session_state["assoc_cohort"] = []
cohort_f = st.sidebar.multiselect("Cohort", _all_cohorts, key="assoc_cohort", label_visibility="collapsed")

st.sidebar.markdown("**Age group**")
_a1, _a2 = st.sidebar.columns(2)
if _a1.button("Select all", key="assoc_age_all"):
    st.session_state["assoc_age"] = _bin_display
if _a2.button("Clear", key="assoc_age_clear"):
    st.session_state["assoc_age"] = []
age_group_display_f = st.sidebar.multiselect("Age group", _bin_display, key="assoc_age", label_visibility="collapsed")
age_group_f = [_display_to_bin[d] for d in age_group_display_f]

st.sidebar.markdown("**Brain model**")
_b1, _b2 = st.sidebar.columns(2)
if _b1.button("Select all", key="assoc_brain_all"):
    st.session_state["assoc_brain_model"] = _all_brain_models
if _b2.button("Clear", key="assoc_brain_clear"):
    st.session_state["assoc_brain_model"] = []
brain_model_f = st.sidebar.multiselect("Brain Model", _all_brain_models, key="assoc_brain_model", label_visibility="collapsed")

st.sidebar.markdown("**Epi model**")
_e1, _e2 = st.sidebar.columns(2)
if _e1.button("Select all", key="assoc_epi_all"):
    st.session_state["assoc_epi_model"] = _all_epi_models
if _e2.button("Clear", key="assoc_epi_clear"):
    st.session_state["assoc_epi_model"] = []
epi_model_f = st.sidebar.multiselect("Epi Model", _all_epi_models, key="assoc_epi_model", label_visibility="collapsed")

filtered = df[
    (df["cohort"].isin(cohort_f)) &
    (df["age_bin"].isin(age_group_f)) &
    (df["brain_model"].isin(brain_model_f)) &
    (df["epi_model"].isin(epi_model_f))
].copy()

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.subheader("Summary")
_ct_cohorts = [_COHORT_CT_MAP.get(c, c) for c in cohort_f]
_ct_overlap = ct[(ct["cohort"].isin(_ct_cohorts)) & (ct["modality"] == "overlap_brain_epi")]
_n_overlap = int(_ct_overlap["total n"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overlap samples", f"{_n_overlap:,}",
          help="Participants with both brain and epigenetic data across selected cohorts, including repeated measures across timepoints and arrays")
c2.metric("Effect sizes", f"{len(filtered):,}",
          help="Number of individual brain–epigenetic age association estimates (one per cohort × timepoint × brain model × epi model combination)")
c3.metric("Cohorts", filtered["cohort"].nunique(),
          help="Number of distinct cohorts contributing to the selected associations")
c4.metric("Brain models", filtered["brain_model"].nunique(),
          help="Number of distinct brain age models included in the selected associations")
c5.metric("Epi models", filtered["epi_model"].nunique(),
          help="Number of distinct epigenetic clocks included in the selected associations")

st.dataframe(
    filtered[["cohort", "timepoint", "brain_model", "epi_model",
              "RLM_Estimate_scaled", "RLM_SE_coeftest_HC3_scaled"]]
    .reset_index(drop=True),
    use_container_width=True,
)


# ── Meta-analysis (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _compute_assoc_pub_data(raw_df: pd.DataFrame):
    d = raw_df.copy()

    # Fixed model orderings matching paper (R code)
    _GEN1_ASSOC  = sorted(["AltumAge", "CorticalClock", "Hannum", "Horvath2013",
                            "PCBrainAge", "PedBE", "Wu", "ZhangBLUP", "ZhangEN",
                            "cAge", "skinHorvath"], key=str.lower)
    _GEN2P_ASSOC = sorted(["AdaptAge", "DamAge", "DNAmTL", "DunedinPACE",
                            "PCGrimAge", "PhenoAge"], key=str.lower)
    _BRAIN_ORDER = ["Centile2", "DBN", "DevBrainAge", "ENIGMA", "Kaufmann",
                    "PyBrainAge", "Pyment", "DunedinPACNI"]

    # ── Fig 4A ────────────────────────────────────────────────────────────────
    pair_meta    = _compute_meta_assoc(d, group_cols=["brain_model", "epi_model"])
    brain_margin = _compute_meta_assoc(d, group_cols=["brain_model"])
    brain_margin["epi_model"] = "Pooled"

    epi_margin   = _compute_meta_assoc(d, group_cols=["epi_model"])
    epi_margin["brain_model"] = "Pooled"

    _gdata = d.dropna(subset=["RLM_Estimate_scaled", "assoc_var"])
    _gdata = _gdata[_gdata["assoc_var"] > 0]
    _gmu, _glb, _gub, _gmeth = _pool_group(
        _gdata, yi_col="RLM_Estimate_scaled", vi_col="assoc_var"
    )
    grand_row = pd.DataFrame([{
        "brain_model": "Pooled", "epi_model": "Pooled",
        "pooled_beta": _gmu, "ci_lb": _glb, "ci_ub": _gub,
        "k": len(_gdata), "meta_model": _gmeth,
    }])

    plot_est = pd.concat(
        [pair_meta, brain_margin, epi_margin, grand_row], ignore_index=True
    )
    plot_est["brain_model_facet"] = plot_est["brain_model"].apply(
        lambda x: "Epi clocks (pooled)" if x == "Pooled" else x
    )

    pair_raw = d[["cohort", "brain_model", "epi_model",
                  "RLM_Estimate_scaled"]].dropna().copy()
    pair_raw["brain_model_facet"] = pair_raw["brain_model"]

    # Fixed epi y-order: Pooled at bottom (y=0), Gen2+ above (reversed alpha),
    # Gen1 at top (reversed alpha) — matches R: epi_levels <- c("Pooled", rev(gen2plus), rev(gen1))
    present_epi = set(pair_raw["epi_model"].dropna().unique())
    gen1_p  = [m for m in _GEN1_ASSOC  if m in present_epi]
    gen2_p  = [m for m in _GEN2P_ASSOC if m in present_epi]
    extra_p = [m for m in sorted(present_epi)
               if m not in _GEN1_ASSOC and m not in _GEN2P_ASSOC]
    epi_y_order = ["Pooled"] + list(reversed(gen2_p)) + list(reversed(gen1_p)) + extra_p

    # Divider y-positions
    pooled_div_y = 0.5                        # solid:  Pooled / Gen2+
    gen_div_y    = 0.5 + len(gen2_p)          # dashed: Gen2+  / Gen1

    # Fixed brain panel order to match paper (left → right)
    present_brain = {m for m in plot_est["brain_model_facet"].unique()
                     if m != "Epi clocks (pooled)"}
    brain_panels_ordered = [m for m in _BRAIN_ORDER if m in present_brain]
    remaining_brain = [m for m in sorted(present_brain) if m not in _BRAIN_ORDER]
    facet_panels = brain_panels_ordered + remaining_brain + ["Epi clocks (pooled)"]

    # ── Fig 4C ────────────────────────────────────────────────────────────────
    assoc_meta_violin = _compute_meta_assoc(d, group_cols=["age_bin", "model_combi"])
    counts = _k_counts(d)

    all_combis    = sorted(assoc_meta_violin["model_combi"].unique(), key=str.lower)
    combi_palette = {
        c: KELLY_COLORS[i % len(KELLY_COLORS)]
        for i, c in enumerate(all_combis)
    }

    return (plot_est, pair_raw, epi_y_order, facet_panels,
            pooled_div_y, gen_div_y,
            assoc_meta_violin, counts, combi_palette)


@st.cache_data
def _load_r_assoc_meta():
    try:
        r_est = pd.read_csv("data/assoc_plot_est.csv").rename(columns={
            "b":     "pooled_beta",
            "ci.lb": "ci_lb",
            "ci.ub": "ci_ub",
        })
        r_est["brain_model_facet"] = r_est["brain_model"].apply(
            lambda x: "Epi clocks (pooled)" if x == "Pooled" else x
        )
        return r_est
    except FileNotFoundError:
        return None

with st.spinner("Running meta-analyses…"):
    (plot_est, pair_raw, epi_y_order, facet_panels,
     pooled_div_y, gen_div_y,
     assoc_meta_violin, counts, combi_palette) = _compute_assoc_pub_data(filtered)

if len(filtered) == len(df):
    _r_plot_est = _load_r_assoc_meta()
    if _r_plot_est is not None:
        plot_est = _r_plot_est


# ── Publication figures ────────────────────────────────────────────────────────
st.divider()
st.header("Publication Figures")
st.caption("Replicating paper figures — responds to sidebar filters on the left.")

# ── Shared focus selector (applies to both Fig 4A and Fig 4C) ─────────────────
_fc1, _fc2 = st.columns([1, 2])
_focus_type = _fc1.radio(
    "Focus on",
    ["All models", "One brain model", "One epi clock"],
    index=0,
)
if _focus_type == "One brain model":
    _brain_opts = [p for p in facet_panels if p != "Epi clocks (pooled)"]
    _sel_brain  = _fc2.selectbox("Brain model", _brain_opts)
    _sel_epi    = None
elif _focus_type == "One epi clock":
    _epi_opts  = [m for m in epi_y_order if m != "Pooled"]
    _sel_epi   = _fc2.selectbox("Epi clock", _epi_opts)
    _sel_brain = None
else:
    _sel_brain = None
    _sel_epi   = None

# Derive display variables for Fig 4A
if _focus_type == "One brain model":
    _disp_panels = [_sel_brain, "Epi clocks (pooled)"]
    _disp_raw    = pair_raw
    _disp_est    = plot_est
    _disp_y      = epi_y_order
elif _focus_type == "One epi clock":
    _disp_panels = facet_panels
    _disp_raw    = pair_raw[pair_raw["epi_model"] == _sel_epi]
    _disp_est    = plot_est[plot_est["epi_model"].isin([_sel_epi, "Pooled"])]
    _disp_y      = ["Pooled", _sel_epi]
else:
    _disp_panels = facet_panels
    _disp_raw    = pair_raw
    _disp_est    = plot_est
    _disp_y      = epi_y_order

# Derive filtered data for Fig 4C
if _focus_type == "One brain model":
    _raw_4c  = filtered[filtered["brain_model"] == _sel_brain]
    _meta_4c = assoc_meta_violin[
        assoc_meta_violin["model_combi"].isin(
            filtered[filtered["brain_model"] == _sel_brain]["model_combi"].dropna().unique()
        )
    ]
elif _focus_type == "One epi clock":
    _raw_4c  = filtered[filtered["epi_model"] == _sel_epi]
    _meta_4c = assoc_meta_violin[
        assoc_meta_violin["model_combi"].isin(
            filtered[filtered["epi_model"] == _sel_epi]["model_combi"].dropna().unique()
        )
    ]
else:
    _raw_4c  = filtered
    _meta_4c = assoc_meta_violin

# ── Fig 4B chord diagram function ─────────────────────────────────────────────
def _make_chord_fig(pm, brain_models, epi_models):
    """
    Chord diagram matching R layout:
      Brain models: exact top semicircle    (5° → 175° CCW)
      Epi clocks:   exact bottom semicircle (185° → 355° CCW)
    Bezier curves connect pairs; colour = sign of pooled β, width = |β|.
    Labels are radial (pointing outward from centre) with readable orientation.
    """
    pm = pm.dropna(subset=["pooled_beta"]).copy()
    pm = pm[pm["brain_model"].isin(brain_models) & pm["epi_model"].isin(epi_models)]
    if pm.empty:
        return go.Figure()

    # ── Colour palettes ────────────────────────────────────────────────────
    _BRAIN_PAL = {
        "Centile2":     "#8B1A1A",
        "DBN":          "#1874CD",
        "DevBrainAge":  "#20B2AA",
        "DunedinPACNI": "#2E8B57",
        "ENIGMA":       "#9370DB",
        "Kaufmann":     "#E08B00",
        "PyBrainAge":   "#6B238E",
        "Pyment":       "#4BA3C7",
    }
    for i, m in enumerate(brain_models):
        if m not in _BRAIN_PAL:
            _BRAIN_PAL[m] = KELLY_COLORS[i % len(KELLY_COLORS)]
    _EPI_PAL = {m: KELLY_COLORS[i % len(KELLY_COLORS)] for i, m in enumerate(epi_models)}

    # ── Geometry ───────────────────────────────────────────────────────────
    OUTER_R = 1.0
    ARC_T   = 0.08
    INNER_R = OUTER_R - ARC_T
    LABEL_R = OUTER_R + 0.26
    SEG_GAP = np.radians(0.8)

    # Brain: 5° → 175° counterclockwise through the top (exact top semicircle, ±5° gap)
    # Epi:  185° → 355° counterclockwise through the bottom (exact bottom semicircle, ±5° gap)
    B_START, B_END = np.radians(5),   np.radians(175)
    E_START, E_END = np.radians(185), np.radians(355)

    def assign_arcs(models, arc_s, arc_e):
        n = len(models)
        if n == 0:
            return {}
        span = (arc_e - arc_s - (n - 1) * SEG_GAP) / n
        arcs = {}
        for i, m in enumerate(models):
            s = arc_s + i * (span + SEG_GAP)
            e = s + span
            arcs[m] = (s, e, (s + e) / 2)
        return arcs

    brain_arcs = assign_arcs(brain_models, B_START, B_END)
    epi_arcs   = assign_arcs(epi_models,   E_START, E_END)

    fig = go.Figure()

    # ── Outer arc segments ─────────────────────────────────────────────────
    def add_arc_seg(m, s, e, color):
        ang = np.linspace(s, e, 60)
        xo, yo = OUTER_R * np.cos(ang), OUTER_R * np.sin(ang)
        xi, yi = INNER_R * np.cos(ang[::-1]), INNER_R * np.sin(ang[::-1])
        fig.add_trace(go.Scatter(
            x=list(xo) + list(xi) + [xo[0]],
            y=list(yo) + list(yi) + [yo[0]],
            fill="toself", fillcolor=color,
            line=dict(color=color, width=0.3),
            mode="lines", showlegend=False,
            hoverinfo="name", name=m,
        ))

    for m, (s, e, _) in brain_arcs.items():
        add_arc_seg(m, s, e, _BRAIN_PAL.get(m, "#888"))
    for m, (s, e, _) in epi_arcs.items():
        add_arc_seg(m, s, e, _EPI_PAL.get(m, "#888"))

    # ── Bezier chords ──────────────────────────────────────────────────────
    beta_max = pm["pooled_beta"].abs().max()
    pm_sorted = pm.reindex(pm["pooled_beta"].abs().sort_values(ascending=True).index)
    t    = np.linspace(0, 1, 80)
    CTRL = 0.15

    for _, row in pm_sorted.iterrows():
        bm, em, beta = row["brain_model"], row["epi_model"], row["pooled_beta"]
        if bm not in brain_arcs or em not in epi_arcs:
            continue
        bx = INNER_R * np.cos(brain_arcs[bm][2])
        by = INNER_R * np.sin(brain_arcs[bm][2])
        ex = INNER_R * np.cos(epi_arcs[em][2])
        ey = INNER_R * np.sin(epi_arcs[em][2])
        xb = (1-t)**3*bx + 3*(1-t)**2*t*(CTRL*bx) + 3*(1-t)*t**2*(CTRL*ex) + t**3*ex
        yb = (1-t)**3*by + 3*(1-t)**2*t*(CTRL*by) + 3*(1-t)*t**2*(CTRL*ey) + t**3*ey
        abs_b = abs(beta)
        alpha = 0.10 + 0.40 * (abs_b / beta_max)
        width = 0.3  + 2.5  * (abs_b / beta_max)
        color = (f"rgba(91,155,213,{alpha:.2f})" if beta > 0
                 else f"rgba(244,130,100,{alpha:.2f})")
        fig.add_trace(go.Scatter(
            x=xb, y=yb, mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
            hovertemplate=f"<b>{bm} – {em}</b><br>β = {beta:.3f}<extra></extra>",
        ))

    # ── Radial labels (perpendicular to arc edge) ──────────────────────────
    # Radial labels, always readable. Annotation point is placed just outside
    # the arc; text starts there and extends outward.
    # Right half (deg < 90 or deg > 270): text reads outward, first letter nearest arc.
    # Left half (90 ≤ deg ≤ 270): flip 180° so it stays readable; last letter nearest arc.
    def add_label(m, mid, label_r, fsize=9):
        deg = np.degrees(mid % (2 * np.pi))   # 0–360 standard CCW
        x   = label_r * np.cos(mid)
        y   = label_r * np.sin(mid)
        # Plotly textangle is clockwise; standard deg is CCW, so negate.
        # Right half: first letter near arc, text reads outward.
        # Left half: flip 180° so text stays readable; last letter near arc.
        if deg < 90 or deg > 270:
            tangle, ax = -deg, "center"
        else:
            tangle, ax = 180 - deg, "center"
        fig.add_annotation(
            x=x, y=y, text=m, showarrow=False,
            font=dict(size=fsize, color="#333333"),
            textangle=tangle, xanchor=ax, yanchor="middle",
        )

    for m, (_, _, mid) in brain_arcs.items():
        add_label(m, mid, LABEL_R, fsize=11)

    # Epi labels: all at same radius — radial orientation prevents collision
    for m, (_, _, mid) in epi_arcs.items():
        add_label(m, mid, LABEL_R, fsize=10)

    # ── Legend ─────────────────────────────────────────────────────────────
    for name, col in [("Positive β", "rgba(91,155,213,0.85)"),
                      ("Negative β", "rgba(244,130,100,0.85)")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=col, size=10, symbol="square"),
            name=name, showlegend=True,
            legendgroup="dir", legendgrouptitle_text="Direction",
        ))


    fig.update_layout(
        height=680,
        # scaleanchor ensures a perfect circle regardless of container width
        xaxis=dict(visible=False, range=[-1.42, 1.42]),
        yaxis=dict(visible=False, range=[-1.42, 1.42], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            x=1.01, y=0.5, xanchor="left", yanchor="middle",
            font=dict(size=10),
            tracegroupgap=10,
        ),
        margin=dict(l=20, r=160, t=10, b=10),
    )
    return fig


tab_4a, tab_4b, tab_4c = st.tabs([
    "Fig 4A — Associations by model pair",
    "Fig 4B — Association structure (chord)",
    "Fig 4C — Associations by age group",
])


# ── Fig 4A ────────────────────────────────────────────────────────────────────
with tab_4a:
    st.subheader("Brain-PAR – Epi-PAR associations across model combinations")
    _4a_col1, _4a_col2 = st.columns([3, 1])
    _4a_col1.caption(
        "Each panel shows one brain model. Coloured dots = raw cohort-level associations "
        "(hover for details); black diamonds = pooled meta-analysis estimates per epi clock. "
        "Final panel pools across all brain models. "
        "Tip: click and drag on any panel to zoom in; double-click to reset."
    )
    _sig_only      = _4a_col2.checkbox("Sig. epi clocks only (pooled across brain models)", value=False)
    _sig_brain_only = _4a_col2.checkbox("Sig. brain models only (pooled across epi clocks)", value=False)

    if _sig_only:
        # Significant = CI doesn't cross zero in the "Epi clocks (pooled)" panel
        # (brain_margin rows: pooled across all brain models)
        _sig_epi = set(
            _disp_est[
                (_disp_est["brain_model_facet"] == "Epi clocks (pooled)") &
                (_disp_est["epi_model"] != "Pooled") &
                ((_disp_est["ci_lb"] > 0) | (_disp_est["ci_ub"] < 0))
            ]["epi_model"].dropna()
        )
        _disp_y   = ["Pooled"] + [m for m in _disp_y if m in _sig_epi]
        _disp_est = _disp_est[_disp_est["epi_model"].isin(set(_disp_y))]
        _disp_raw = _disp_raw[_disp_raw["epi_model"].isin(set(_disp_y))]

    if _sig_brain_only:
        # Significant = CI doesn't cross zero in the brain_margin (pooled across all epi clocks)
        _sig_brain = set(
            _disp_est[
                (_disp_est["epi_model"] == "Pooled") &
                (_disp_est["brain_model_facet"] != "Epi clocks (pooled)") &
                ((_disp_est["ci_lb"] > 0) | (_disp_est["ci_ub"] < 0))
            ]["brain_model_facet"].dropna()
        )
        # Always keep the "Epi clocks (pooled)" summary panel
        _disp_panels = [p for p in _disp_panels if p in _sig_brain or p == "Epi clocks (pooled)"]

    n_panels = len(_disp_panels)
    n_cols   = min(5, n_panels)
    n_rows   = math.ceil(n_panels / n_cols)

    fig_4a = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=_disp_panels,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.14,
    )

    y_pos         = {m: i for i, m in enumerate(_disp_y)}
    shown_cohorts = set()
    rng           = np.random.default_rng(42)



    for idx, panel in enumerate(_disp_panels):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        # vertical reference line at x = 0 — drawn below all traces
        fig_4a.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-0.5, y1=len(epi_y_order) - 0.4,
            line=dict(color="#999999", width=1.2),
            layer="below",
            row=r, col=c,
        )

        # solid divider: Pooled / Gen2+
        fig_4a.add_hline(
            y=pooled_div_y,
            line_color="#555555", line_width=1.0,
            row=r, col=c,
        )
        # dashed divider: Gen2+ / Gen1
        if gen_div_y > pooled_div_y:
            fig_4a.add_hline(
                y=gen_div_y,
                line_color="#555555", line_width=0.8,
                line_dash="dash",
                row=r, col=c,
            )

        # raw cohort scatter
        raw_sub = _disp_raw[_disp_raw["brain_model_facet"] == panel].copy()
        raw_sub["y_num"] = raw_sub["epi_model"].map(y_pos)
        raw_sub = raw_sub.dropna(subset=["y_num", "RLM_Estimate_scaled"])
        raw_sub["y_jitter"] = (
            raw_sub["y_num"] + rng.uniform(-0.18, 0.18, len(raw_sub))
        )

        for cohort in sorted(raw_sub["cohort"].dropna().unique()):
            sub  = raw_sub[raw_sub["cohort"] == cohort]
            show = cohort not in shown_cohorts
            fig_4a.add_trace(go.Scatter(
                x=sub["RLM_Estimate_scaled"],
                y=sub["y_jitter"],
                mode="markers",
                marker=dict(
                    color=COHORT_PALETTE.get(cohort, "#888888"),
                    size=5, opacity=0.55,
                ),
                name=cohort,
                legendgroup=cohort,
                showlegend=show,
                hovertemplate=(
                    f"<b>{cohort}</b><br>"
                    "Epi: %{customdata}<br>"
                    "β = %{x:.3f}<extra></extra>"
                ),
                customdata=sub["epi_model"].values,
            ), row=r, col=c)
            shown_cohorts.add(cohort)

        # meta estimates
        est_sub = _disp_est[_disp_est["brain_model_facet"] == panel]
        for _, est in est_sub.iterrows():
            yi = y_pos.get(est["epi_model"])
            if yi is None:
                continue
            is_pooled = (est["epi_model"] == "Pooled")

            # Colour by significance: purple = significant (CI doesn't cross zero), black = n.s.
            _lb, _ub = est["ci_lb"], est["ci_ub"]
            if not (pd.isna(_lb) or pd.isna(_ub)):
                _is_sig = (_lb > 0) or (_ub < 0)
                _est_color = "#7B2FBE" if _is_sig else "black"
            else:
                _is_sig = False
                _est_color = "black"

            # CI line
            if not (pd.isna(_lb) or pd.isna(_ub)):
                fig_4a.add_trace(go.Scatter(
                    x=[_lb, _ub],
                    y=[yi, yi],
                    mode="lines",
                    line=dict(color=_est_color, width=2.5 if is_pooled else 1.0),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=r, col=c)

            # point estimate (diamond for pooled, filled circle for per-epi)
            fig_4a.add_trace(go.Scatter(
                x=[est["pooled_beta"]],
                y=[yi],
                mode="markers",
                marker=dict(
                    color=_est_color,
                    symbol="diamond" if is_pooled else "circle",
                    size=7 if is_pooled else 6,
                ),
                name="Overall pooled",
                legendgroup="meta",
                showlegend=False,
                hovertemplate=(
                    f"<b>{est['epi_model']}</b><br>"
                    f"β = {est['pooled_beta']:.3f}<br>"
                    f"95% CI: [{est['ci_lb']:.3f}, {est['ci_ub']:.3f}]"
                    "<extra>Meta</extra>"
                ),
            ), row=r, col=c)

    # Pooled estimate legend entries — two states: significant (purple) and n.s. (black)
    # The <br> in the group title adds a small gap below the heading without a full blank row.
    for i, (_leg_color, _leg_name) in enumerate([("#7B2FBE", "Pooled (significant)"), ("black", "Pooled (n.s.)")]):
        fig_4a.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=_leg_color, symbol="diamond", size=5),
            name=_leg_name,
            legendgroup="meta",
            legendgrouptitle_text="Meta-analysis" if i == 0 else None,
            showlegend=True,
        ))

    # y-axis tick labels (shared) — only show on leftmost column
    fig_4a.update_yaxes(
        tickmode="array",
        tickvals=list(y_pos.values()),
        ticktext=list(y_pos.keys()),
        tickfont=dict(size=8),
        range=[-0.6, len(_disp_y) - 0.4],
    )

    # Fixed ±0.25 x-range with ticks at -0.25, 0.00, 0.25 (matching publication figure)
    fig_4a.update_xaxes(
        range=[-0.25, 0.25],
        tickmode="array",
        tickvals=[-0.25, 0.00, 0.25],
        ticktext=["-0.25", "0.00", "0.25"],
        tickfont=dict(size=8),
        title_font=dict(size=8),
        zeroline=False,
    )
    fig_4a.update_annotations(font_size=10)  # subplot titles

    fig_4a.update_layout(
        height=n_rows * 500,
        legend=dict(
            title="Cohort",
            x=0.99, y=0.45,
            xanchor="right",
            yanchor="top",
            font=dict(size=8),
            itemsizing="constant",
            tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=130, r=40, t=80, b=40),
    )

    st.plotly_chart(fig_4a, use_container_width=True)
    if len(filtered) == len(df):
        st.caption("Pooled estimates from multilevel random-effects meta-analysis (rma.mv, REML; metafor R package).")
    else:
        st.caption("Pooled estimates from DerSimonian-Laird random-effects meta-analysis (approximate; applied to filtered subset).")


# ── Fig 4B ────────────────────────────────────────────────────────────────────
with tab_4b:
    st.subheader("Chord diagram of Brain-PAR – Epi-PAR association strength across model combinations")
    st.caption(
        "Chord diagram of Brain-PAR – Epi-PAR pooled associations. "
        "Upper arc = brain models (coloured segments); lower arc = epi clocks. "
        "Chord width ∝ |β|; blue = positive β, salmon = negative β. "
        "Hover over a chord for exact values."
    )

    # ── Model order matches the R paper figure ─────────────────────────────
    # Brain: Pyment → ENIGMA → Kaufmann → DBN → DunedinPACNI → DevBrainAge
    #         → PyBrainAge → Centile2  (upper-right arc, CCW through top to upper-left)
    _BRAIN_ORDER_4B = [
        "Pyment", "ENIGMA", "Kaufmann", "DBN",
        "DunedinPACNI", "DevBrainAge", "PyBrainAge", "Centile2",
    ]
    # Epi: DunedinPACE → … → DNAmTL  (lower-left arc, CCW through bottom to lower-right)
    _EPI_ORDER_4B = [
        "DunedinPACE", "PCGrimAge", "PCBrainAge", "CorticalClock",
        "AltumAge", "skinHorvath", "DamAge", "ZhangBLUP",
        "Horvath2013", "PhenoAge", "ZhangEN", "Wu",
        "cAge", "PedBE", "AdaptAge", "Hannum", "DNAmTL",
    ]

    _present_b = set(plot_est.loc[plot_est["brain_model"] != "Pooled", "brain_model"].dropna())
    _present_e = set(plot_est.loc[plot_est["epi_model"]   != "Pooled", "epi_model"].dropna())

    # Ordered lists filtered to models actually present; append any extras at end
    _brain_4b = [m for m in _BRAIN_ORDER_4B if m in _present_b]
    _brain_4b += [m for m in sorted(_present_b) if m not in set(_BRAIN_ORDER_4B)]
    _epi_4b   = [m for m in _EPI_ORDER_4B   if m in _present_e]
    _epi_4b   += [m for m in sorted(_present_e) if m not in set(_EPI_ORDER_4B)]

    _pm_4b = plot_est[
        (plot_est["brain_model"] != "Pooled") &
        (plot_est["epi_model"]   != "Pooled")
    ].copy()

    # Apply Focus-on filter
    if _focus_type == "One brain model" and _sel_brain:
        _pm_4b    = _pm_4b[_pm_4b["brain_model"] == _sel_brain]
        _brain_4b = [m for m in _brain_4b if m == _sel_brain]
    elif _focus_type == "One epi clock" and _sel_epi:
        _pm_4b  = _pm_4b[_pm_4b["epi_model"] == _sel_epi]
        _epi_4b = [m for m in _epi_4b if m == _sel_epi]

    fig_4b = _make_chord_fig(_pm_4b, _brain_4b, _epi_4b)
    st.plotly_chart(fig_4b, use_container_width=True)
    if len(filtered) == len(df):
        st.caption("Pooled estimates from multilevel random-effects meta-analysis (rma.mv, REML; metafor R package).")
    else:
        st.caption("Pooled estimates from DerSimonian-Laird random-effects meta-analysis (approximate; applied to filtered subset).")


# ── Fig 4C ────────────────────────────────────────────────────────────────────
with tab_4c:
    st.subheader("Brain-PAR – Epi-PAR associations by developmental age group")
    st.caption(
        "Violin + boxplot of association betas by brain developmental age group. "
        "Coloured dots = pooled meta-analysis estimates per model combination "
        "(hover for details; no legend — too many combinations). "
        "Pooled estimates from DerSimonian-Laird random-effects meta-analysis (approximate)."
    )

    fig_4c = go.Figure()

    bin_x = {b: i for i, b in enumerate(BRAIN_BIN5_LEVELS)}
    BOX_OFF  = -0.22   # box shifted left of centre
    DOT_START = 0.05   # meta dots start just right of centre
    DOT_END   = 0.40   # meta dots end here

    # y=0 reference (behind everything)
    fig_4c.add_shape(
        type="line", x0=-0.5, x1=len(BRAIN_BIN5_LEVELS) - 0.5,
        y0=0, y1=0, layer="below",
        line=dict(color="#cccccc", width=1.0),
    )

    # ── Violin (grey shading, behind box) ─────────────────────────────────
    for b in BRAIN_BIN5_LEVELS:
        xi     = bin_x[b]
        subset = _raw_4c.loc[_raw_4c["age_bin"] == b, "RLM_Estimate_scaled"].dropna()
        if subset.empty:
            continue
        fig_4c.add_trace(go.Violin(
            y=subset,
            x=[xi + BOX_OFF] * len(subset),
            name=b,
            fillcolor="rgba(200,200,200,0.45)",
            line=dict(color="rgba(160,160,160,0.4)", width=0.5),
            points=False,
            box_visible=False,
            meanline_visible=False,
            width=0.55,
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Box plot, offset left (on top of violin) ───────────────────────────
    for b in BRAIN_BIN5_LEVELS:
        xi     = bin_x[b]
        subset = _raw_4c.loc[_raw_4c["age_bin"] == b, "RLM_Estimate_scaled"].dropna()
        if subset.empty:
            continue
        fig_4c.add_trace(go.Box(
            y=subset,
            x=[xi + BOX_OFF] * len(subset),
            name=b,
            boxpoints=False,
            fillcolor="rgba(240,240,240,0.8)",
            line=dict(color="#222222", width=1.5),
            width=0.22,
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Meta estimates + vertical CIs, offset right ───────────────────────
    all_combis = sorted(_meta_4c["model_combi"].dropna().unique(), key=str.lower)
    n_c        = len(all_combis)
    offsets    = np.linspace(DOT_START, DOT_END, n_c) if n_c > 1 else [(DOT_START + DOT_END) / 2]
    off_map    = dict(zip(all_combis, offsets))

    for combi in all_combis:
        sub = _meta_4c[_meta_4c["model_combi"] == combi]
        if sub.empty:
            continue
        color = combi_palette.get(combi, "#888888")
        off   = off_map[combi]

        rows = [
            (row["age_bin"], row["pooled_beta"],
             row.get("ci_lb", float("nan")), row.get("ci_ub", float("nan")))
            for _, row in sub.iterrows() if row["age_bin"] in bin_x
        ]
        if not rows:
            continue

        _CI_CAP = 0.5   # don't display CI arms wider than this
        xs     = [bin_x[b] + off           for b, _, _, _ in rows]
        ys     = [beta                      for _, beta, _, _ in rows]
        e_hi   = [min(ub - beta, _CI_CAP) if pd.notna(ub) else 0 for _, beta, _, ub in rows]
        e_lo   = [min(beta - lb, _CI_CAP) if pd.notna(lb) else 0 for _, beta, lb, _ in rows]

        fig_4c.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.85,
                        line=dict(color="white", width=0.6)),
            error_y=dict(
                type="data", symmetric=False,
                array=e_hi, arrayminus=e_lo,
                color="rgba(180,180,180,0.25)", thickness=0.8, width=0,
            ),
            name=combi,
            showlegend=False,
            hovertemplate=(
                f"<b>{combi}</b><br>"
                "Age: %{customdata}<br>"
                "β = %{y:.3f}<extra>Meta</extra>"
            ),
            customdata=[b for b, _, _, _ in rows],
        ))

    # ── k= labels — Scatter text traces so hover works ────────────────────
    # Build per-bin hover text and k counts directly from _raw_4c
    # (avoids pandas Categorical/StringDtype issues in _k_counts on newer pandas)

    def _parse_array(tp):
        tp_up = str(tp).upper()
        for arr in ["EPIC", "450K", "850K", "27K"]:
            if arr in tp_up:
                return arr
        return None

    cnt_map = {}
    _bin_hover = {}
    for b in bin_x.keys():
        _sub = _raw_4c.loc[_raw_4c["age_bin"] == b].dropna(subset=["cohort", "timepoint", "mean_age"])
        if _sub.empty:
            continue
        _uniq = _sub.groupby(["cohort", "timepoint"])["mean_age"].mean().reset_index().sort_values("mean_age")
        cnt_map[str(b)] = len(_uniq)
        lines = []
        for _, row in _uniq.iterrows():
            age_str = f"{row['mean_age']:.1f} y"
            arr = _parse_array(row["timepoint"])
            arr_str = f", {arr}" if arr else ""
            lines.append(f"{row['cohort']}{arr_str} ({age_str})")
        _bin_hover[str(b)] = "<br>".join(lines)

    y_floor = _raw_4c["RLM_Estimate_scaled"].min(skipna=True) if not _raw_4c.empty else -0.3
    k_y = y_floor - 0.05
    for b, xi in bin_x.items():
        k = cnt_map.get(str(b))
        if k is not None:
            detail = _bin_hover.get(str(b), "")
            fig_4c.add_trace(go.Scatter(
                x=[xi], y=[k_y],
                mode="text",
                text=[f"k={k}"],
                textfont=dict(size=10, color="black"),
                textposition="middle center",
                hovertemplate=(
                    f"<b>k={k} cohort-timepoints</b><br>{detail}<extra></extra>"
                ),
                showlegend=False,
            ))

    fig_4c.update_xaxes(
        tickmode="array",
        tickvals=list(bin_x.values()),
        ticktext=list(bin_x.keys()),
        tickangle=30,
        title_text="Developmental age group",
    )
    fig_4c.update_yaxes(title_text="Robust standardised β", range=[k_y - 0.08, 0.55])
    fig_4c.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=60, r=40, t=20, b=110),
        boxmode="overlay",
        violinmode="overlay",
    )

    st.plotly_chart(fig_4c, use_container_width=True)

render_footer()
