# import streamlit as st

# st.set_page_config(
#     page_title="Model Performance Dashboard",
#     layout="wide"
# )

# st.title("Model Performance Dashboard")

# st.markdown("""
# Use the sidebar to navigate:

# - Brain Models
# - Epigenetic Clocks
# """)

import streamlit as st
import pandas as pd
import plotly.express as px

# colours are set in: .streamlit/config.toml 

st.set_page_config(page_title="MIND Consortium Ageing Dashboard", layout="wide")

@st.cache_data
def load_data():
    brain = pd.read_csv("data/perf_brain.csv")
    epi = pd.read_csv("data/perf_epi.csv")
    return brain, epi

brain_df, epi_df = load_data()

st.title("Developmental Brain–Epigenetic Ageing Dashboard")

st.markdown("""
### MIND Consortium Study Overview

This dashboard explores how different epigenetic and brain age estimation models perform across development (birth–24 years),
using data from 15 cohorts and over 21,000 samples.
""")

st.divider()

# --- STUDY SCOPE ---
st.subheader("Study Scope")

c1, c2, c3 = st.columns(3)

c1.metric("Cohorts", "15")
c2.metric("Total Samples", "21,696")
c3.metric("Age Range", "0–24 years")

st.divider()

# --- COHORT DESCRIPTIVES ---

cohorts = pd.DataFrame([
    ["ALSPAC", "UK", "Population-based birth cohort", 3, "172–1841", "0–20"],
    ["BHRC", "Brazil", "High-risk cohort", 3, "406–739", "10–17"],
    ["DCHS", "South Africa", "High-risk cohort", 1, "122", "5"],
    ["GUSTO", "Singapore", "Population-based birth cohort", 4, "118–810", "0–6"],
    ["Kids2Health (0y)", "Germany", "High-risk cohort", 1, "189–228", "0"],
    ["Kids2Health (8–9y)", "Germany", "High-risk cohort", 2, "50–384", "8–9"],
    ["MTwiNS", "US", "Twin study", 3, "149–310", "0–15"],
    ["Oregon ADHD-1000", "US", "Case-control cohort", 5, "14–187", "9–17"],
    ["CannTeen", "UK", "High-risk cohort", 2, "54–125", "16–17"],
    ["FinnBrain", "Finland", "Birth cohort", 1, "172", "5"],
    ["GenR", "Netherlands", "Birth cohort", 5, "96–3199", "0–18"],
    ["NICAP", "Australia", "ADHD screening cohort", 1, "70–82", "10"],
    ["UCI cohort", "US", "Community cohort", 2, "42–133", "0–5"],
    ["SAND / FFCWS", "US", "High-risk cohort", 2, "114–1122", "9–15"]
], columns=[
    "Cohort", "Country", "Design", "Timepoints", "Sample_Size_Range", "Age_Range"
])

st.subheader("Cohort Descriptives")

# c1, c2 = st.columns([2, 1])

# with c1:
#     st.dataframe(cohorts, use_container_width=True)

# with c2:
#     fig = px.bar(
#         cohorts,
#         x="Cohort",
#         y="Timepoints",
#         color="Country",
#         title="Longitudinal Depth per Cohort"
#     )
#     st.plotly_chart(fig, use_container_width=True)

st.dataframe(cohorts, use_container_width=True)

fig = px.bar(
        cohorts,
        x="Cohort",
        y="Timepoints",
        color="Country",
        title="Longitudinal Depth per Cohort"
    )
st.plotly_chart(fig, use_container_width=True)


import pandas as pd
import streamlit as st

# -------------------------
# EPIGENETIC CLOCKS
# -------------------------
epi_data = [
    ["Bernabeu 2023", "-", "cAge", "Chronological age", "Whole blood*", "2.0-101.0"],
    ["Bohlin 2016", "-", "-", "Gestational age", "Cord blood", "~280 days"],
    ["de Lima Camillo 2022", "AltumAge", "Chronological age", "Pan-tissue", "-", "-0.8-112"],
    ["Haftorn 2021", "EPIC", "Gestational age", "Cord blood", "-", "216-299 days"],
    ["Hannum 2013", "-", "Chronological age", "Whole blood", "-", "19.0-101.0"],
    ["Horvath 2013", "Pan-tissue clock", "Chronological age", "51 tissues", "-", "-0.5-100.0"],
    ["Horvath 2018", "Skin & Blood Clock", "Chronological age", "8 tissues", "-", "-0.2-94.0"],
    ["Knight 2016", "-", "Gestational age", "Cord blood & blood spots", "-", "24-44 weeks"],
    ["McEwen 2019", "PedBE", "Chronological age", "Buccal cells", "-", "0.2-19.5"],
    ["Shireby 2020", "Cortical Clock", "Chronological age", "Cortex", "-", "1.3-108.0"],
    ["Thrush 2022", "PCBrainAge", "Chronological age", "Cortex", "-", "20.0-97.0"],
    ["Wu 2019", "-", "Chronological age", "Whole blood", "-", "0.8-17.7"],
    ["Zhang Q 2019", "-", "Chronological age", "Whole blood*", "-", "2.0-104.0"],
    ["Higgins-Chen 2022", "PCGrimAge", "Mortality risk", "Whole blood", "-", "24.0-92.0"],
    ["Levine 2018", "PhenoAge", "Mortality risk", "Whole blood", "-", "20-90+"],
    ["Lu 2019", "DNAmTL", "Leukocyte telomere length", "Whole blood", "-", "22.2-93.1"],
    ["Belsky 2022", "DunedinPACE", "Pace of ageing", "Whole blood", "-", "45"],
    ["Ying 2024", "AdaptAge", "Chronological age", "Whole blood", "-", "23.7-75.0"],
    ["Ying 2024", "DamAge", "Chronological age", "Whole blood", "-", "23.7-75.0"],
]

epi_df = pd.DataFrame(
    epi_data,
    columns=[
        "Clock",
        "Training Sample Reference",
        "Name",
        "Trained Phenotype",
        "Tissue",
        "Age Range"
    ],
)

# -------------------------
# BRAIN AGE MODELS
# -------------------------
brain_data = [
    ["Bashyam 2020", "1st gen", "-", "DeepBrainNet", "Chronological age", "minimal processing", "3.0-95.0"],
    ["Leonardsen 2022", "1st gen", "-", "Pyment", "-", "-", "3.0-85.0"],
    ["Cole (unpublished*)", "1st gen", "-", "PyBrainAge", "-", "extracted", "2.0-100.0"],
    ["Han 2021", "1st gen", "ENIGMA", "-", "-", "-", "18.0-75.0"],
    ["Kaufmann 2019", "1st gen", "Brainage (Kaufmann)", "BrainAge", "-", "-", "3.0-96.0"],
    ["Yu 2024", "1st gen", "Centile Brain BrainAge2", "BrainAge2", "-", "-", "5.0-40.0"],
    ["Whitman 2025", "3rd gen", "DunedinPACNI", "Pace of ageing", "-", "-", "45"],
]

brain_df = pd.DataFrame(
    brain_data,
    columns=[
        "Training Sample",
        "Generation",
        "Reference",
        "Name",
        "Trained Phenotype",
        "Processing",
        "Age Range",
    ],
)

# -------------------------
# STREAMLIT DISPLAY
# -------------------------
st.subheader("Epigenetic Clocks")
st.dataframe(epi_df, use_container_width=True)

st.subheader("Brain Age Models")
st.dataframe(brain_df, use_container_width=True)


# --- KEY FINDINGS ---
st.subheader("Key Findings")

f1, f2, f3, f4 = st.columns(4)

f1.info("Epigenetic clock accuracy improves from childhood to early adulthood")
f2.info("Brain age accuracy improves from childhood to early adulthood")
f3.info("Weak association between brain and epigenetic age estimates")
f4.info("[place holder]")

st.divider()

# --- RESEARCH QUESTIONS ---
st.subheader("Research Questions")

st.markdown("""
- How do different epigenetic and brain age models perform in predicting chronological age
across development?
- How strong are cross-sectional associations between different early markers of biological age
(i.e., epigenetic and brain age)?
""")

st.divider()
