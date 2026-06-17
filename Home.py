import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="MIND Consortium Ageing Dashboard", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.title("MIND Consortium Ageing Dashboard")
st.markdown(
    "Exploring how epigenetic and brain age models perform across development "
    "(birth–24 years), using data from 15 cohorts and over 21,000 samples. "
    "Navigate the pages in the sidebar to explore model performance interactively."
)

st.divider()

# ── STUDY SCOPE ───────────────────────────────────────────────────────────────
st.subheader("Study Scope")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cohorts", "15")
c2.metric("Total Samples", "21,696")
c3.metric("Age Range", "0–24 years")
c4.metric("Epigenetic Clocks", "20")
c5.metric("Brain Age Models", "8")

st.divider()

# ── COHORT DATA (used by both map and table) ──────────────────────────────────
cohorts = pd.DataFrame([
    ["ALSPAC",            "UK",              "Population-based birth cohort",                       3, "172–1841",  "0–20"],
    ["BHRC",              "Brazil",          "High-risk cohort",                                    3, "406–739",   "10–17"],
    ["DCHS",              "South Africa",    "High-risk cohort",                                    1, "122",       "5"],
    ["GUSTO",             "Singapore",       "Population-based birth cohort",                       4, "118–810",   "0–6"],
    ["K2H infancy",       "Germany",         "High-risk cohort",                                    1, "189–228",   "0"],
    ["K2H childhood",     "Germany",         "High-risk cohort",                                    2, "50–384",    "7–9"],
    ["MTwiNS",            "US",              "Twin study, high-risk population-based birth cohort", 3, "149–310",   "8–15"],
    ["Oregon ADHD-1000",  "US",              "Community-recruited case-control cohort",             5, "40–187",    "9–17"],
    ["CannTeen",          "UK",              "High-risk cohort",                                    2, "54–125",    "16–17"],
    ["FinnBrain",         "Finland",         "Population-based birth cohort",                       1, "172",       "5"],
    ["GenR",              "The Netherlands", "Population-based birth cohort",                       5, "96–3199",   "0–18"],
    ["NICAP",             "Australia",       "Community cohort, screening for ADHD",                1, "70–82",     "10"],
    ["UCI cohort",        "US",              "Community cohort",                                    2, "42–133",    "0–5"],
    ["FFCWS",             "US",              "Population-based birth cohort",                       2, "114–1122",  "9–15"],
    ["TAG",               "US",              "Community cohort",                                    6, "54–151",    "11–19"],
], columns=[
    "Cohort abbreviation", "Country", "Study design",
    "Number of time points", "Sample size range", "Mean age range (years)"
])

# ── COHORT MAP ────────────────────────────────────────────────────────────────
st.subheader("Cohort Locations")

# Country-level centroids (accurate with available data)
country_centroids = {
    "UK":              (55.38,  -3.44),
    "Brazil":          (-14.24, -51.93),
    "South Africa":    (-30.56,  22.94),
    "Singapore":       (1.35,  103.82),
    "Germany":         (51.17,   10.45),
    "Finland":         (61.92,   25.75),
    "The Netherlands": (52.13,    5.29),
    "US":              (37.09,  -95.71),
    "Australia":       (-25.27, 133.78),
}

country_cohorts = cohorts.groupby("Country")["Cohort abbreviation"].apply(
    lambda x: ", ".join(x)
).reset_index().rename(columns={"Cohort abbreviation": "Cohort"})
country_cohorts["n"] = cohorts.groupby("Country").size().values
country_cohorts["lat"] = country_cohorts["Country"].map(lambda c: country_centroids[c][0])
country_cohorts["lon"] = country_cohorts["Country"].map(lambda c: country_centroids[c][1])

fig_map = px.scatter_geo(
    country_cohorts,
    lat="lat", lon="lon",
    size="n",
    color="Country",
    hover_name="Country",
    hover_data={"Cohort": True, "n": True, "lat": False, "lon": False},
    projection="natural earth",
    size_max=20,
)
fig_map.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(title="Country", x=1.0, y=0.5),
    height=380,
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption("Dot size reflects number of cohorts per country. Hover for cohort names.")

st.divider()

# ── COHORT DESCRIPTIVES ───────────────────────────────────────────────────────
st.subheader("Cohort Descriptives")

st.table(cohorts.set_index("Cohort abbreviation"))

st.divider()

# ── EPIGENETIC CLOCKS ─────────────────────────────────────────────────────────
st.subheader("Epigenetic Clocks")

epi_data = [
    ["1st gen", "Bernabeu 2023",        "cAge",               "Chronological age",         "Whole blood*",            "2.0–101.0"],
    ["1st gen", "Bohlin 2016",          "–",                  "Gestational age",            "Cord blood",              "~280 days"],
    ["1st gen", "de Lima Camillo 2022", "AltumAge",           "Chronological age",          "Pan-tissue",              "-0.8–112"],
    ["1st gen", "Haftorn 2021",         "EPIC",               "Gestational age",            "Cord blood",              "216–299 days"],
    ["1st gen", "Hannum 2013",          "–",                  "Chronological age",          "Whole blood",             "19.0–101.0"],
    ["1st gen", "Horvath 2013",         "Pan-tissue clock",   "Chronological age",          "51 tissues",              "-0.5–100.0"],
    ["1st gen", "Horvath 2018",         "Skin & Blood Clock", "Chronological age",          "8 tissues",               "-0.2–94.0"],
    ["1st gen", "Knight 2016",          "–",                  "Gestational age",            "Cord blood & blood spots","24–44 weeks"],
    ["1st gen", "McEwen 2019",          "PedBE",              "Chronological age",          "Buccal cells",            "0.2–19.5"],
    ["1st gen", "Shireby 2020",         "Cortical Clock",     "Chronological age",          "Cortex",                  "1.3–108.0"],
    ["1st gen", "Thrush 2022",          "PCBrainAge",         "Chronological age",          "Cortex",                  "20.0–97.0"],
    ["1st gen", "Wu 2019",              "–",                  "Chronological age",          "Whole blood",             "0.8–17.7"],
    ["1st gen", "Zhang Q 2019",         "ZhangEN",            "Chronological age",          "Whole blood*",            "2.0–104.0"],
    ["1st gen", "Zhang Q 2019",         "ZhangBLUB",          "Chronological age",          "Whole blood*",            "2.0–104.0"],
    ["2nd gen", "Higgins-Chen 2022",    "PCGrimAge",          "Mortality risk",             "Whole blood",             "24.0–92.0"],
    ["2nd gen", "Levine 2018",          "PhenoAge",           "Mortality risk",             "Whole blood",             "20–90+"],
    ["2nd gen", "Lu 2019",              "DNAmTL",             "Leukocyte telomere length",  "Whole blood",             "22.2–93.1"],
    ["3rd gen", "Belsky 2022",          "DunedinPACE",        "Pace of ageing",             "Whole blood",             "45"],
    ["4th gen", "Ying 2024",            "AdaptAge",           "Chronological age",          "Whole blood",             "23.7–75.0"],
    ["4th gen", "Ying 2024",            "DamAge",             "Chronological age",          "Whole blood",             "23.7–75.0"],
]

epi_df = pd.DataFrame(epi_data, columns=[
    "Generation", "Reference", "Name", "Trained Phenotype", "Tissue", "Age range",
])

st.table(epi_df.set_index("Generation"))
st.caption("Age range in years unless otherwise specified.")

st.divider()

# ── BRAIN AGE MODELS ──────────────────────────────────────────────────────────
st.subheader("Brain Age Models")

brain_data = [
    ["1st gen", "Bashyam 2020",        "DeepBrainNet",            "Chronological age", "minimal processing", "3.0–95.0"],
    ["1st gen", "Leonardsen 2022",     "Pyment",                  "Chronological age", "minimal processing", "3.0–85.0"],
    ["1st gen", "Cole (unpublished*)", "PyBrainAge",              "Chronological age", "extracted",          "2.0–100.0"],
    ["1st gen", "",                    "Developmental Brain Age",  "Chronological age", "extracted",          "9.0–19.0"],
    ["1st gen", "Han 2021",            "ENIGMA",                  "Chronological age", "extracted",          "18.0–75.0"],
    ["1st gen", "Kaufmann 2019",       "Brainage (Kaufmann)",     "Chronological age", "extracted",          "3.0–96.0"],
    ["1st gen", "Yu 2024",             "Centile Brain BrainAge2", "Chronological age", "extracted",          "5.0–40.0"],
    ["3rd gen", "Whitman 2025",        "DunedinPACNI",            "Pace of ageing",    "extracted",          "45"],
]

brain_df = pd.DataFrame(brain_data, columns=[
    "Generation", "Reference", "Name", "Trained Phenotype", "Processing", "Age range (years)",
])

st.table(brain_df.set_index("Generation"))
st.caption("\\* + 1 dataset saliva")

st.divider()

# ── KEY FINDINGS ──────────────────────────────────────────────────────────────
st.subheader("Key Findings")

f1, f2, f3 = st.columns(3)
f1.info("Epigenetic clock accuracy improves from childhood to early adulthood")
f2.info("Brain age model accuracy improves from childhood to early adulthood")
f3.info("Weak cross-sectional association between brain and epigenetic age estimates")
