"""
Shared UI components: sidebar logo and page footer.
"""
import streamlit as st
from pathlib import Path

_LOGO_PATH = Path("assets/mind_logo.png")

_FOOTER_HTML = """
<hr style="margin-top: 3rem; border: none; border-top: 0.5px solid #ddd;">
<div style="text-align: center; color: #999; font-size: 11px; padding: 8px 0 16px;">
    <b style="color: #777;">Cite this dashboard:</b>
    Staginnus M, et al. (2026).
    Associations between epigenetic and brain age across development:
    Findings from the MIND Consortium. <i>In preparation.</i>
    <br>
    <b style="color: #777;">MIND Consortium:</b>
    Schuurmans et al., <i>Molecular Psychiatry</i> (2025) —
    <a href="https://www.nature.com/articles/s41380-025-03203-w"
       style="color: #999;">doi:10.1038/s41380-025-03203-w</a>
    &nbsp;·&nbsp;
    <a href="https://www.erasmusmc.nl/en/research/groups/methylation-imaging-and-neurodevelopment-mind-consortium"
       style="color: #999;">erasmusmc.nl/MIND</a>
    <br>
    Data shared for research purposes only. Contact the consortium before reuse.
    &nbsp;·&nbsp; © 2025 MIND Consortium
</div>
"""

def render_sidebar_logo():
    if _LOGO_PATH.exists():
        st.sidebar.image(str(_LOGO_PATH), use_container_width=True)
        st.sidebar.markdown(
            "<div style='margin-bottom: 0.5rem;'></div>",
            unsafe_allow_html=True,
        )

def render_footer():
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
