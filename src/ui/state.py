"""Cached access to data and analysis results for the Streamlit app."""
from __future__ import annotations

import streamlit as st

from src.data_loader import load_data
from src.engine.analysis import AnalysisResult, run_analysis


@st.cache_resource(show_spinner="Loading plant data and running analytics…")
def get_result(source: str = "csv") -> AnalysisResult:
    return run_analysis(load_data(source))
