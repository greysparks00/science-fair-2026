# local web dashboard to visualize and compare results from different iterations/models

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Science Fair 2026 Dashboard",
    layout="wide"
)

st.title("Science Fair 2026 Dashboard")
st.caption("Machine Learning Experiment Dashboard")

# get data from ./runs/...json
runs = [
    json.loads(path.read_text())
    for path in Path("runs").glob("*.json")
]

if not runs:
    st.warning("No experiment results found.")
    st.stop()

# create a DataFrame from the list of runs
df = pd.DataFrame(runs)

st.subheader("Model Comparison")
st.dataframe(df, hide_index=True)

# chart for cross-validation performance
st.subheader("Cross-Validation Performance")

chart = df.set_index("model")[
    ["cv_accuracy", "cv_balanced_accuracy", "cv_roc_auc"]
]

st.bar_chart(chart, y_label="Score")