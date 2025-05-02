import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# cache stats load
@st.cache_data
def load_image_stats(version="v1"):
    path = Path("outputs") / version / "image_stats.csv"
    return pd.read_csv(path)

# cache sample paths
@st.cache_data
def list_image_paths(split="train", version="v1"):
    base = Path("input/datasets/cherry_leaf_dataset/cherry-leaves") / split
    return {cls.name: list((base/cls.name).glob("*")) for cls in base.iterdir() if cls.is_dir()}

def page_leaf_atlas_body():
    st.title("Leaf Atlas")
    st.markdown("---")

    # Load stats
    df_stats = load_image_stats()

    # Sidebar filters
    st.sidebar.header("Visualization Controls")
    classes = df_stats["class"].unique().tolist()
    selected_classes = st.sidebar.multiselect(
        "Select classes to display", classes, default=classes
    )

    # Histogram of pixel variance
    st.subheader("Pixel Variance Distribution")
    hist_fig = px.histogram(
        df_stats[df_stats["class"].isin(selected_classes)],
        x="variance",
        color="class",
        nbins=50,
        title="Pixel Variance by Class",
        marginal="box",
        hover_data=["filepath"],
    )
    st.plotly_chart(hist_fig, use_container_width=True)

