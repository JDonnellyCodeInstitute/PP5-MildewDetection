import streamlit as st
import plotly.express as px
import random
from PIL import Image

# cache stats load and sample paths
from src.streamlit_utils import load_image_stats, list_image_paths


def page_leaf_atlas_body():
    st.title("Leaf Atlas")

    st.info(
        """
        **What you’re seeing:**
        This page visualises per-image statistics (mean and variance of pixel intensities)
        for healthy vs. powdery-mildew leaves. Use the histogram and scatter plots to compare
        how texture varies between classes, then browse randomly sampled images to see
        real examples from the dataset.

        **How to interact:**
        - **Histogram:** Hover or click legend items to focus on one class and inspect counts by variance.
        - **Scatter:** Zoom or pan to examine trends in mean vs. variance.
        - **Sample Collage:** Adjust the slider to show more or fewer random training images per class.
        """
    )

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

# Scatter of mean vs. variance
    st.subheader("Mean vs. Variance Scatter")
    scatter_fig = px.scatter(
        df_stats[df_stats["class"].isin(selected_classes)],
        x="mean",
        y="variance",
        color="class",
        title="Per-Image Mean vs. Variance",
        hover_data=["filepath"],
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Sample collage
    st.subheader("Random Sample Collage")
    n_samples = st.slider("Images per class", min_value=4, max_value=16, step=4, value=12)
    split_paths = list_image_paths(split="train")

    for cls in selected_classes:
        st.markdown(f"**{cls.title()} Samples**")
        paths = split_paths.get(cls, [])
        samples = random.sample(paths, min(n_samples, len(paths)))
        cols = st.columns(int(n_samples**0.5))
        for i, img_path in enumerate(samples):
            img = Image.open(img_path)
            cols[i % len(cols)].image(img, use_container_width=True)
