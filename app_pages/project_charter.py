import streamlit as st
import plotly.graph_objects as go

# cached loader for metrics.json
from src.streamlit_utils import load_metrics


def page_project_charter_body():
    # Header
    st.title("🍒 Cherry Leaf Powdery Mildew Detector Project Charter")
    st.markdown("---")

    # Overview
    st.markdown(
        """
        **Overview**

        Powdery mildew poses a serious threat to cherry production by reducing leaf
        photosynthesis and fruit quality. Currently, each tree requires ~30 min of
        manual leaf inspection. Our goal is to build a CNN-based system that
        classifies leaf images as *Healthy* or *Powdery Mildew* in under one second,
        enabling scalable, real-time monitoring across large orchards.
        """
    )

    # Link to the Kaggle dataset
    st.markdown(
        """
        **Dataset Source:**
        [Cherry Leaves Dataset on Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves)
        """
    )

    # Objectives & User Stories
    st.subheader("Objectives & User Stories")
    st.markdown("""
    - **Objective 1:** Ingest and prepare the cherry-leaf dataset
      - *User Story:* As a data engineer, I want to fetch and clean the dataset so that only valid leaf images remain ready for analysis.

    - **Objective 2:** Visually explore differences between healthy and infected leaves
      - *User Story:* As an agronomist, I want side-by-side comparisons, statistical summaries and montages so I can understand key visual mildew indicators.

    - **Objective 3:** Validate key hypotheses with statistical rigour
      - *User Story:* As a statistician, I want to see t-tests on pixel mean, variance and model recall so I can be confident our findings are significant.

    - **Objective 4:** Automate detection with high accuracy and usability
      - *User Story:* As a farm manager, I want the model to flag infected leaves with ≥ 90 % recall and an interactive interface to upload new images and download results.
    """)

    # Hypotheses
    st.subheader("Project Hypotheses & Validation")
    st.markdown("""
    1. **Model Accuracy Hypothesis**
    A CNN fine-tuned on cherry-leaf images will achieve ≥ 90 % recall on the *Powdery Mildew* class
    *Validation:* Evaluate on the held-out test set, compute recall with a 95 % CI, and perform a one-sample t-test against the 50 % random-guess baseline (α = 0.05).

    2. **Image Variance Hypothesis**
    Powdery mildew leaves exhibit higher pixel-intensity variance than healthy leaves
    *Validation:* Compute per-image variance for each class and apply a two-sample (Welch’s) t-test (α = 0.05).

    3. **Image Mean Intensity Hypothesis**
    The mean pixel-intensity of mildew-infected leaves differs from that of healthy leaves
    *Validation:* Compute per-image mean intensity and apply a two-sample (Welch’s) t-test (α = 0.05).

    4. **Learning-Rate & EarlyStopping Hypothesis**
    Lowering the learning rate to 1 × 10⁻⁴ *and* using EarlyStopping yields smoother convergence and higher final validation accuracy than the initial higher learning rate run.
    *Validation:* Compare Run 1 (LR = 1 × 10⁻³, no EarlyStopping) vs Run 2 (LR = 1 × 10⁻⁴, EarlyStopping) learning curves; quantify oscillations and plateau onset via epoch-wise val_loss variance and peak val_accuracy.
    """)

    # load live metrics
    metrics = load_metrics()
    current_recall = metrics["recall_mildew"]

    # Gauge chart: target vs. current recall
    st.subheader("Key Success Metric")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_recall,  # uses live recall
            delta={'reference': 0.90, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 1], 'tickformat': '.0%'},
                'bar': {'color': "#2E91E5"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.90
                },
            },
            title={'text': "Powdery Mildew Recall"}
        )
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.caption("Gauge compares current test recall against our 90 % target.")

    st.markdown("---")
    st.write(
        "This page anchors our Business Understanding phase: it defines our "
        "goals, user stories, hypotheses, and the primary success metric we "
        "must meet to validate the project."
    )
