import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# cached loader for metrics.json
@st.cache_data
def load_metrics(version="v1"):
    metrics_path = Path("outputs") / version / "metrics.json"
    return json.loads(metrics_path.read_text())

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

    # Objectives & User Stories
    st.subheader("Objectives & User Stories")
    st.markdown("""
    - **Objective 1:** Visually explore differences between healthy and infected leaves  
      - *User Story:* As an agronomist, I want side-by-side comparisons of leaf samples so I can understand key visual indicators of powdery mildew.  
    - **Objective 2:** Automate detection with high accuracy  
      - *User Story:* As a farm manager, I want the model to flag infected leaves with ≥ 90 % recall so that I can prioritize treatment quickly.
    """)

    # Hypotheses
    st.subheader("Project Hypotheses & Validation")
    st.markdown("""
    1. **Model Accuracy Hypothesis**  
       A CNN fine-tuned on cherry-leaf images will achieve ≥ 90 % recall on the *Powdery Mildew* class  
       *Validation:* Evaluate on held-out test set, compute recall with a 95 % CI, and perform a one-sample t-test against the 50 % random baseline (α = 0.05).

    2. **Image Variance Hypothesis**  
       Powdery mildew leaves exhibit higher pixel-intensity variance than healthy leaves  
       *Validation:* Compute per-image variance for each class and apply a two-sample t-test (α = 0.05).
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
