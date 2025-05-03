import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# cache loader for training history
@st.cache_data
def load_history(version="v1"):
    path = Path("outputs") / version / "history_run2.json"
    return json.loads(path.read_text())

def page_training_dashboard_body():
    st.title("Training Dashboard")
    
    st.info(
        """
        **What you’re seeing:**  
        Interactive plots of our Run 2 training history: toggle between loss and accuracy  
        to inspect convergence, and examine the top-performing epochs by validation accuracy.
        
        **How to interact:**  
        - **Metric selector:** Choose “Loss” or “Accuracy” to update the line chart.  
        - **Hover & zoom:** Hover over lines to see exact values; drag to zoom in on epochs.  
        - **Top epochs:** See which epochs achieved the highest validation accuracy.
        """
    )
    st.markdown("---")
    
    # Load history
    history = load_history()
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.reset_index(inplace=True)
    
    # Metric selector
    metric = st.selectbox("Select metric to plot", ["loss", "accuracy"], index=0)
    val_metric = f"val_{metric}"
    
    # Line chart of train vs. validation
    st.subheader(f"Run 2 {metric.capitalize()} over Epochs")
    fig = px.line(
        df,
        x="epoch",
        y=[metric, val_metric],
        labels={"value": metric.capitalize(), "epoch": "Epoch", "variable": "Dataset"},
        title=f"Training vs. Validation {metric.capitalize()}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)