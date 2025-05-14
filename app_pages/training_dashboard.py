import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# cache loader for training history
from src.streamlit_utils import load_history

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
    
    # Bar chart of top-5 validation accuracies
    if metric == "accuracy":
        st.subheader("Top 5 Validation Accuracy Epochs")
        top5 = df.nlargest(5, "val_accuracy")[["epoch", "val_accuracy"]]
        bar = px.bar(
            top5,
            x="epoch",
            y="val_accuracy",
            labels={"epoch": "Epoch", "val_accuracy": "Validation Accuracy"},
            title="Top 5 Epochs by Validation Accuracy",
            text=top5["val_accuracy"].map("{:.2%}".format)
        )
        bar.update_traces(textposition="outside")
        st.plotly_chart(bar, use_container_width=True)
    
    st.markdown("---")
    st.write(
        "Use the selector above to switch between loss and accuracy. "
        "When viewing accuracy, explore the top epochs to see where the model performed best."
    )
