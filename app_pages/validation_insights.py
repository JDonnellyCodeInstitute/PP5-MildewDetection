import streamlit as st
import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Cache loaders
@st.cache_data
def load_metrics(version="v1"):
    p = Path("outputs") / version / "metrics.json"
    return json.loads(p.read_text())

@st.cache_data
def load_image_stats(version="v1"):
    return pd.read_csv(Path("outputs") / version / "image_stats.csv")

@st.cache_data
def load_confusion_matrix_image(version="v1"):
    return Path("outputs") / version / "figures" / "confusion_matrix.png"

def page_validation_insights_body():
    st.title("Validation & Insights")
    st.info(
        """
        **What you’re seeing:**  
        - The confusion matrix gives a summary of true vs. predicted labels on the test set.  
        - The boxplot shows how pixel‐variance differs between healthy and mildew leaves.  
        
        **Why it matters:**  
        - Confusion matrix reveals any remaining misclassifications and overall accuracy.  
        - Pixel‐variance boxplot ties back to our Image Variance Hypothesis, showing feature separation.
        """
    )
    st.markdown("---")

    # Load and display confusion matrix
    st.subheader("Confusion Matrix")
    cm_path = load_confusion_matrix_image()
    st.image(str(cm_path), caption="Test‐set Confusion Matrix", use_container_width=True)

    # Load metrics and show success/failure
    metrics = load_metrics()
    recall = metrics["recall_mildew"]
    if recall >= 0.90:
        st.success(f"✅ Model meets recall target: {recall:.0%} ≥ 90 %")
    else:
        st.error(f"❌ Model below recall target: {recall:.0%} < 90 %")

    # Boxplot of pixel variance by class
    st.subheader("Pixel Variance by Class (Test Set)")
    df_stats = load_image_stats()
    plt.figure(figsize=(6,4))
    sns.boxplot(x="class", y="variance", data=df_stats)
    plt.title("Pixel Variance Distribution by Class")
    plt.xlabel("Class")
    plt.ylabel("Variance")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Next steps if needed
    with st.expander("Next steps if recall < target"):
        st.write(
            """
            - Collect more diverse samples for the under‐performing class  
            - Increase augmentation intensity to reduce overfitting  
            - Experiment with alternative architectures or regularization  
            - Revisit thresholding on prediction probabilities  
            """
        )
