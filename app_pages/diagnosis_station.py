import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.graph_objects as go

# Cache helpers
@st.cache_resource
def get_model(version="v1"):
    path = Path("models") / "run2_model.h5"
    return load_model(path)

@st.cache_data
def load_class_indices(version="v1"):
    path = Path("outputs") / version / "class_indices.json"
    return json.loads(path.read_text())

@st.cache_data
def load_image_shape(version="v1"):
    import pickle
    path = Path("outputs") / version / "image_shape.pkl"
    return pickle.loads(path.read_bytes())

def page_diagnosis_station_body():
    st.title("Diagnosis Station")
    st.info(
        """
        **What you’re seeing:**  
        Upload one or more leaf images to get instant Healthy vs. Powdery Mildew predictions with confidence scores.
        
        **How to interact:**  
        - Drag & drop PNG/JPG files or click “Browse files”  
        - Adjust the decision threshold to tune sensitivity  
        - Download a CSV of your batch results for record-keeping
        """
    )
    st.markdown("---")

    # Upload
    uploaded = st.file_uploader(
        "Upload leaf images", type=["png","jpg","jpeg"], accept_multiple_files=True
    )
    threshold = st.slider(
        "Decision threshold for mildew class", 0.0, 1.0, 0.5, 0.01
    )

    if uploaded:
        model = get_model()
        class_idx = load_class_indices()
        shape = load_image_shape()

        results = []
        for file in uploaded:
            # Load & preprocess
            img = Image.open(file).convert("RGB").resize(shape[:2])
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, 0)

            # Predict probability of mildew
            prob = model.predict(arr, verbose=0)[0][0]

            # Determine label
            label = "powdery_mildew" if prob >= threshold else "healthy"

            results.append({
                "filename": file.name,
                "predicted": label,
                "confidence": float(prob)
            })

            # Display image
            st.image(img, caption=file.name, use_container_width=True)

            # DEBUG: show raw mildew probability
            st.write("Raw model output (probability of mildew):", prob)

            # choose which probability to show on the gauge
            if label == "powdery_mildew":
                display_prob = prob
            else:
                display_prob = 1 - prob

            pct = display_prob * 100  # scale to 0–100

            # render gauge with confidence level
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={
                    'suffix': '%',
                    'valueformat': '.0f'
                },
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickformat': '.0f'
                    }
                },
                title={'text': f"Confidence: {label.replace('_',' ').title()}"}
            ))

            # Bug fix: gives each chart a unique key to avoid duplicate-element errors
            chart_key = f"gauge_{file.name}"

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=chart_key
            )
            st.markdown("---")

        # Batch results table & download
        df = pd.DataFrame(results)
        st.subheader("Batch Results")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="mildew_predictions.csv",
            mime="text/csv"
        )
