import json, pickle, pandas as pd
from pathlib import Path
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def get_model(version="v1"):
    return load_model(Path("models")/"run2_model.h5")

@st.cache_data
def load_metrics(version="v1"):
    return json.loads((Path("outputs")/version/"metrics.json").read_text())

@st.cache_data
def load_image_stats(version="v1"):
    return pd.read_csv(Path("outputs")/version/"image_stats.csv")

@st.cache_data
def load_class_indices(version="v1"):
    return json.loads((Path("outputs")/version/"class_indices.json").read_text())

@st.cache_data
def load_image_shape(version="v1"):
    return pickle.loads((Path("outputs")/version/"image_shape.pkl").read_bytes())

@st.cache_data
def list_image_paths(split="train", version="v1"):
    base = Path("input/datasets/cherry_leaf_dataset/cherry-leaves")/split
    return {p.name:list((base/p.name).glob("*")) for p in base.iterdir() if p.is_dir()}

@st.cache_data
def load_train_paths(version="v1"):
    base = Path("input/datasets/cherry_leaf_dataset/cherry-leaves")/"train"
    paths=[]
    for cls in base.iterdir():
        if cls.is_dir(): paths+=list(cls.glob("*"))
    return paths

@st.cache_data
def load_confusion_matrix_image(version="v1"):
    return Path("outputs")/version/"figures"/"confusion_matrix.png"
