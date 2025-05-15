import json, pickle, pandas as pd
from pathlib import Path
import streamlit as st
from src.model_utils import get_or_download_model

@st.cache_resource
def get_model(version="v1"):
    file_id   = "1ui6d2t-dTrq2kbgF0OFPta7KJc2E3cYE"
    model_path = Path("models") / "run2_model.h5"
    return get_or_download_model(file_id, model_path)

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

@st.cache_data
def load_history(version="v1"):
    path = Path("outputs") / version / "history_run2.json"
    return json.loads(path.read_text())

@st.cache_data
def load_tests(version="v1"):
    path = Path("outputs") / version / "hypothesis_tests.json"
    text = path.read_text()
    return json.loads(text)
