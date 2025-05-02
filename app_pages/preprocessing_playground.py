import streamlit as st
import numpy as np
import random
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

# cache the list of training image paths
@st.cache_data
def load_train_paths(version="v1"):
    data_dir = Path("input/datasets/cherry_leaf_dataset/cherry-leaves") / "train"
    paths = []
    for cls in data_dir.iterdir():
        if cls.is_dir():
            paths += list(cls.glob("*"))
    return paths

def page_preprocessing_playground_body():
    st.title("Preprocessing Playground")

    st.info(
            """
            **What you’re seeing:**  
            This page lets you experiment with our image-augmentation parameters (rotation, zoom, shifts, flips)  
            and see how they affect real cherry-leaf photos. Adjust the controls in the sidebar, then compare the  
            raw image (left) with a single augmented example (right).  
            
            **Why it matters:**  
            Augmentations help the model generalize to new lighting, orientations, and leaf shapes—crucial for  
            robust mildew detection in the field.
            """
        )

    st.markdown("---")

    # Sidebar controls for augmentation intensity
    st.sidebar.header("Augmentation Settings")
    rotation = st.sidebar.slider("Rotation range (°)", 0, 45, 20)
    zoom = st.sidebar.slider("Zoom range", 0.0, 0.5, 0.1, step=0.05)
    width_shift = st.sidebar.slider("Width shift", 0.0, 0.3, 0.1, step=0.05)
    height_shift = st.sidebar.slider("Height shift", 0.0, 0.3, 0.1, step=0.05)
    h_flip = st.sidebar.checkbox("Horizontal flip", True)
    v_flip = st.sidebar.checkbox("Vertical flip", False)

    st.markdown(f"**Rotation:** ±{rotation}° &nbsp;&nbsp; **Zoom:** ±{zoom*100:.0f}%&nbsp;&nbsp;"
                f"**Width shift:** ±{width_shift*100:.0f}%&nbsp;&nbsp;"
                f"**Height shift:** ±{height_shift*100:.0f}%")

    # Build ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=rotation,
        zoom_range=zoom,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        fill_mode='nearest'
    )

    # Pick 3 random images from train set
    paths = load_train_paths()
    samples = random.sample(paths, 3)

    st.subheader("Raw vs. Augmented Samples")
    for img_path in samples:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        arr = img_to_array(img)
        arr = np.expand_dims(arr, 0)  # shape (1,H,W,3)

        # Generate one augmented version
        aug_iter = datagen.flow(arr, batch_size=1)
        aug_img_arr = next(aug_iter)[0]

        # Display side by side
        col1, col2 = st.columns(2)
        col1.image(img, caption=f"Original: {img_path.name}", use_container_width=True)
        col2.image(array_to_img(aug_img_arr), caption="Augmented", use_container_width=True)

    st.markdown("---")
    st.write(
        "Use the sliders and checkboxes in the sidebar to tweak augmentation parameters "
        "and see how they transform raw leaf images in real time."
    )
