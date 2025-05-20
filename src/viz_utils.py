import numpy as np
from matplotlib.image import imread


def compute_stats_for_class(class_dir):
    """
    Compute the per-pixel mean and standard-deviation images for all images in a class folder.

    Args:
        class_dir (Path): Path to the directory containing class images.

    Returns:
        tuple:
            mean_img (ndarray): Array of shape (H, W, C) representing the mean pixel values.
            std_img (ndarray): Array of shape (H, W, C) representing the pixel-wise standard deviation.
    """
    imgs = []
    for img_path in (class_dir).glob('*'):
        img = imread(str(img_path)).astype(np.float32) / 255.0
        imgs.append(img)
    stack = np.stack(imgs, axis=0)  # shape = (Num, Hgt, Wdth, Chnl)
    mean_img = np.mean(stack, axis=0)
    std_img = np.std(stack, axis=0)
    return mean_img, std_img
