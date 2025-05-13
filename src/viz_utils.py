import numpy as np
from matplotlib.image import imread

def compute_stats_for_class(class_dir):
    imgs = []
    for img_path in (class_dir).glob('*'):
        img = imread(str(img_path)).astype(np.float32) / 255.0
        imgs.append(img)
    stack = np.stack(imgs, axis=0)  # shape = (Num, Hgt, Wdth, Chnl)
    mean_img = np.mean(stack, axis=0)
    std_img  = np.std(stack, axis=0)
    return mean_img, std_img
