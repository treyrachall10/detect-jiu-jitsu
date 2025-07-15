import re
import pandas as pd
import cv2 as cv
import os
import numpy as np
from pathlib import Path
from typing import Union
from torchvision.transforms import v2
from sklearn.decomposition import PCA
import ast
from sklearn.preprocessing import normalize


# Returns a clean, absolute file path
def normalize_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# Create folders for csv files
def make_output_dirs():
    os.makedirs(r"csv_output/no_features", exist_ok=True)
    os.makedirs(r"csv_output/has_features", exist_ok=True)

# Returns the frame number from file name as an int
def get_frame_num(file):
    return int(re.findall(r'\d+', file)[0]) - 1

# Normalizes and reduces dimensionality to 2D
def reduce_feature_list_normalize(df):
    df['features'] = df['features'].apply(ast.literal_eval)
    x = np.stack(df['features'].to_list())
    x_norm = normalize(x, norm='l2')
    x_2d = PCA(n_components=2).fit_transform(x_norm)
    return x_2d

def print_label_cluster_pairs(member_mask, images, k, color):
        indices = np.where(member_mask)[0]
        for i in indices:
            print(f"filename: {images[i]} --> cluster: {k} --> color: {tuple(color)}")