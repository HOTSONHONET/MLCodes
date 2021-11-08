"""
Importing Libraries

"""


# Standard imports
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from colorama import Fore
from glob import glob
import json
from pprint import pprint
import time
import cv2
from enum import Enum
from IPython.display import display

# For Data preparation
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *

# For building models
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# For Transformer
import transformers
from transformers import AutoTokenizer, BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


import warnings
warnings.filterwarnings("ignore")


def giveHistogram(df: "data File", col_name: str, bins=None, dark=False):
    """
    To create histogram plots

    """
    fig = px.histogram(df, x=col_name, template="plotly_dark" if dark else "ggplot2",
                       nbins=bins if bins != None else 1 + int(np.log2(len(df))))
    fig.update_layout(
        title_text=f"Distribution of {col_name}",
        title_x=0.5,
    )
    fig.show()


def widthAndHeightDist(df: "data_file", col_name: "col name that contains the img path", dark=False):
    """
    Give Histogram distribution of image width and height

    """
    widths = []
    heights = []
    bins = 1 + int(np.log2(len(df)))
    total_images = list(df[col_name].values)
    for idx in trange(len(total_images), desc="Collecting widths and heights...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position=0, leave=True):
        cur_path = total_images[idx]
        h, w, _ = cv2.imread(cur_path).shape
        widths.append(w)
        heights.append(h)

    figW = px.histogram(widths, nbins=bins,
                        template="plotly_dark" if dark else "ggplot2")
    figW.update_layout(title='Distribution of Image Widths', title_x=0.5)
    figW.show()

    figH = px.histogram(heights, nbins=bins,
                        template="plotly_dark" if dark else "ggplot2")
    figH.update_layout(title='Distribution of Image Heights', title_x=0.5)
    figH.show()


def buildGridImages(df: "data_file", img_path_col_name: str, label_col_name: str, nrows=5, ncols=4, img_size=512):
    """
    To build an image grid
    """

    df = df.sample(nrows*ncols)
    paths = df[img_path_col_name].values
    labels = df[label_col_name].values

    text_color = (255, 255, 255)
    box_color = (0, 0, 0)

    plt.figure(figsize=(20, 12))
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i+1)
        img = cv2.imread(paths[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.axis("off")
        plt.title(str(labels[i]))
        plt.imshow(img)

    plt.tight_layout()
    plt.show()


def create_folds_regression(data, target="target", num_splits=5):
    """
    Helper function to create folds

    """
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)

    # Applying Sturg's rule to calculate the no. of bins for target
    num_bins = int(1 + np.log2(len(data)))

    data.loc[:, "bins"] = pd.cut(data[target], bins=num_bins, labels=False)

    kf = StratifiedKFold(n_splits=num_splits)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop(["bins"], axis=1)
    return data
