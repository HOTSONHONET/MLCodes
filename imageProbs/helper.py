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


# For Data preparation
from sklearn.preprocessing import StandardScaler
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
