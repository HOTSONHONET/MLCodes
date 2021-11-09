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

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# For building models
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Tensorflow modules
import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *


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


def rmse_score(y_label, y_preds):
    """
    Gives RMSE score
    """
    return np.sqrt(mean_squared_error(y_label, y_preds))


def rmse_tf(y_label, y_preds):
    """
    Gives RMSE score, useful for NN training

    """
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_label, y_preds))))


def trainRegModels(df: "data_file", features: list, label: str):
    """
    To automate the training of regression models. Considering
        > RMSE
        > R2 score

    """
    regModels = {
        "LinearRegression": LinearRegression(),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=2),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100),
        "LGBMRegressor": LGBMRegressor(),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(random_state=0),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_jobs=-1),
        "RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
        "XGBRegressor": XGBRegressor(n_jobs=-1),
        "CatBoostRegressor": CatBoostRegressor(iterations=900, depth=5, learning_rate=0.05, loss_function='RMSE'),
    }

    # Will return this as a data frame
    summary = {
        "Model": [],
        "Avg R2 Train Score": [],
        "Avg R2 Val Score": [],
        "Avg RSME Train Score": [],
        "Avg RSME Val Score": []
    }

    # Training
    for idx in trange(len(regModels.keys()), desc="Models are training...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position=0, leave=True):
        name = list(regModels.keys())[idx]
        model = regModels[name]

        # Initializing all the scores to 0
        r2_train = 0
        r2_val = 0
        rmse_train = 0
        rmse_val = 0

        # Running K-fold Cross-validation on every model
        for fold in range(5):
            train_df = df.loc[df.kfold != fold].reset_index(drop=True)
            val_df = df.loc[df.kfold == fold].reset_index(drop=True)

            train_X = train_df[features]
            train_Y = train_df[label]
            val_X = val_df[features]
            val_Y = val_df[label]

            cur_model = model
            if name == 'CatBoostRegressor':
                cur_model.fit(train_X, train_Y, verbose=False)
            else:
                cur_model.fit(train_X, train_Y)

            Y_train_preds = model.predict(train_X)
            Y_val_preds = model.predict(val_X)

            # Collecting the scores
            r2_train += r2_score(train_Y, Y_train_preds)
            r2_val += r2_score(val_Y, Y_val_preds)

            rmse_train += rmse_score(train_Y, Y_train_preds)
            rmse_val += rmse_score(val_Y, Y_val_preds)

        # Pushing the scores and the Model names
        summary["Model"].append(name)
        summary["Avg R2 Train Score"].append(r2_train/5)
        summary["Avg R2 Val Score"].append(r2_val/5)
        summary["Avg RSME Train Score"].append(rmse_train/5)
        summary["Avg RSME Val Score"].append(rmse_val/5)

    # Finally returning the summary dictionary as a dataframe
    summary_df = pd.DataFrame(summary)
    return summary_df
