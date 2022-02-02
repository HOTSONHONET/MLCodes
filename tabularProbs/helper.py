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
import plotly.figure_factory as ff

import seaborn as sns
from tqdm import trange
from colorama import Fore
from glob import glob
import json
from pprint import pprint
import time
import cv2
from enum import Enum
from IPython.display import display, HTML
from pandas_profiling import ProfileReport
import random
import inspect

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


# Classification Models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


import warnings
warnings.filterwarnings("ignore")


class Config(Enum):
    '''
    It basically contains all the path location and other stuffs

    '''

    def __str__(self):
        return self.value

    TRAIN_CSV = ""
    TEST_CSV = ""
    SAMPLE_CSV = ""
    TRAIN_DIR = ""
    TEST_DIR = ""


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


    """
    Helper function to show summary of dataFrame in an interactive manner

    """
    data_df_report = ProfileReport(data_df)
    data_df_report.to_file(output_file=fileName)
    HTML(filename=fileName)


def plotCorrelation(df: "dataFrame"):
    """

    Helper function to plot correlation plot

    """
    data = [
        go.Heatmap(
            z=df.corr().values,
            x=df.columns.values,
            y=df.columns.values,
            colorscale='Rainbow',
            reversescale=False,
            #                 text = True,
            opacity=1.0)
    ]

    layout = go.Layout(
        title='Pearson Correlation plot',
        title_x=0.5,
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=900, height=700)

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_scatterMatrix(df: "dataframe", cols: list):
    """
    Helper function to plot scatter matrix
    """
    data_matrix = df.loc[:, cols]
    data_matrix["index"] = np.arange(1, len(data_matrix)+1)
    # scatter matrix
    fig = ff.create_scatterplotmatrix(data_matrix, diag='box', index='index', colormap='Portland',
                                      colormap_type='cat',
                                      height=700, width=700)
    fig.show()


def create_folds(data, target="label", regression=True, num_splits=5):
    """
    Helper function to create folds

    """
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    kf = StratifiedKFold(n_splits=num_splits)

    if regression:
        # Applying Sturg's rule to calculate the no. of bins for target
        num_bins = int(1 + np.log2(len(data)))

        data.loc[:, "bins"] = pd.cut(data[target], bins=num_bins, labels=False)
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'kfold'] = f
        data = data.drop(["bins"], axis=1)
    else:
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data[target].values)):
            data.loc[v_, 'kfold'] = f

    return data


def rmse_score(y_label, y_preds):
    """
    Gives RMSE score
    """
    return np.sqrt(mean_squared_error(y_label, y_preds))


def trainRegModels(df: "data_file", useStandardization: bool, features: list, label: str, sortByRMSE = True):
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
    folds = 1 + max(df.kfold.values)
    for idx in trange(len(regModels.keys()), desc=f"Models are training, LABEL: {label}...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position=0, leave=True):
        name = list(regModels.keys())[idx]
        model = regModels[name]

        # Initializing all the scores to 0
        r2_train = 0
        r2_val = 0
        rmse_train = 0
        rmse_val = 0

        # Running K-fold Cross-validation on every model
        for fold in range(folds):
            train_df = df.loc[df.kfold != fold].reset_index(drop=True)
            val_df = df.loc[df.kfold == fold].reset_index(drop=True)

            train_X = train_df[features]
            train_Y = train_df[label]
            val_X = val_df[features]
            val_Y = val_df[label]
            
            if useStandardization:
                ss = StandardScaler()
                ss.fit_transform(train_X)
                ss.transform(val_X)
                
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
        summary["Avg R2 Train Score"].append(r2_train/folds)
        summary["Avg R2 Val Score"].append(r2_val/folds)
        summary["Avg RSME Train Score"].append(rmse_train/folds)
        summary["Avg RSME Val Score"].append(rmse_val/folds)

    # Finally returning the summary dictionary as a dataframe
    summary_df = pd.DataFrame(summary)
    if sortByRMSE:
        summary_df = summary_df.sort_values(["Avg RSME Val Score", "Avg R2 Val Score"], ascending = True)
    else:
        summary_df = summary_df.sort_values(["Avg R2 Val Score", "Avg RSME Val Score", ], ascending = True)
    return summary_df


def saveModelsKaggle(dir_name: str, title: "title of dataset", token_path="../input/kaggletoken/kaggle.json"):
    """
     > Helper function to automate the process of saving models 
        as kaggle datasets using kaggle API   
     > dir_name should be compatible with hyperlink formats
     > Internet should be enabled

    """
    if not os.path.exists(token_path):
        print("Token doesn't exist")
        return

    if not os.path.exists(f"./{dir_name}"):
        print("Directory doesn't exist")
        return

    os.system(
        f"""
        
        pip install kaggle
        cp {token_path} ./
        cp ./kaggle.json ../../root/
        mkdir ../../root/.kaggle
        mv ../../root/kaggle.json ../../root/.kaggle/kaggle.json

        chmod 600 /root/.kaggle/kaggle.json
        kaggle datasets init -p ./{dir_name}
        
        """
    )
    # Upto this we will be having a meta data file in the form of a json
    with open(f"./{dir_name}/dataset-metadata.json", 'r+') as file_:
        meta_data = json.load(file_)
        meta_data['title'] = f'{title}'
        meta_data['id'] = f'hotsonhonet/{title}'
        file_.seek(0)
        json.dump(meta_data, file_, indent=4)
        file_.truncate()

    os.system(f"""
        kaggle datasets create -p ./{dir_name} --dir-mode zip
    """)

    print("[INFO] Dataset saved successfully")


def trainClfModels(df: "data_file", features: list, label: str):
    """
    To automate the training of regression models. Considering
        > Accuracy
        > Precision
        > Recall
        > F1-score

    """
    clfModels = {
        "LogisticRegression": LogisticRegression(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=2),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=0, n_estimators=100),
        "LGBMClassifier": LGBMClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=0),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_jobs=-1),
        "RandomForestClassifier": RandomForestClassifier(n_jobs=-1),
        "XGBClassifier": XGBClassifier(n_jobs=-1),
        "CatBoostClassifier": CatBoostClassifier(iterations=900, depth=5, learning_rate=0.05, loss_function='CrossEntropy'),
    }

    # Will return this as a data frame
    summary = {
        "Model": [],
        "Avg Accuracy Train Score": [],
        "Avg Accuracy Val Score": [],
        "Avg Precision Train Score": [],
        "Avg Precision Val Score": [],
        "Avg Recall Train Score": [],
        "Avg Recall Val Score": [],
        "Avg F1-score Train Score": [],
        "Avg F1-score Val Score": []
    }

    # Training
    folds = 1 + max(df.kfold.values)
    for idx in trange(len(clfModels.keys()), desc="Models are training...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position=0, leave=True):
        name = list(clfModels.keys())[idx]
        model = clfModels[name]

        # Initializing all the scores to 0
        accuracy_train = 0
        accuracy_val = 0
        recall_train = 0
        recall_val = 0
        precision_train = 0
        precision_val = 0
        f1_train = 0
        f1_val = 0

        # Running K-fold Cross-validation on every model
        for fold in range(folds):
            train_df = df.loc[df.kfold != fold].reset_index(drop=True)
            val_df = df.loc[df.kfold == fold].reset_index(drop=True)

            train_X = train_df[features]
            train_Y = train_df[label]
            val_X = val_df[features]
            val_Y = val_df[label]

            cur_model = model
            if name == 'CatBoostClassifier':
                cur_model.fit(train_X, train_Y, verbose=False)
            else:
                cur_model.fit(train_X, train_Y)

            Y_train_preds = model.predict(train_X)
            Y_val_preds = model.predict(val_X)

            # Collecting the scores
            accuracy_train += accuracy_score(train_Y, Y_train_preds)
            accuracy_val += accuracy_score(val_Y, Y_val_preds)

            recall_train += recall_score(train_Y,
                                         Y_train_preds, average='binary')
            recall_val += recall_score(val_Y, Y_val_preds, average='binary')

            precision_train += precision_score(train_Y,
                                               Y_train_preds, average='binary')
            precision_val += precision_score(val_Y,
                                             Y_val_preds, average='binary')

            f1_train += f1_score(train_Y, Y_train_preds, average='binary')
            f1_val += f1_score(val_Y, Y_val_preds, average='binary')

        # Pushing the scores and the Model names
        summary["Model"].append(name)
        summary["Avg Accuracy Train Score"].append(accuracy_train/folds)
        summary["Avg Accuracy Val Score"].append(accuracy_val/folds)

        summary["Avg Recall Train Score"].append(recall_train/folds)
        summary["Avg Recall Val Score"].append(recall_val/folds)

        summary["Avg Precision Train Score"].append(precision_train/folds)
        summary["Avg Precision Val Score"].append(precision_val/folds)

        summary["Avg F1-score Train Score"].append(f1_train/folds)
        summary["Avg F1-score Val Score"].append(f1_val/folds)

    # Finally returning the summary dictionary as a dataframe
    summary_df = pd.DataFrame(summary)
    return summary_df
