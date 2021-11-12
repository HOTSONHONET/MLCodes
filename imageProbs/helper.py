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
# To ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(
    f"GPU is available : {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}")


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


def setSeed(seed):
    """
    Setting the seed of all the random function to maintain reproducibility

    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)
    tf.random.set_seed(seed)
    print('SEEDITIZATION DONE !')


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


class ImgDataLoader:
    """
    Gives img data in the form of batches

    """

    def __init__(self,
                 df: "Data_File",
                 path_col: list,
                 target_col: str,
                 regression_type=True,
                 rescale=False,
                 batch_size=32,
                 img_shape=224,
                 resize_with_pad=False,
                 do_augment=False,
                 repeat=False,
                 shuffle=False
                 ):

        self.df = df
        self.path_col = path_col
        self.target_col = target_col
        self.regression_type = regression_type
        self.rescale = rescale
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.resize_with_pad = resize_with_pad
        self.do_augment = do_augment
        self.repeat = repeat
        self.shuffle = shuffle

    @tf.function
    def doAugment(self, img: "Tensor"):
        """
        Perform augmentation over the image tensor
        """
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.95, 1.05)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.95, 1.05)
        img = tf.image.random_hue(img, 0.05)

        return img

    @tf.function
    def process_img(self, path: str, label=None):
        """
        A function to apply augmentation and process the images

        """
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)

        if self.rescale:
            img = img/255.0

        if self.resize_with_pad:
            img = tf.image.resize_with_pad(img, self.img_shape, self.img_shape)
        else:
            img = tf.image.resize(img, (self.img_shape, self.img_shape))

        if self.do_augment:
            img = self.doAugment(img)

        if label is not None:
            return img, label

        return img

    def __call__(self):
        if self.target_col is not None:
            data_gen = tf.data.Dataset.from_tensor_slices(
                (self.df[self.path_col].values, self.df[self.target_col].values))
        else:
            data_gen = tf.data.Dataset.from_tensor_slices(
                (self.df[self.path_col].values))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        data_gen = data_gen.map(self.process_img, num_parallel_calls=AUTOTUNE)
        if self.repeat:
            data_gen = data_gen.repeat()

        if self.shuffle:
            data_gen = data_gen.shuffle(1024, reshuffle_each_iteration=True)

        return data_gen.batch(self.batch_size).prefetch(AUTOTUNE)


def trainEngine(tf_model: "tf compiled model", data_df: "cv dataframe"):
    """
    It will take the model and will perfrom the full k-folds training
        > model : can be a class with __call__ method or a function

    """

    def givePlotsInOne(training_summary: dict, useDark=False, title="Plot"):
        """
        Helper function to plot the training result
        """

        fig = go.Figure()
        for k in summary.keys():
            if(k != "epochs"):
                fig.add_trace(go.Scatter(x=summary["epochs"], y=summary[k],
                                         mode='lines+markers',
                                         name=k))

                fig.update_layout(
                    title_text=title,
                    title_x=.5,
                    xaxis_title="Epochs",
                    yaxis_title="Values",
                    template="plotly_dark" if useDark else "ggplot2"
                )

        fig.show()

    def train_model(tf_model):
        model = tf_model()

        K.clear_session()
        LEARNING_RATE = 1e-2
        DECAY_STEPS = 100
        DECAY_RATE = 0.99

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE,
            staircase=True
        )

        # Creating an early stopper
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        training_history = model.fit_generator(
            train_data_gen,
            validation_data=val_data_gen,
            epochs=Config.EPOCHS.value,
            verbose=1,
            use_multiprocessing=True,
            workers=-1,
        )
        return training_history

    folds = max(data_df['kfold'])
    for fold in range(folds):
        print(Fore.GREEN)
        print("_ "*20, "\n")
        print(f"{' '*11}Current Fold : {fold + 1}")
        print("_ "*20, "\n")

        train_data = data_df.loc[data_df.kfold != fold]
        val_data = data_df.loc[data_df.kfold == fold]

        train_data_gen = ImgDataLoader(
            train_data,
            "path",
            "Pawpularity",
            rescale=True,
            img_shape=Config.IMG_SHAPE.value,
            do_augment=True,
            repeat=False,
            shuffle=True
        )()
        val_data_gen = ImgDataLoader(
            val_data,
            "path",
            "Pawpularity",
            rescale=True,
            img_shape=Config.IMG_SHAPE.value,
            do_augment=False,
            repeat=False,
            shuffle=False
        )()

        training_history = train_model(tf_model)

        summary = {
            "epochs": [d for d in range(1, Config.EPOCHS.value + 1)],
            "loss": training_history.history['loss'],
            "val_loss": training_history.history['val_loss'],
            #             "lr" : training_history.history['lr']
        }

        givePlotsInOne(training_summary=summary, useDark=False,
                       title=f"For Fold {fold + 1}")


def loadModels(arch: "function or location of json file", weightFiles: "location of weight files"):
    """
    Helper function to load all the models produced in cross validation training
        > arch : class with __call__ / function() / JSON file path
        > weightFile location : location of the file that contains h5 files
        > return: List of models

    """

    weightFiles = glob(weightFiles + "/*.h5")
    models = []

    if type(arch) == str:
        json_file = open(arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

    for wfile in weightFiles:
        if type(arch) == str:
            model = tf.keras.model.from_json(loaded_model_json)
        else:
            model = arch()

        model.load_weights(wfile)
        models.append(model)

    return models
