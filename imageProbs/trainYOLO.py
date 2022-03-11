# Standard imports
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from skimage import io
import ast

import seaborn as sns
from tqdm import trange, tqdm
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


class Config(Enum):
    '''
    It basically contains all the path location and other stuffs
    '''

    def __str__(self):
        return self.value

    TRAIN_CSV = "../input/global-wheat-detection/train.csv"
    TEST_CSV = "../input/global-wheat-detection/sample_submission.csv"
    TRAIN_DIR = "../input/global-wheat-detection/train"
    TEST_DIR = "../input/global-wheat-detection/test"
    OUTPUT_PATH = "./yolov5/output"
    IMG_SHAPE = 1024
    CONFIG_FILENAME = "ws_data"
    EPOCHS = 20
    BATCH_SIZE = 8


def process_data(data_df: "pandas dataFrame", image_id_col: str, bbox_col: str, label_col: str, path_col: str, config_filename="data", test_size=0.1):
    """
    Helper function to build dataset for yolo training
        > Yolo expects the data in the form: (label, x_center, y_center, Width,  Height)
        > return df_train, df_val

    """
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    OUTPUT_FOLDER_NAME = Config.OUTPUT_PATH.value.split("/")[-1]
    if not os.path.exists(Config.OUTPUT_PATH.value):
        os.system(
            f'''
                cd ./yolov5
                mkdir {OUTPUT_FOLDER_NAME} 
                cd {OUTPUT_FOLDER_NAME}
                mkdir images
                mkdir labels
                cd images
                mkdir train
                mkdir validation
                cd ..
                cd labels
                mkdir train
                mkdir validation
                cd ../../
                tree {OUTPUT_FOLDER_NAME}
                cd ../
            ''')

    # For converting string form of list to original form
    data_df.bbox = data_df.bbox.apply(ast.literal_eval)

    # Encoding all labels
    mapper = {k: d for d, k in enumerate(set(data_df[label_col]))}
    data_df[label_col] = data_df[label_col].apply(lambda x: int(mapper[x]))

    # Grouping the bounding boxes wrt image_id, label_col and path_col
    data_df = data_df.groupby(by=[image_id_col, label_col, path_col])[
        bbox_col].apply(list).reset_index(name=bbox_col)

    # Dividing the data into train and val set
    df_train, df_val = train_test_split(data_df,
                                        test_size=test_size,
                                        random_state=42,
                                        shuffle=1
                                        )
    df_train = df_train.reset_index(drop=1)
    df_val = df_val.reset_index(drop=1)

    print(f"[INFO] Train_SHAPE : {df_train.shape}, VAL_SHAPE: {df_val.shape}")

    data_dict = {"train": df_train, "validation": df_val}
    for data_type, data in data_dict.items():
        for idx in trange(len(data), desc=f"Processing {data_type}...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position=0, leave=True):
            row = data.loc[idx]
            image_name = row[image_id_col]
            bounding_boxes = row[bbox_col]
            label = row[label_col]
            path = row[path_col]
            yolo_data = []
            for bbox in bounding_boxes:
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]

                x_center = x + w/2
                y_center = y + h/2

                x_center, y_center, w, h = tuple(
                    map(lambda x: x/Config.IMG_SHAPE.value, (x_center, y_center, w, h)))
                yolo_data.append([label, x_center, y_center, w, h])

            yolo_data = np.array(yolo_data)
            np.savetxt(
                f"{Config.OUTPUT_PATH.value}/labels/{data_type}/{image_name}.txt",
                yolo_data,
                fmt=["%d", "%f", "%f", "%f", "%f"]
            )

            os.system(
                f"""
                cp {path} {Config.OUTPUT_PATH.value}/images/{data_type}/{path.split("/")[-1]}

                """
            )

    with open(f"./yolov5/{config_filename}.yaml", "w+") as file_:
        file_.write(
            f"""
            
            train: {OUTPUT_FOLDER_NAME}/images/train
            val: {OUTPUT_FOLDER_NAME}/images/validation
            nc: {len(mapper)}
            names: {list(mapper.keys())}
            
            """
        )
    file_.close()
    print("[INFO] Done with data processing")


def trainYoloModel(model_name: str, config_filename: str, preTrainedWeights_path=None):
    """
    Helper function to train YOLO v5 models

    """
    mapper = {}
    for idx, model_ in enumerate(glob("yolov5/models/*yaml")):
        mapper[idx + 1] = model_
        print(f"{idx + 1} =>  {model_.split('/')[-1].split('.')[0]}")

    model = mapper[int(input(f"Select the model from the idx: "))]
    if preTrainedWeights_path is not None:
        os.system(
            f"""
                python yolov5/train.py --img {Config.IMG_SHAPE.value} --batch {Config.BATCH_SIZE.value} --epochs {Config.EPOCHS.value} --data yolov5/{config_filename}.yaml --cfg {model} --name {model_name} --weights {preTrainedweights_path}
            
            """
        )
    else:
        os.system(
            f"""
                python yolov5/train.py --img {Config.IMG_SHAPE.value} --batch {Config.BATCH_SIZE.value} --epochs {Config.EPOCHS.value} --data yolov5/{config_filename}.yaml --cfg {model} --name {model_name}
            """
        )

        
  def predict(images_path:"path to the test images", weights_path: "path to the weights folder"):
    """
    Helper function to make predictions over images using Yolo
    """
    os.system(
        f"""
            python yolov5/detect.py --source {images_path} --weights {weights_path}
        """)
