# Standard imports
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from collections import Counter
import json

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
import emoji
import pickle
import datasets
import re
import gc  # Garbage collector
from tqdm.auto import tqdm
import logging
from time import time

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
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# NLP Tools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, TweetTokenizer

# Tensorflow modules
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow import TensorSpec
from tensorflow.python.framework import dtypes
from tensorflow.keras.callbacks import *
# Transformers
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import TFAutoModelForSequenceClassification


import warnings
warnings.filterwarnings("ignore")


class Config(Enum):
    '''
    It basically contains all the path location and other stuffs

    '''

    def __str__(self):
        return self.value

    TRAIN_CSV = ""
    TEST_CSV = "."
    TEXT_COLS = ['commentText']
    META_COLS = []
    LABEL_COL = []

    AUTOTUNE = tf.data.AUTOTUNE
    EPOCHS = 15
    MAX_LEN = 256
    BATCH_SIZE = 128
    MURIL_PATH = "google/muril-base-cased"
    XLM_PATH = 'roberta-base'
    mBERT_PATH = "bert-base-multilingual-cased"
    RES_PATH = "./Results"


def setSeed(seed=42):
    """
    Setting the seed of all the random function to maintain reproducibility
    Also creates a results folder

    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('[INFO] SEEDITIZATION DONE !')

    if not os.path.exists(Config.RES_PATH.value):
        os.mkdir(Config.RES_PATH.value)
        print(f"[INFO] Created {Config.RES_PATH.value}")


def setStrategy():
    """

    Helper function to set strategy wrt to accelator present
    > Code is from TF official docs
    > Returns strategy or resource allocator for training

    """
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy


def f1_score(y_true, y_pred):
    """
    Custom function to calculate f1_score
    > Modifyable for precision and recall

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def fast_encode(texts, tokenizer, chunk_size=512, maxlen=512):
    """

    Helper function to encode sentences for transformer

    """

    input_ids = []
    tt_ids = []
    at_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size]
        encs = tokenizer(
            text_chunk,
            max_length=Config.MAX_LEN.value,
            padding='max_length',
            truncation=True
        )

        input_ids.extend(encs['input_ids'])
        tt_ids.extend(encs['token_type_ids'])
        at_ids.extend(encs['attention_mask'])

    return {'input_ids': input_ids,
            'token_type_ids': tt_ids,
            'attention_mask': at_ids}


def create_model(transformer_model):
    """
    Custom function for fine tuning tranformer model

    """
    input_id_layer = Input(shape=(Config.MAX_LEN.value,),
                           dtype=tf.int32, name='input_ids')
    attention_mask_layer = Input(
        shape=(Config.MAX_LEN.value,), dtype=tf.int32, name='attention_mask')

    transformer = transformer_model(
        input_ids=input_id_layer, attention_mask=attention_mask_layer)[0]
    transformer_output = transformer[:, 0, :]

    x = Dropout(0.2)(transformer_output)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(
        inputs=[input_id_layer, attention_mask_layer], outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        metrics=['AUC', 'accuracy', f1_score],
        loss='binary_crossentropy'
    )

    return model


@tf.function
def train_prep_function(embeddings, target):
    """

    Helper function for making data loader for transformer
    > Can be modified as per type of dataset
    > Can be modified as per type of Transformer
        >> XLMR takes only input_ids and attention_mask
        >> Other also takes token_type_ids as input
    > Used in runEngine

    """
    input_ids = embeddings['input_ids']
    attention_mask = embeddings['attention_mask']

    target = tf.cast(target, tf.int32)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, target


def runEngine(df: "tokenized-dataset"):
    """
    Helper function to train transformer

    """

    model_training_logs = {}
    folds = max(df['kfold']) + 1
    Fores = [Fore.CYAN, Fore.LIGHTMAGENTA_EX,
             Fore.GREEN, Fore.YELLOW, Fore.RED]
    for fold in range(folds):
        print(Fores[fold % len(Fores)], "$"*20,
              f" Fold#{fold + 1} running ", "$"*20)
        train_set = df.loc[df['kfold'] != fold]
        val_set = df.loc[df['kfold'] == fold]

#         train_set = df.head(2)
#         val_set = df.tail(2)

        train_embeddings = {
            'input_ids': train_set['input_ids'].tolist(),
            "attention_mask": train_set['attention_mask'].tolist()
        }
        val_embeddings = {
            'input_ids': val_set['input_ids'].tolist(),
            "attention_mask": val_set['attention_mask'].tolist()
        }

        y_train = train_set['label'].tolist()
        y_val = val_set['label'].tolist()

        # creating Dataset
        start_data_prep = time()
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_embeddings, y_train))
        train_dataset = (train_dataset.shuffle(1024*2).map(
            train_prep_function,
            num_parallel_calls=Config.AUTOTUNE.value).repeat().batch(Config.BATCH_SIZE.value).prefetch(Config.AUTOTUNE.value)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_embeddings, y_val))
        val_dataset = (val_dataset.map(
            train_prep_function,
            num_parallel_calls=Config.AUTOTUNE.value).batch(Config.BATCH_SIZE.value).prefetch(Config.AUTOTUNE.value)
        )
        end_data_prep = time()
        print(
            f"[INFO] Created Dataset...{round(end_data_prep - start_data_prep, 2)//60} mins")

        # Clearing backend session
        K.clear_session()
        print("[INFO] Backend Cleared...")

        # Model Fitting
        weight_save_path = f"{Config.RES_PATH.value}/weights"
        if not os.path.exists(weight_save_path):
            os.mkdir(weight_save_path)

        model_checkpoint = ModelCheckpoint(f'{weight_save_path}/mBERT_fold{fold+1}.h5',
                                           monitor='val_auc',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='max')

        TRAIN_STEPS = len(train_set)//Config.BATCH_SIZE.value//4
        VAL_STEPS = len(val_set)//Config.BATCH_SIZE.value

#         TRAIN_STEPS,VAL_STEPS = 1,1

        print("[INFO] Cleaning up train_set and val_set")
        del train_set
        del val_set
        gc.collect()

        train_start = time()

        with strategy.scope():
            transformer_model = TFAutoModel.from_pretrained(
                Config.mBERT_PATH.value)
            model = create_model(transformer_model)
        training_logs = model.fit(train_dataset,
                                  steps_per_epoch=TRAIN_STEPS,
                                  validation_data=val_dataset,
                                  validation_steps=VAL_STEPS,
                                  epochs=Config.EPOCHS.value,
                                  callbacks=[model_checkpoint])

        model_training_logs[f"fold#{fold + 1}"] = training_logs.history
        model_training_logs[f"fold#{fold + 1}"]['epochs'] = [
            d for d in range(1, Config.EPOCHS.value + 1)]

        print("[INFO] Cleaning up train_dataset and val_dataset")
        del train_dataset
        del val_dataset
        del model
        gc.collect()
        train_end = time()
        print(
            f"Fold#{fold + 1} training...{round(train_end - train_start, 2)//60} mins")
    print(f"[INFO] Done with Training...")

    print(
        f"[INFO] Saving models stats inside {Config.RES_PATH.value}/training_stats.json")
    with open(f'{Config.RES_PATH.value}/training_stats.json', 'w') as fp:
        json.dump(model_training_logs, fp)

    """
    You can load the stats as 
    with open(f'{Config.RES_PATH.value}/training_stats.json', 'r') as fp:
        data = json.load(fp)
    
    """

    return model_training_logs
