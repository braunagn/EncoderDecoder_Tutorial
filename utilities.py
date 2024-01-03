import pandas as pd
import numpy as np
import time

import config
import sentence_prep
import tokenizer_prep
import dataset
import model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split


def read_in_rawdata_and_cleanup():
    df = pd.read_csv(
        f"{config.REPO_DIR}/sentences.tsv", sep="\t", usecols=[1, 3], names=["en", "nl"]
    )
    return sentence_prep.initial_cleanup(df)


def make_train_test_datasets(tokenizer, data):
    """Encodes data with trained tokenizer and prepares
    pytorch Datasets.  Returns 3x Dataloaders (training, train, test)
    """
    nl_encoded = tokenizer.encode_batch(data.nl.values)
    en_encoded = tokenizer.encode_batch(data.en.values)
    # grouped by sentence/sequence length
    ignore_token_ids = [
        tokenizer.token_to_id(x) for x in config.SPECIAL_TOKENS.values()
    ]
    grouped_data = tokenizer_prep.group_sentences(
        nl_encoded, en_encoded, ignore_token_ids
    )

    # model train/test datasets and dataloaders
    X1 = np.array([x[0] for x in grouped_data]).reshape(len(grouped_data), config.T)
    X2 = np.array([x[1] for x in grouped_data]).reshape(len(grouped_data), config.T)
    pad_token_id = tokenizer.token_to_id(config.SPECIAL_TOKENS["PAD_TOKEN"])
    y = np.array([x[1][1:] + [pad_token_id] for x in grouped_data])
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1,
        X2,
        y,
        test_size=config.TEST_SPLIT,  # change as desired
        shuffle=False,  # already shuffled and grouped
    )

    train_data = dataset.LanguageDataset(
        X1_train, X2_train, y_train, pad_token_id=pad_token_id
    )
    test_data = dataset.LanguageDataset(
        X1_test, X2_test, y_test, pad_token_id=pad_token_id
    )
    training_dl = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # keep sequences of the same length together
        drop_last=False,
    )
    # for loss performance over train/test datasets (vs. batch being trained on)
    train_dl = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE_EVAL,
        shuffle=True,  # sample across the dataset, regardless of sequence len
        drop_last=False,
    )
    test_dl = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE_EVAL,
        shuffle=True,
        drop_last=False,
    )

    return training_dl, train_dl, test_dl

def load_model(tokenizer=None):
    if config.LOAD_PATH_TRAINED_MODEL_OBJ is not None:
        # reload an exiting model to continue training
        print(f"loading model from {config.LOAD_PATH_TRAINED_MODEL_OBJ}...")
        return torch.load(config.LOAD_PATH_TRAINED_MODEL_OBJ, map_location=config.DEVICE)
    # or initialize a new model
    return model.LanguageModel(tokenizer).to(config.DEVICE)
