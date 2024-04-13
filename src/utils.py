from typing import Dict, List, Union
import os
import json
import openpyxl
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import random
from gluonts.time_feature import time_features_from_frequency_str
from src.config.mainconf import Config


class FileIO:
    @staticmethod
    def read_text(filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def write_text(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write(data)

    @staticmethod
    def append_text(data: str, filename):
        with open(filename, "a", encoding="utf8") as f:
            f.write("\n" + data)

    @staticmethod
    def read_json(filename):
        with open(filename, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def write_json(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f, default=str)

    @staticmethod
    def read_excel(filename, sheet_name="Sheet1"):
        wb_obj = openpyxl.load_workbook(filename)
        return wb_obj[sheet_name]

    @staticmethod
    def read_csv(filename: str) -> List[Dict]:
        with open(filename, "r", encoding="utf8") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            return [
                {h: x for h, x in zip(header, row) if h}
                for row in csvreader
            ]

    @staticmethod
    def write_numpy(filename: str, array: np.array) -> None:
        with open(filename, 'wb') as f:
            np.save(f, array)

    @staticmethod
    def read_numpy(filename: str) -> None:
        with open(filename, 'rb') as f:
            return np.load(f)


def standardize(x: Tensor, targ_ids: List[int] = None) -> Tensor:
    '''Standardizes a tensor along the dimension 'dim'.
    Only standardize the first 'n_feat' features while leave the rest unchanged.'''
    if targ_ids is not None:
        if isinstance(targ_ids, int):
            targ_ids = [targ_ids]
        mean = x[:, :, targ_ids].mean(1, keepdim=True)
        std = x[:, :, targ_ids].std(1, keepdim=True) + 1e-9
        x_standard = x
        x_standard[:, :, targ_ids] = (x[:, :, targ_ids] - mean) / std
    else:
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True) + 1e-9
        x_standard = (x - mean) / std
    return x_standard, mean.squeeze(1), std.squeeze(1)


def differentiate(seq: Tensor, targ_ids: list[int] = None) -> Tensor:
    '''Differentiate each mutli-variate time-series. dims = (batch, seq, n_feat)'''
    if targ_ids is not None:
        if isinstance(targ_ids, int):
            targ_ids = [targ_ids]
        batch_size, seq_len, n_feat = seq[:, :, targ_ids].shape
        seq[:, :, targ_ids] = torch.stack([
            seq[:, i, targ_ids] - seq[:, i - 1, targ_ids]
            for i in range(seq_len)
        ], dim=1)
        seq[:, 0, targ_ids] = torch.zeros(n_feat)
    else:
        batch_size, seq_len, n_feat = seq.shape
        seq = torch.stack([seq[:, i] - seq[:, i - 1] for i in range(seq_len)], dim=1)
        seq[:, 0] = torch.zeros(n_feat)
    return seq


def expand_time_range(starting_date: Union[int, List[int]], n_days_context: int,
                      n_days_pred: int) -> List[int]:
    '''Expands a list of starting dates (in the ordinal format: int) to a list of
    n_days dates (string YYYY-MM-DD).

    Parameters
    ----------
    starting_date: List[int] - Batch of starting dates in the ordinal format.
    n_days_context: int - Number of days in the context.
    n_days_pred: int - Number of days in the prediction.
    '''
    if isinstance(starting_date, int):
        starting_date = [starting_date]
    datetimes_past = [
        pd.date_range(datetime.fromordinal(day), periods=n_days_context, freq='D')
        for day in starting_date
    ]
    datetimes_future = [
        pd.date_range(x[-1], periods=n_days_pred + 1, freq='D')[1:]
        for x in datetimes_past
    ]
    # Convert to np.array.
    datetimes_past = np.array([np.array(x) for x in datetimes_past])
    datetimes_future = np.array([np.array(x) for x in datetimes_future])
    return datetimes_past, datetimes_future


def create_time_features(dates_batch: list[list[np.datetime64]],
                         freq: str = '1D') -> Tensor:
    '''Creates time features from dates. (batch, seq)'''
    # Add scaled day_of_week, day_of_monthm, ...
    batch_size = len(dates_batch)
    dates_batch = [pd.to_datetime(dates) for dates in dates_batch]
    time_features_trans = time_features_from_frequency_str(freq)
    time_features = Tensor(np.array(
        [[trans(dates) for dates in dates_batch] for trans in time_features_trans]
    )).permute(1, 2, 0)
    # Add age feature.
    age_feature = torch.log10(2.0 + torch.arange(time_features.shape[1])).view(1, -1, 1).repeat(batch_size, 1, 1)
    time_features = torch.cat([time_features, age_feature], dim=-1)
    return time_features


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def prepare_savedir(config: Config, dataset: str) -> str:
    name = '{}_emb{}-ff{}-nhead{}-depth{}-seqlen{}-batch{}_stnrd{}_diff{}'.format(
        config.model.attention_type, config.model.emb_size,
        config.model.dim_feedforward, config.model.n_att_heads,
        config.model.depth, config.model.max_seq_len, config.training.batch_size,
        config.training.standardize, config.training.differentiate
    )
    savedir = f'{config.outputs_folder}/{dataset}/{name}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    return savedir
