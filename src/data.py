import pandas as pd
import numpy as np


class ElectricProdDataset():
    def __init__(self) -> None:
        self.X = 'DATE'
        self.Y = 'IPG2211A2N'

    @staticmethod
    def load(filename: str, train_perc: float, dev_perc: float) -> np.ndarray:
        assert train_perc + dev_perc < 1
        data = pd.read_csv(filename)
        data.sort_values('DATE', inplace=True)
        y = data['IPG2211A2N'].values
        # Divide into train, dev and test.
        n = len(y)
        tr, d = int(n * train_perc), int(n * dev_perc)
        train, dev, test = y[: tr], y[tr: tr + d], y[tr + d: ]
        return train, dev, test


class DailyTemp:
    def __init__(self) -> None:
        self.X = 'Date'
        self.Y = 'Daily minimum temperatures'

    @staticmethod
    def load(filename: str, train_perc: float, dev_perc: float) -> np.ndarray:
        assert train_perc + dev_perc < 1
        data = pd.read_csv(filename)
        data.sort_values('Date', inplace=True)
        y = data['Daily minimum temperatures'].values.astype(np.float32)
        # Divide into train, dev and test.
        n = len(y)
        tr, d = int(n * train_perc), int(n * dev_perc)
        train, dev, test = y[: tr], y[tr: tr + d], y[tr + d: ]
        return train, dev, test
