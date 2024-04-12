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
        # Xonvert to datetime.
        data['DATE'] = pd.to_datetime(data['DATE'])
        data.sort_values('DATE', inplace=True)
        y = data['IPG2211A2N'].values
        dates = data['DATE'].values
        # Divide into train, dev and test.
        n = len(y)
        tr, d = int(n * train_perc), int(n * dev_perc)
        train = {'dates': dates[: tr], 'values': y[: tr]}
        dev = {'dates': dates[tr: tr + d], 'values': y[tr: tr + d]}
        test = {'dates': dates[tr + d:], 'values': y[tr + d:]}
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
        train, dev, test = y[: tr], y[tr: tr + d], y[tr + d:]
        return train, dev, test


class DailyDehliClimate:
    def __init__(self) -> None:
        self.X = 'date'
        self.Y = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

    @staticmethod
    def load(filename: str, train_perc: float) -> np.ndarray:
        def _remove_outlayers_pressure(y: np.ndarray) -> np.ndarray:
            """
            Remove outlayers from for the MEANPRESSURE, by setting them to the avg
            of prev and next value.
            """
            mask = (y[:, 3] < 990) | (y[:, 3] > 1022)
            ids = np.where(mask)[0]
            y[ids, 3] = (y[ids - 1, 3] + y[ids + 1, 3]) / 2
            # If first value is outlayer, take the avg of the next two
            # (otherwise there is the id=-1 problem).
            if 0 in ids:
                y[0, 3] = (y[1, 3] + y[2, 3]) / 2
            return y

        Y = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
        # Load train and dev data.
        data = pd.read_csv(f'{filename}Train.csv')
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values('date', inplace=True)
        y = data[Y].values.astype(np.float32)
        dates = data['date'].values
        y = _remove_outlayers_pressure(y)

        n_tain = int(train_perc * len(y))
        train = {'dates': dates[: n_tain], 'values': y[: n_tain]}
        dev = {'dates': dates[n_tain:], 'values': y[n_tain:]}

        # Load test and dev data.
        data = pd.read_csv(f'{filename}Test.csv')
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values('date', inplace=True)
        y = data[Y].values.astype(np.float32)
        y = _remove_outlayers_pressure(y)
        dates = data['date'].values
        test = {'dates': dates, 'values': y}
        return train, dev, test
