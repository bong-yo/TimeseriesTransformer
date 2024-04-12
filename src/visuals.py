import matplotlib.pyplot as plt
import numpy as np


class DisplayPreds:
    @staticmethod
    def plot(dates: np.array, preds: np.array, trues: np.array,
             title: str) -> None:
        plt.title(title)
        x = range(len(dates))
        plt.plot(x, trues, label='trues')
        plt.plot(x, preds, label='preds')
        plt.xticks(x[::7], dates[::7], rotation=45)
        plt.legend()
        plt.show()
