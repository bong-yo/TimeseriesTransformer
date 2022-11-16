import matplotlib.pyplot as plt
import numpy as np


class DisplayPreds:
    @staticmethod
    def plot(inps, preds, trues):
        x = range(len(preds))
        # plt.plot(x, inps, label='inps')
        plt.plot(x, trues, label='trues')
        plt.plot(x, preds, label='preds')
        plt.legend()
        plt.show()
