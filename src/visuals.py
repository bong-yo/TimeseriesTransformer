import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np


class DisplayPreds:
    @staticmethod
    def plot_PredsVsTrues(dates: np.array, preds: np.array, trues: np.array,
                          title: str) -> None:
        plt.title(title)
        x = range(len(dates))
        plt.plot(x, trues, label='trues')
        plt.plot(x, preds, label='preds')
        plt.xticks(x[::7], dates[::7], rotation=45)
        plt.legend()
        plt.show()

    # @staticmethod
    # def plot_AttentionWeights(att_weights: list[Tensor], title: str,
    #                           savefile: str) -> None:
    #     '''att_weights: list of each encoder block attention weights
    #     of shape (batch_size, n_heads, seq_len, seq_len)'''
    #     plt.title(title)
    #     batch, n_heads, seq_len, seq_len = att_weights[0].shape
    #     attn_weights = torch.stack(att_weights).reshape(-1, seq_len, seq_len)
    #     attn_weights = attn_weights.mean(dim=0).squeeze().cpu().numpy()
    #     # Plot heatmap of attention weights.
    #     plt.imshow(attn_weights, cmap='hot', interpolation='nearest')
    #     plt.colorbar()
    #     plt.show()
    #     plt.savefig(savefile)

    @staticmethod
    def plot_AttentionWeights(att_weights: list[Tensor],
                              att_vv: list[Tensor],
                              att_vt: list[Tensor], savefile: str) -> None:
        '''att_weights: list of each encoder block attention weights
        of shape (batch_size, n_heads, seq_len, seq_len)'''
        # Averag
        batch, seq_len, seq_len = att_weights[0].shape
        attn_weights = torch.stack(att_weights).reshape(-1, seq_len, seq_len)
        att_vv = torch.stack(att_vv).reshape(-1, seq_len, seq_len)
        att_vt = torch.stack(att_vt).reshape(-1, seq_len, seq_len)
        attn_weights = attn_weights.mean(dim=0).squeeze().cpu().numpy()
        att_vv = att_vv.mean(dim=0).squeeze().cpu().numpy()
        att_vt = att_vt.mean(dim=0).squeeze().cpu().numpy()
        # Find min and max value of att_vv and att_vt
        vmin = min(att_vv.min(), att_vt.min())
        vmax = max(att_vv.max(), att_vt.max())
        # Plot heatmap of three attention weights.
        fig, axs = plt.subplots(3, 1, figsize=(5, 15))
        fig.suptitle('Cross-Attention Weights')
        im0 = axs[0].imshow(att_vv, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        axs[0].autoscale(False)
        axs[0].set_title('Value-Value attention weights')
        fig.colorbar(im0, ax=axs[0], shrink=0.805)
        im1 = axs[1].imshow(att_vt, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        axs[1].autoscale(False)
        axs[1].set_title('Value-Temporal attention weights')
        fig.colorbar(im1, ax=axs[1], shrink=0.805)
        im2 = axs[2].imshow(attn_weights, cmap='hot', interpolation='nearest')
        axs[2].set_title('Total attention weights')
        fig.colorbar(im2, ax=axs[2], shrink=0.805)
        plt.tight_layout()
        plt.show()
        fig.savefig(savefile)
