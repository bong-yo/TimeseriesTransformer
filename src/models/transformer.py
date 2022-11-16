from typing import Optional
import numpy as np
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from src.utils import FileIO


class PositionalEncoding1D(nn.Module):
    def __init__(self, size: int) -> None:
        """
        :param size: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_size = size
        size = int(np.ceil(size / 2) * 2)
        self.size = size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, size, 2).float() / size))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, seq_len, ch)
        :return: Positional Encoding Matrix of size (batch_size, seq_len, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, seq_len, orig_size = tensor.shape
        pos_x = torch.arange(seq_len, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((seq_len, self.size), device=tensor.device).type(tensor.type())
        emb[:, : self.size] = emb_x

        self.cached_penc = emb[None, :, :orig_size].repeat(batch_size, 1, 1)
        return self.cached_penc


class TimeFormer(nn.Module):
    def __init__(self,
                 input_size: int,
                 emb_size: int,
                 dim_feedforward: int,
                 n_att_heads: int,
                 depth: int,
                 dropout: float,
                 max_seq_len: int,
                 out_size: int) -> None:
        super(TimeFormer, self).__init__()
        assert emb_size % 2 == 0
        # Model params.
        self.input_size = input_size
        self.emb_size = emb_size
        self.dim_feedforward = dim_feedforward
        self.n_att_heads = n_att_heads
        self.depth = depth
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.out_size = out_size
        self.cached_mask = None
        # Learnable modules.
        self.f = nn.GELU()
        self.drop = nn.Dropout(dropout, inplace=False)
        self.inp = nn.Linear(input_size, emb_size // 2)
        self.transblock = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_att_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.pe = PositionalEncoding1D(emb_size)
        self.encoder = TransformerEncoder(
            encoder_layer=self.transblock,
            num_layers=depth, 
            norm=None
        )
        self.out = nn.Linear(emb_size, out_size)

    def forward(self, input_seq: Tensor, pad_mask: Optional[LongTensor] = None):
        """
        :param input_seq: 3d tensor of size (batch_size, seq_len, input_size)
        :param pad_mask: 2d tensor with 0 on padding (batch_.size, seq_len)
        :return: Transformer embeddings sequence of size (batch_size, seq_len, emb_size)
        """
        # Compute diagonal attention mask to prevent tokens to be bale to attend future 
        # tokens in the time-series
        seq_len = input_seq.shape[1]
        diag_mask = self._generate_triangle_mask(seq_len)  # ATTENTION: 1 means the token WILL be masked, 0 means it WILL NOT.
        h = self.f(self.inp(input_seq)) # Create initial embeddigs.
        h = torch.cat((h, self.pe(h)), dim=-1)  # Add positional embeddings.
        # h = h + self.pe(h)
        h = self.encoder(h, diag_mask)  # Run transformer encoder.
        return self.out(h)

    def _generate_triangle_mask(self, sz: int) -> Tensor:
        """
        :param sz: int - length of the sequence
        :return: 2d triangle tensor with -inf on the upper triangle and 0 on the diag and 
        lower triangle
        """
        if self.cached_mask is None:
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            self.cached_mask = mask.masked_fill(mask == 1, float('-inf'))
        return self.cached_mask

    def save(self, dirname: str) -> None:
        torch.save(self, f'{dirname}/model.pth')

    @classmethod
    def load(cls, dirname: str) -> None:
        model = torch.load(f'{dirname}/model.pth')
        model.eval()
        return model
