from typing import Literal
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from src.models.time_trans.layers import (TransformerBlock, TransformerEncoder,
                                          TransformerCrossBlock,
                                          TransformerCrossEncoder)
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class TimeFormer(nn.Module):
    def __init__(self,
                 past_values_size: int,
                 time_feats_size: int,
                 emb_size: int,
                 dim_feedforward: int,
                 n_att_heads: int,
                 depth: int,
                 dropout: float,
                 out_size: int,
                 attention_type: Literal['standard', 'cross']) -> None:
        super(TimeFormer, self).__init__()
        assert emb_size % 2 == 0
        assert attention_type in ['standard', 'cross']
        # Model params.
        self.attention_type = attention_type
        self.past_values_size = past_values_size
        self.time_feats_size = time_feats_size
        self.inp_tot_size = past_values_size + time_feats_size
        self.emb_size = emb_size
        self.dim_feedforward = dim_feedforward
        self.n_att_heads = n_att_heads
        self.depth = depth
        self.dropout = dropout
        self.out_size = out_size
        self.cached_mask = None
        # Constant transformations.
        self.add_pos_embs = Summer(PositionalEncoding1D(emb_size))
        self.f = nn.GELU()
        self.drop = nn.Dropout(dropout, inplace=False)
        # Learnable transforms.
        self.inp = nn.Linear(self.inp_tot_size, emb_size)
        self.inp_values = nn.Linear(past_values_size, emb_size)
        self.inp_temp = nn.Linear(time_feats_size, emb_size)
        TransBlock = TransformerCrossBlock if attention_type == 'cross' \
            else TransformerBlock
        self.transblock = TransBlock(
            d_model=emb_size,
            nhead=n_att_heads,
            dropout_rate=dropout,
            dim_feedforward=dim_feedforward
        )
        TransEnc = TransformerCrossEncoder if attention_type == 'cross' \
            else TransformerEncoder
        self.encoder = TransEnc(
            encoder_layer=self.transblock,
            num_layers=depth
        )
        self.out = nn.Linear(emb_size, out_size)

    def forward(self, inputs: Tensor, past_temp_features: Tensor):
        """
        :param past_values: tensor (batch_size, seq_len, n_val_feats) of
                            numerical past values to predict
        :param past_temp_features: tensor (batch_size, seq_len, n_temp_feats) of
                                   past temporal features
        :return: Transformer embeddings sequence of size (batch_size, seq_len,
                 emb_size)
        """
        # Compute diagonal attention mask to prevent tokens to be bale to attend future
        # tokens in the time-series
        temporal_mask = self.diag_mask(inputs.shape[1]).to(inputs.device)
        if self.attention_type == 'standard':
            if self.time_feats_size > 0:
                inputs = torch.cat((inputs, past_temp_features), dim=-1)
            h = self.f(self.inp(inputs))  # Create initial embeddigs.
            h = self.add_pos_embs(h)
            h = self.encoder(h, mask=temporal_mask)[0]  # Run transformer encoder.
        elif self.attention_type == 'cross':
            h_values = self.f(self.inp_values(inputs))
            h_values = self.add_pos_embs(h_values)
            h_temp = self.f(self.inp_temp(past_temp_features))
            h_temp = self.add_pos_embs(h_temp)
            h = self.encoder(h_values, h_temp, mask=temporal_mask)[0]
        return self.out(h)

    def diag_mask(self, seq_len: int) -> LongTensor:
        """
        Diagonal mask to prevent tokens to attend future tokens in the sequence.
        NOTE: In the attention module the mask is additive:
        'attention_scores += (mask * -1e9)', so mask == 1 means that the token
        is masked, while mask == 0 means that the token is not masked.
        :param seq_len: int, length of the sequence
        :return: LongTensor, diagonal mask of shape (seq_len, seq_len)
        """
        if self.cached_mask is None or self.cached_mask.shape[0] != seq_len:
            self.cached_mask = torch.triu(torch.ones(seq_len, seq_len))
        return self.cached_mask

    def save(self, dirname: str) -> None:
        torch.save(self, f'{dirname}/model.pth')

    @classmethod
    def load(cls, dirname: str) -> None:
        model = torch.load(f'{dirname}/model.pth')
        model.eval()
        return model
