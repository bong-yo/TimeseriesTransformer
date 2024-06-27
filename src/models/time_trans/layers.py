import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import MultiheadAttention
import math
import copy


class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # dimension
        self.wq = nn.Linear(D, D * H)
        self.wk = nn.Linear(D, D * H)
        self.wv = nn.Linear(D, D * H)
        self.dense = nn.Linear(D * H, D)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.wv.weight)
        nn.init.xavier_normal_(self.dense.weight)

    def concat_heads(self, x: Tensor):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))   # (B, S, D*H)
        return x

    def split_heads(self, x: Tensor):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x: Tensor, mask: Tensor):
        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)
        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention)  # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    '''Multi-head cross-attention module between two sequences of
    inputs'''
    def __init__(self, D, H):
        super(MultiHeadCrossAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # dimension
        self.wq1 = nn.Linear(D, D * H)
        self.wk1 = nn.Linear(D, D * H)
        self.wv1 = nn.Linear(D, D * H)
        self.wq12 = nn.Linear(D, D * H)
        self.wk12 = nn.Linear(D, D * H)
        self.wq2 = nn.Linear(D, D * H)
        self.wk2 = nn.Linear(D, D * H)
        self.wv2 = nn.Linear(D, D * H)
        self.dense1 = nn.Linear(D * H, D)
        self.dense2 = nn.Linear(D * H, D)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.wq1.weight)
        nn.init.xavier_normal_(self.wk1.weight)
        nn.init.xavier_normal_(self.wv1.weight)
        nn.init.xavier_normal_(self.wq12.weight)
        nn.init.xavier_normal_(self.wk12.weight)
        nn.init.xavier_normal_(self.wq2.weight)
        nn.init.xavier_normal_(self.wk2.weight)
        nn.init.xavier_normal_(self.wv2.weight)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)

    def concat_heads(self, x: Tensor):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))   # (B, S, D*H)
        return x

    def split_heads(self, x: Tensor):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor
                ) -> tuple[Tensor, Tensor]:
        '''Forward pass of the cross-attention module between two sequences
        of inputs: the main one x1 and the secondary one x2. x2 is only used to 
        support the attention mechanism (only employed as keys),
        not to produce the output.'''
        q1 = self.wq1(x1)  # (B, S, D*H)
        k1 = self.wk1(x1)  # (B, S, D*H)
        v1 = self.wv1(x1)  # (B, S, D*H)
        q12 = self.wq12(x1)  # (B, S, D*H)  x1 is still the query of x2 but with a different projection.
        k12 = self.wk12(x2)  # (B, S, D*H)
        q1 = self.split_heads(q1)  # (B, H, S, D)
        k1 = self.split_heads(k1)  # (B, H, S, D)
        v1 = self.split_heads(v1)  # (B, H, S, D)
        q12 = self.split_heads(q12)  # (B, H, S, D)
        k12 = self.split_heads(k12)  # (B, H, S, D)

        attn_scores1 = torch.abs(torch.matmul(q1, k1.transpose(-1, -2)))  # (B,H,S,S)
        attn_scores12 = torch.abs(torch.matmul(q12, k12.transpose(-1, -2)))  # (B,H,S,S)
        # Combine the two attention scores.
        attn1 = attn_scores1.clone().detach()
        attn12 = attn_scores12.clone().detach()
        attn_scores = attn_scores1  # (B,H,S,S)
        attn_scores /= math.sqrt(self.D)  # Scaled dot product.
        # Add mask to the scaled tensor.
        if mask is not None:
            attn_scores += (mask * -1e9)

        attention_weights = nn.Softmax(dim=-1)(attn_scores)  # (B,H,S,S)
        scaled_attention = torch.matmul(attention_weights, v1)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention)  # (B, S, D*H)
        concat_k12 = self.concat_heads(k12)  # (B, S, D*H)
        output_attn1 = self.dense1(concat_attention)  # (B, S, D)
        output_ff2 = self.dense2(concat_k12)

        return output_attn1, output_ff2, attention_weights, attn1, attn12


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout_rate: float) -> None:
        super(TransformerBlock, self).__init__()
        D = d_model
        H = nhead
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(D, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.mha = MultiHeadAttention(D, H)
        self.activation = nn.GELU()

    def forward(self, x: Tensor, look_ahead_mask: Tensor):
        # Attention block.
        x_att, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        x_att = self.dropout1(x_att)  # (B,S,D)
        x = self.layernorm1(x + x_att)  # (B,S,D)
        # Feed forward block.
        x_ff = self.activation(self.linear1(x))
        x_ff = self.linear2(self.dropout(x_ff))
        x_ff = self.dropout2(x_ff)
        output = self.layernorm2(x + x_ff)  # (B, S, D)
        return output, attn_weights


class TransformerEncoder(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, encoder_layer: TransformerBlock, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor):
        attn_weights = []
        for layer in self.encoder_layers:
            x, attn_w = layer(x, mask)
            attn_weights.append(attn_w)
        return x, attn_weights


class TransformerCrossBlock_from_scratch(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout_rate: float) -> None:
        '''Uses MultiheadAttention implemented from scratch.'''
        super(TransformerCrossBlock_from_scratch, self).__init__()
        D = d_model
        H = nhead
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(D, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.mhca = MultiheadAttention(D, H)
        self.activation = nn.GELU()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.mhca.init_weights()

    def forward(self, x: Tensor, x2: Tensor, look_ahead_mask: Tensor):
        # Attention block.
        x_attn, x2_ff, attn_weights, att_scores1, att_scores12 = \
            self.mhca(x, x2, look_ahead_mask)  # (B, S, D)
        x_attn = self.dropout1(x_attn)  # (B,S,D)
        x = self.layernorm1(x + x_attn)  # (B,S,D)
        # Feed forward block.
        x_ff = self.activation(self.linear1(x))
        x_ff = self.linear2(self.dropout(x_ff))
        x_ff = self.dropout2(x_ff)
        output = self.layernorm2(x + x_ff)  # (B, S, D)
        return output, x2_ff, attn_weights, att_scores1, att_scores12


class TransformerCrossBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout_rate: float) -> None:
        '''Uses MultiheadAttention implemented by torch.nn.'''
        super(TransformerCrossBlock, self).__init__()
        D = d_model
        H = nhead
        self.dropout_rate = dropout_rate
        self.linear_time = nn.Linear(D, D)
        self.linear1 = nn.Linear(D, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.self_attn1 = MultiheadAttention(D, H, batch_first=True)
        self.self_attn2 = MultiheadAttention(D, H, batch_first=True)
        self.gelu = nn.GELU()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x: Tensor, x2: Tensor, look_ahead_mask: Tensor):
        look_ahead_mask = F._canonical_mask(
            mask=look_ahead_mask,
            mask_name="look_ahead_mask",
            other_type=None,
            other_name="",
            target_type=x.dtype,
            check_other=False,
        ).to(dtype=torch.bool, device=x.device)
        # Attention block.
        h_attn, attn_1 = self.self_attn1(x, x, x, attn_mask=look_ahead_mask)  # (B,S,D)
        _, attn_2 = self.self_attn2(x, x2, x2, attn_mask=look_ahead_mask)  # (B,S,D)
        h2_ff = self.linear_time(self.dropout(x2))
        h1_attn = self.dropout(h_attn)  # (B,S,D)
        h1 = self.layernorm1(x + h1_attn)  # (B,S,D)
        # Feed forward block.
        h1_ff = self.gelu(self.linear1(h1))
        h1_ff = self.linear2(self.dropout(h1_ff))
        h1_ff = self.dropout(h1_ff)
        h1_out = self.layernorm2(h1 + h1_ff)  # (B, S, D)
        attn_weights = attn_1 + attn_2
        attn_weights /= attn_weights.sum(dim=-1, keepdim=True)
        return h1_out, h2_ff, attn_weights, attn_1, attn_2


class TransformerCrossEncoder(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, encoder_layer: TransformerCrossBlock, num_layers: int):
        super(TransformerCrossEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        # Initialize weights with xavier normal.
        for layer in self.encoder_layers:
            layer.init_weights()

    def forward(self, x: Tensor, x2: Tensor, mask: Tensor):
        attn_weights, attn_scores1, attn_scores12 = [], [], []
        for layer in self.encoder_layers:
            x, x2, attn_w, annt1, attn2 = layer(x, x2, mask)
            attn_weights.append(attn_w)
            attn_scores1.append(annt1)
            attn_scores12.append(attn2)
        return x, attn_weights, attn_scores1, attn_scores12
