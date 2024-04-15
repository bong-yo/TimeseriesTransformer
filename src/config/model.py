from typing import Literal
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field("TimeFormer", description="Name of the model to use")
    values_size: int = Field(4, description="Number of input features")
    time_feats_size: int = Field(4, description="Size of time features embeddings")
    emb_size: int = Field(64, description="Size of transformer hidden layer")
    dim_feedforward: int = Field(128, description='Dimension of the ff layer in the transformer')
    n_att_heads: int = Field(1, description="Number of attention heads")
    depth: int = Field(6, description="Number of transformer blocks")
    max_seq_len: int = Field(14, description="Lengths of longest seq supported")
    out_size: int = Field(4, description="Dimension of output embeddings")
    attention_type: Literal['standard', 'cross'] = Field('cross', description="Type of attention to use")
    saves: str = Field('model_saves')
