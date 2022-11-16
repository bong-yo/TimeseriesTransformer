from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = Field("TimeFormer", description="Name of the model to use")
    input_size: int = Field(1, description="Size of innput embeddings")
    emb_size: int = Field(32, description="Size of transformer hidden layer")
    dim_feedforward: int = Field(64, description='Dimension of the ff layer in the transformer')
    n_att_heads: int = Field(1, description="Number of attention heads")
    depth: int = Field(1, description="Number of transformer blocks")
    max_seq_len: int = Field(200, description="Lengths of longest seq supported")
    out_size: int = Field(1, description="Dimension of output embeddings")
    saves: str = Field('model_saves')
