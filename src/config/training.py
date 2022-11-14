from pydantic import BaseModel, Field
from torch.nn import MSELoss


class Losses:
    mse: MSELoss

class TrainConfig(BaseModel):
    epochs: int = Field(100, description="Max number of epochs in training.")
    batch_size: int = Field(4, description="Batch size")
    lr: float = Field(1e-3, description="Learning rate")
    patience: int = Field(5, description="Num. epochs without performance improvement before early stopping the training.")
    delta: float = Field(0, description="Min value to consider that performance actually improved.")
    p_dropout: float = Field(0.3, description="Droput prob. to apply during trainig.")
