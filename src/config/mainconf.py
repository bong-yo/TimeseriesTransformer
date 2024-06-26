from pydantic import BaseModel, Field
from torch import cuda
from src.config.data import DatasetsConfig
from src.config.training import TrainConfig
from src.config.model import ModelConfig


class Config(BaseModel):
    seed: int = Field(1337, description="Random seed for deterministic runs")
    name: str = Field("Timeformer", description="Name of the experiment")
    outputs_folder: str = Field("outputs", description="Path to the folder to store outputs if any")
    modelsaves: str = Field('model_saves', description='model weights and configuration are saved in this folder')
    device: str = Field("cuda" if cuda.is_available() else "cpu", description="Device to use for training")

    datasets: DatasetsConfig = DatasetsConfig()
    training: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
