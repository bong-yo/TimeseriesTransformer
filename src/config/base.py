from pydantic import BaseSettings, Field
from torch import cuda
from src.config.data import DatasetsConfig
from src.config.training import TrainConfig
from src.config.model import ModelConfig


class Config(BaseSettings):
    seed: int = Field(1337, description="Random seed for deterministic runs")
    name: str = Field("Timeformer", description="Name of the experiment")
    outputs_folder: str = Field("outputs", description="Path to the folder to store outputs if any")
    device: str = Field("cuda" if cuda.is_available() else "cpu", description="Device to use for training")

    datasets: DatasetsConfig = DatasetsConfig()
    training: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()