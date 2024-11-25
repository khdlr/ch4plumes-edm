import sys
from dataclasses import dataclass
import pyrallis


@dataclass
class ModelConfig:
  backbone: str
  model_dim: int
  iterations: int
  coord_features: bool
  stop_grad: bool
  weight_sharing: bool
  head: str


@dataclass
class Config:
  """Training config for Machine Learning"""

  name: str
  model: ModelConfig
  loss_function: str
  loss_stepwise: bool
  batch_size: int
  wandb_id: str = ""


config: Config
this = sys.modules[__name__]


def load_config():
  this.config = pyrallis.parse(config_class=Config, config_path="config.yaml")


__all__ = ["config", "load_config"]
