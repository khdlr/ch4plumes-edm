import sys
from dataclasses import dataclass

from pyrallis.argparsing import parse


@dataclass
class ModelConfig:
  backbone: str
  model_dim: int
  blocks: int
  iterations: int
  vertices: int


@dataclass
class Config:
  """Training config for Machine Learning"""

  name: str
  model: ModelConfig
  loss_function: str
  loss_stepwise: bool
  batch_size: int
  samples_per_image: int
  seed: int
  wandb_id: str = ""


config = parse(config_class=Config, config_path="config.yaml")


def load_config(path):
  global config
  config = parse(config_class=Config, config_path=path)


__all__ = ["config", "load_config"]
