from typing import Optional
from dataclasses import dataclass

from pyrallis.argparsing import parse


@dataclass
class ModelConfig:
  backbone: str
  head: str
  vertices: int


@dataclass
class Config:
  """Training config for Machine Learning"""

  dataset: str
  model: ModelConfig
  loss_function: str
  loss_stepwise: bool
  batch_size: int
  samples_per_image: int
  seed: int
  name: Optional[str] = None
  resume_from: Optional[str] = None
  wandb_id: str = ""


config = parse(config_class=Config, config_path="config.yaml")


def load_config(path):
  global config
  config = parse(config_class=Config, config_path=path)


__all__ = ["config", "load_config"]
