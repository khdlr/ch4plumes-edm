from typing import Optional
from dataclasses import dataclass

from pyrallis.argparsing import parse


@dataclass
class Config:
  """Training config for Machine Learning"""

  dataset: str
  model_type: str
  loss_function: str
  batch_size: int
  seed: int
  name: Optional[str] = None
  resume_from: Optional[str] = None
  wandb_id: str = ""


config = parse(config_class=Config, config_path="config.yaml")


def load_config(path):
  global config
  config = parse(config_class=Config, config_path=path)


__all__ = ["config", "load_config"]
