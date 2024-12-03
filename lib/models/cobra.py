import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx

from . import backbones
from . import nnutils as nn
from . import snake_utils


class COBRA(nnx.Module):
  def __init__(
    self,
    config,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.backbone = getattr(backbones, config.backbone)(c_in=4, rngs=rngs)
    self.model_dim = config.model_dim
    self.iterations = config.iterations
    self.vertices = config.vertices
    self.head = snake_utils.SnakeHead(576, self.model_dim, rngs=rngs)
    self.dropout = nn.ChannelDropout(rngs=rngs)
    self.rngs = rngs

  def __call__(self, imagery, dropout_rate=0.0):
    feature_maps = self.backbone(imagery, dropout_rate=dropout_rate)

    init_keys = jax.random.split(self.rngs(), imagery.shape[0])
    init_fn = partial(snake_utils.random_bezier, vertices=self.vertices)
    vertices = jax.vmap(init_fn)(init_keys)
    steps = [vertices]

    for _ in range(self.iterations):
      vertices = jax.lax.stop_gradient(vertices)
      vertices = vertices + self.head(vertices, feature_maps)
      steps.append(vertices)

    return {"snake_steps": steps, "snake": vertices}

  def ddpm_forward(self, imagery, vertices):
    "Get snake head output when already given some vertices, useful for DDPM training"
    features = self.backbone(imagery, dropout_rate=0.0)
    return self.head(vertices, features)
