import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx

from . import backbones
from . import nnutils as nn
from . import snake_utils
from .transformer import Transformer
from .unet1d import UNet1D


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
    self.head = SnakeHead(576, self.model_dim, config.blocks, rngs=rngs)
    self.dropout = nn.ChannelDropout(rngs=rngs)
    self.rngs = rngs

  def __call__(self, imagery, sigma, dropout_rate=0.0):
    feature_maps = self.backbone(imagery, dropout_rate=dropout_rate)

    init_keys = jax.random.split(self.rngs(), imagery.shape[0])
    init_fn = partial(snake_utils.random_bezier, vertices=self.vertices)
    vertices = jax.vmap(init_fn)(init_keys)
    steps = [vertices]

    for _ in range(self.iterations):
      vertices = jax.lax.stop_gradient(vertices)
      vertices = vertices + self.head(vertices, feature_maps, sigma)
      steps.append(vertices)

    return {"snake_steps": steps, "snake": vertices}


class SnakeHead(nnx.Module):
  def __init__(self, d_in, d_hidden, blocks, *, rngs: nnx.Rngs):
    super().__init__()

    D = d_in
    C = d_hidden

    self.init_coords = nnx.Conv(2, C, [1], strides=1, rngs=rngs)

    # self.model = Transformer(d_hidden, blocks, rngs=rngs)
    self.model = UNet1D(d_hidden, rngs=rngs)

    # Initialize offset predictor with 0 -> default to no change
    self.mk_offset = nnx.ConvTranspose(
      C,
      2,
      [1],
      strides=1,
      use_bias=False,
      kernel_init=nnx.initializers.zeros,
      rngs=rngs,
    )

    self.init_features = nnx.Conv(D, C, [1], strides=1, use_bias=False, rngs=rngs)
    self.dropout = nn.ChannelDropout(rngs=rngs)

    self.local_condition_gate = nnx.Linear(1, C, rngs=rngs)
    self.global_condition_gate = nnx.Linear(1, C, rngs=rngs)

  def __call__(self, vertices, features, sigma, *, dropout_rate=0.0):
    # Start with coord features
    x = self.init_coords(vertices)
    if features is not None:
      # Condition on local image features
      vertex_features = []
      for feature_map in features:
        feat = jax.vmap(snake_utils.sample_at_vertices, [0, 0])(vertices, feature_map)
        feat = self.dropout(feat, dropout_rate=dropout_rate)
        vertex_features.append(feat)
      gate = jax.nn.sigmoid(self.local_condition_gate(sigma))
      x += gate * self.init_features(jnp.concatenate(vertex_features, axis=-1))

      # Condition on global image features
    x = jax.nn.silu(x)

    x = self.model(x)

    offsets = self.mk_offset(x)
    # jax.debug.print("Init: {} – Pred: {}", jnp.abs(x).mean(), jnp.abs(offsets).mean())

    return offsets
