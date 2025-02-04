# Adapted from:
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
from flax import nnx
from functools import partial

from .. import nnutils as nn


class Block(nnx.Module):
  """ConvNeXt Block"""

  def __init__(self, dim, layer_scale_init_value=1e-6, *, rngs: nnx.Rngs):
    self.dwconv = nnx.Conv(dim, dim, (7, 7), feature_group_count=dim, rngs=rngs)
    self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
    self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
    self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)
    self.gamma = nnx.Param(layer_scale_init_value * jnp.ones((1, 1, 1, dim)))

  def __call__(self, x):
    input = x
    x = self.dwconv(x)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = jax.nn.gelu(x)
    x = self.pwconv2(x)
    if self.gamma is not None:
      x = self.gamma * x
    # TODO: Drop path
    x = input + x
    return x


class ConvNeXt(nnx.Module):
  """Xception backbone like the one used in CALFIN"""

  def __init__(self, c_in, depths, dims, *, rngs: nnx.Rngs):
    stem = nnx.Sequential(
      nnx.Conv(c_in, dims[0], (4, 4), strides=(4, 4), rngs=rngs),
      nnx.LayerNorm(dims[0], epsilon=1e-6, rngs=rngs),
    )
    self.downsample_layers = [stem]
    for i in range(3):
      self.downsample_layers.append(
        nnx.Sequential(
          nnx.LayerNorm(dims[i], epsilon=1e-6, rngs=rngs),
          nnx.Conv(dims[i], dims[i + 1], (2, 2), strides=(2, 2), rngs=rngs),
        )
      )
    self.stages = []
    for i in range(4):
      self.stages.append(
        nnx.Sequential(
          *[Block(dim=dims[i], rngs=rngs) for j in range(depths[i])],
        )
      )

  def __call__(self, x, dropout_rate=0.0):
    feature_maps = []
    for i in range(4):
      x = self.downsample_layers[i](x)
      x = self.stages[i](x)
      feature_maps.append(x)
    return feature_maps[1], feature_maps[3]


ConvNeXt_T = partial(ConvNeXt, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
ConvNeXt_S = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
ConvNeXt_B = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
ConvNeXt_L = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
ConvNeXt_XL = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
