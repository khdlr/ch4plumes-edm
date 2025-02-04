import jax
import jax.numpy as jnp
from flax import nnx

from functools import partial


class Block(nnx.Module):
  """ConvNeXt Block"""

  def __init__(self, dim, layer_scale_init_value=1e-6, *, rngs: nnx.Rngs):
    self.dwconv = nnx.Conv(dim, dim, (7,), feature_group_count=dim, rngs=rngs)
    self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
    self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
    self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)
    self.gamma = nnx.Param(layer_scale_init_value * jnp.ones((1, 1, dim)))

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


class ConvNeXtHead(nnx.Module):
  def __init__(self, depths, dims, *, rngs: nnx.Rngs):
    self.down_layers = []
    self.up_layers = []
    self.down_stages = []
    self.up_stages = []
    self.dims = dims
    self.depths = depths

    for i in range(3):
      self.down_layers.append(
        nnx.Sequential(
          nnx.LayerNorm(dims[i], epsilon=1e-6, rngs=rngs),
          nnx.Conv(dims[i], dims[i + 1], (2,), strides=(2,), rngs=rngs),
        )
      )
      self.up_layers.append(
        nnx.Sequential(
          nnx.LayerNorm(dims[i + 1], epsilon=1e-6, rngs=rngs),
          nnx.ConvTranspose(dims[i + 1], dims[i], (2,), strides=(2,), rngs=rngs),
        )
      )
      self.down_stages.append(
        nnx.Sequential(
          *[Block(dim=dims[i], rngs=rngs) for _ in range(depths[i])],
        )
      )
      self.up_stages.append(
        nnx.Sequential(
          *[Block(dim=dims[i], rngs=rngs) for _ in range(depths[i])],
        )
      )
    self.bottleneck = nnx.Sequential(
      *[Block(dim=dims[-1], rngs=rngs) for _ in range(depths[-1])],
    )

  def __call__(self, x):
    stack = []
    for i in range(3):
      x = self.down_layers[i](x)
      x = self.down_stages[i](x)
      stack.append(x)
    x = self.bottleneck(x)
    for i in reversed(range(3)):
      x = x + stack[i]
      x = self.up_layers[i](x)
      x = self.up_stages[i](x)
    return x

  def get_dim(self):
    return self.dims[0]


ConvNeXt_T_Head = partial(ConvNeXtHead, depths=[2, 2, 2, 4], dims=[96, 192, 384, 768])
ConvNeXt_S_Head = partial(ConvNeXtHead, depths=[2, 2, 4, 8], dims=[96, 192, 384, 768])
# ConvNeXt_B = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
# ConvNeXt_L = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
# ConvNeXt_XL = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
