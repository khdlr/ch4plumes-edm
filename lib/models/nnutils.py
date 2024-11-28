import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


def identity(x, *args, **kwargs):
  return x


class ChannelDropout(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.rngs = rngs

  def __call__(self, x, dropout_rate: float):
    if dropout_rate < 0 or dropout_rate >= 1:
      raise ValueError("rate must be in [0, 1).")

    if dropout_rate == 0.0:
      return x

    keep_rate = 1.0 - dropout_rate
    mask_shape = (x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])

    keep = jax.random.bernoulli(self.rngs(), keep_rate, shape=mask_shape)
    return x * (keep / keep_rate)


def channel_dropout(x, dropout_rate, *, rngs: nnx.Rngs):
  if dropout_rate < 0 or dropout_rate >= 1:
    raise ValueError("rate must be in [0, 1).")

  if dropout_rate == 0.0:
    return x

  keep_rate = 1.0 - dropout_rate
  mask_shape = (x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])

  keep = jax.random.bernoulli(rngs(), keep_rate, shape=mask_shape)
  return x * (keep / keep_rate)


def sample_dropout(x, dropout_rate, *, rngs: nnx.Rngs):
  if dropout_rate < 0 or dropout_rate >= 1:
    raise ValueError("rate must be in [0, 1).")

  if dropout_rate == 0.0:
    return x

  keep_rate = 1.0 - dropout_rate
  mask_shape = (x.shape[0], *((1,) * (x.ndim - 1)))

  keep = jax.random.bernoulli(rngs(), keep_rate, shape=mask_shape)
  return keep * x


class ConvBNAct(nnx.Module):
  def __init__(
    self,
    c_in: int,
    c_out: int,
    *args,
    bn: bool = True,
    act: Optional[str] = "relu",
    rngs: nnx.Rngs,
    **kwargs,
  ):
    super().__init__()
    kwargs["use_bias"] = False
    self.conv = nnx.Conv(c_in, c_out, *args, **kwargs, rngs=rngs)
    self.args = args
    self.kwargs = kwargs

    if bn:
      self.bn = nnx.BatchNorm(c_out, rngs=rngs)
    else:
      self.bn = identity

    if act is None:
      self.act = identity
    elif hasattr(jax.nn, act):
      self.act = getattr(jax.nn, act)
    else:
      raise ValueError(f"no activation called {act}")

  def __call__(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x


class SepConvBN(nnx.Module):
  def __init__(
    self,
    c_in: int,
    c_out: int,
    kernel_size: int,
    depth_activation=False,
    *,
    rngs: nnx.Rngs,
    **kwargs,
  ):
    super().__init__()
    self.depth_activation = depth_activation

    self.channel_conv = nnx.Conv(c_in, c_out, [1, 1], rngs=rngs, use_bias=False)
    self.spatial_conv = ConvBNAct(
      c_out,
      c_out,
      [kernel_size, kernel_size],
      feature_group_count=c_out,
      **kwargs,
      rngs=rngs,
    )

  def __call__(self, x):
    if self.depth_activation:
      x = jax.nn.relu(x)
    x = self.channel_conv(x)
    x = self.spatial_conv(x)

    return x


def upsample(x, factor=None, shp=None):
  B, H, W, C = x.shape
  if factor is not None:
    H *= factor
    W *= factor
  else:
    H, W = shp
  return jax.image.resize(x, [B, H, W, C], "bilinear")
