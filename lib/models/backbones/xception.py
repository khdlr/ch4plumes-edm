from typing import Tuple
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
from flax import nnx
from functools import partial

from .. import nnutils as nn


class Xception(nnx.Module):
  def __init__(self, c_in, *, rngs: nnx.Rngs):
    pass

  def __call__(self, x, dropout_rate=0.0):
    B, H, W, C = x.shape
    return [jnp.zeros([B, H // 16, W // 16, 64]), jnp.zeros([B, H // 4, W // 4, 512])]


class TrueXception(nnx.Module):
  """Xception backbone like the one used in CALFIN"""

  def __init__(self, c_in, *, rngs: nnx.Rngs):
    block = partial(XceptionBlock, strides=2, return_skip=True, rngs=rngs)
    # Pre-reduction
    self.block1 = block(c_in, [16, 64, 64])
    self.block2 = block(64, [128, 128, 128])
    self.block3 = block(128, [256, 256, 512])

    self.middle = [
      XceptionBlock(512, [512, 512, 512], skip_type="sum", strides=1, rngs=rngs)
      for _ in range(4)
    ]
    self.block4 = XceptionBlock(512, [512, 512, 512], strides=2, rngs=rngs)
    self.block5 = XceptionBlock(
      512, [512, 512, 512], strides=1, kernel_dilation=(1, 2, 4), rngs=rngs
    )

    self.aspp = [
      BDBlock(512, 128, rngs=rngs),
      nn.ConvBNAct(512, 128, (1, 1), act="elu", rngs=rngs),
    ] + [
      nn.SepConvBN(512, 128, (3, 3), kernel_dilation=r, rngs=rngs) for r in range(1, 6)
    ]

    self.final = nn.ConvBNAct(128 * 7, 512, (1, 1), act="elu", rngs=rngs)
    self.skip_final = nn.ConvBNAct(256, 64, (1, 1), act="elu", rngs=rngs)
    self.dropout = nn.ChannelDropout(rngs=rngs)

  def __call__(self, x, dropout_rate=0.0):
    dropout = partial(jax.tree.map, partial(self.dropout, dropout_rate=dropout_rate))

    # Backbone
    x, _ = dropout(self.block1(x))
    x, _ = dropout(self.block2(x))
    x, skip3 = dropout(self.block3(x))

    for block in self.middle:
      x = dropout(block(x))

    x = dropout(self.block4(x))
    x = dropout(self.block5(x))

    # ASPP
    # Image Feature branch
    x = dropout(jnp.concatenate([bl(x) for bl in self.aspp], axis=-1))

    x = dropout(self.final(x))
    skip3 = dropout(self.skip_final(skip3))

    print(x.shape)
    print(skip3.shape)
    return [skip3, x]


class BDBlock(nnx.Module):
  def __init__(self, c_in, c_out, *, rngs: nnx.Rngs):
    self.conv = nn.ConvBNAct(c_in, c_out, [1, 1], act="elu", rngs=rngs)

  def __call__(self, x):
    x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
    x = self.conv(x)
    x = nn.upsample(x, factor=2)
    return x


class XceptionBlock(nnx.Module):
  def __init__(
    self,
    c_in,
    depth_list,
    strides: int | Tuple[int, int],
    skip_type="conv",
    kernel_dilation: int | Tuple[int, int, int] = 1,
    return_skip=False,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.blocks = []
    if isinstance(kernel_dilation, int):
      kernel_dilation = (kernel_dilation, kernel_dilation, kernel_dilation)
    if isinstance(strides, int):
      strides = (strides, strides)
    c_current = c_in
    for i in range(3):
      self.blocks.append(
        nn.SepConvBN(
          c_current,
          depth_list[i],
          kernel_size=(3, 3),
          strides=strides if i == 2 else 1,
          kernel_dilation=kernel_dilation[i],
          rngs=rngs,
        )
      )
      c_current = depth_list[i]

    if skip_type == "conv":
      self.shortcut = nn.ConvBNAct(
        c_in, depth_list[-1], [1, 1], strides=strides, act=None, rngs=rngs
      )
    elif skip_type == "sum":
      self.shortcut = nn.identity
    self.return_skip = return_skip

  def __call__(self, inputs):
    residual = inputs
    for i, block in enumerate(self.blocks):
      residual = block(residual)
      if i == 1:
        skip = residual

    shortcut = self.shortcut(inputs)
    outputs = residual + shortcut

    if self.return_skip:
      return outputs, skip
    else:
      return outputs
