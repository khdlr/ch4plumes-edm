import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class UNet(nnx.Module):
  def __init__(self, C, in_dim, out_dim=None, *, rngs: nnx.Rngs):
    if out_dim is None:
      out_dim = in_dim
    self.init = nnx.Conv(in_dim, C, [1, 1], rngs=rngs)
    self.final = nnx.Conv(C, out_dim, [1, 1], rngs=rngs)

    self.part1 = [
      (
        nnx.Conv(1 * C, 1 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(1 * C, 1 * C, [3, 3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(1 * C, 2 * C, [2, 2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [2, 2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [2, 2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
      ),
    ]
    self.part2 = [
      (
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 2 * C, [2, 2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 2 * C, [2, 2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3, 3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 1 * C, [2, 2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(1 * C, 1 * C, [3, 3], strides=1, rngs=rngs),
        nnx.Conv(1 * C, 1 * C, [3, 3], strides=1, rngs=rngs),
      ),
    ]

  def __call__(self, x, sigma):
    print("TODO: UNet is not using sigma yet...")

    x = self.init(x)
    print("x_raw", x.shape)
    x = jnp.pad(x, [(0, 0), (5, 5), (4, 4), (0, 0)])
    print("x_padded", x.shape)
    stack = []
    for blocks in self.part1:
      stack.append(x)
      for block in blocks:
        x = jax.nn.silu(block(x))

    for blocks, feats in zip(self.part2, reversed(stack)):
      for block in blocks:
        x = jax.nn.silu(block(x))
      x += feats

    x = x[:, 5:-5, 4:-4, :]
    x = self.final(x)
    return x


UNet_T = partial(UNet, 32)
UNet_S = partial(UNet, 48)
UNet_B = partial(UNet, 64)
UNet_L = partial(UNet, 96)
UNet_XL = partial(UNet, 128)
