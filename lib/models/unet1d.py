import jax
from flax import nnx


class UNet1D(nnx.Module):
  def __init__(self, C, *, rngs: nnx.Rngs):
    self.part1 = [
      (
        nnx.Conv(1 * C, 1 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(1 * C, 1 * C, [3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(1 * C, 2 * C, [2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [2], strides=2, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
      ),
    ]
    self.part2 = [
      (
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 2 * C, [2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 2 * C, [2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(2 * C, 2 * C, [3], strides=1, rngs=rngs),
        nnx.ConvTranspose(2 * C, 1 * C, [2], strides=2, rngs=rngs),
      ),
      (
        nnx.Conv(1 * C, 1 * C, [3], strides=1, rngs=rngs),
        nnx.Conv(1 * C, 1 * C, [3], strides=1, rngs=rngs),
      ),
    ]

  def __call__(self, x):
    stack = []
    for blocks in self.part1:
      stack.append(x)
      for block in blocks:
        x = jax.nn.silu(block(x))

    for blocks, feats in zip(self.part2, reversed(stack)):
      for block in blocks:
        x = jax.nn.silu(block(x))
      x += feats

    return x
