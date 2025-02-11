import jax
from flax import nnx
from functools import partial


class UNet1D(nnx.Module):
  def __init__(self, C, *, rngs: nnx.Rngs):
    self.dim = C
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
    for i, blocks in enumerate(self.part1):
      with jax.profiler.TraceAnnotation(f"unet_up{i}"):
        stack.append(x)
        for block in blocks:
          x = jax.nn.silu(block(x))

    for i, (blocks, feats) in enumerate(zip(self.part2, reversed(stack))):
      with jax.profiler.TraceAnnotation(f"unet_down{i}"):
        for block in blocks:
          x = jax.nn.silu(block(x))
        x += feats

    return x

  def get_dim(self):
    return self.dim


UNet1D_T = partial(UNet1D, 256)
UNet1D_S = partial(UNet1D, 384)
UNet1D_B = partial(UNet1D, 512)
UNet1D_L = partial(UNet1D, 768)
