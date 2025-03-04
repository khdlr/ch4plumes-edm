import jax
import jax.numpy as jnp
import augmax
from einops import rearrange
from typing import Union, Sequence, Optional, Tuple
from subprocess import check_output
from inspect import currentframe


def debug_break():
  print("== Debug Break ==")
  cf = currentframe()
  print(cf.f_back)
  input()


def prep(batch, key=None, input_types=None):
  do_augment = key is not None
  ops = []
  if do_augment:
    ops += [
      # augmax.Warp(coarseness=64, strength=8),
      # augmax.Warp(coarseness=32, strength=4),
    ]

  if input_types is None:
    input_types = (
      augmax.InputType.DENSE,
      augmax.InputType.DENSE,
      augmax.InputType.DENSE,
    )
  chain = augmax.Chain(*ops, input_types=input_types)
  if not do_augment:
    key = jax.random.PRNGKey(0)
  subkeys = jax.random.split(key, batch[0].shape[0])
  transformation = jax.vmap(chain)
  outputs = list(transformation(subkeys, batch))

  # Normalize contour
  outputs[1] = 2 * (outputs[1] / outputs[0].shape[1]) - 1.0

  return outputs


def distance_matrix(a, b):
  a = rearrange(a, "(true pred) d -> true pred d", true=1)
  b = rearrange(b, "(true pred) d -> true pred d", pred=1)
  D = jnp.sum(jnp.square(a - b), axis=-1)
  return D


def pad_inf(inp, before, after):
  return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def fmt_num(x):
  if jnp.isinf(x):
    return "∞".rjust(8)
  else:
    return f"{x:.2f}".rjust(8)


def fmt(xs, extra=None):
  tag = ""
  if isinstance(xs, str):
    tag = xs
    xs = extra
  rank = len(xs.shape)
  if rank == 1:
    print(tag, ",".join([fmt_num(x) for x in xs]))
  elif rank == 2:
    print("\n".join(",".join([fmt_num(x) for x in row]) for row in xs))
    print()


def assert_git_clean():
  diff = (
    check_output(["git", "diff", "--name-only", "HEAD"]).decode("utf-8").splitlines()
  )
  if diff and diff != ["config.yml"]:
    assert False, "Won't run on a dirty git state!"


def _infer_shape(
  x: jnp.ndarray,
  size: Union[int, Sequence[int]],
  channel_axis: Optional[int] = -1,
) -> Tuple[int, ...]:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return (1,) + tuple(size if d != channel_axis else 1 for d in range(1, x.ndim))
  elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
    return (1,) * (x.ndim - len(size)) + tuple(size)
  else:
    assert x.ndim == len(size)
    return tuple(size)


def min_pool(
  value: jnp.ndarray,
  window_shape: Union[int, Sequence[int]],
  strides: Union[int, Sequence[int]],
  padding: str = "SAME",
  channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
  """Min pool.
  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, an int or same rank as value.
    strides: Strides of the pooling window, an int or same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped,
      used to infer ``window_shape`` or ``strides`` if they are an integer.
  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return jax.lax.reduce_window(
    value, jnp.inf, jax.lax.min, window_shape, strides, padding
  )


def fnot(fun):
  return lambda x: not fun(x)
