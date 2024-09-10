import jax
import jax.numpy as jnp
import haiku as hk
import optax
import augmax
import numpy as np
from einops import rearrange
from jax.experimental import host_callback
from skimage.measure import find_contours
from functools import partial
from typing import Union, Sequence, Optional, Tuple
from subprocess import check_output
from typing import NamedTuple
import pickle


class TrainingState(NamedTuple):
    params: hk.Params
    buffers: hk.State
    opt: optax.OptState


def changed_state(state, params=None, buffers=None, opt=None):
    return TrainingState(
        params=state.params if params is None else params,
        buffers=state.buffers if buffers is None else buffers,
        opt=state.opt if opt is None else opt,
    )


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open("wb") as f:
        pickle.dump(state, f)


def load_state(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    return state


def prep(batch, key=None, augment=False, input_types=None):
    ops = []
    if augment:
        ops += [
            augmax.HorizontalFlip(),
            augmax.VerticalFlip(),
            augmax.Rotate90(),
            augmax.Rotate(15),
            augmax.Warp(coarseness=64, strength=8),
            augmax.Warp(coarseness=32, strength=4),
        ]
    ops += [augmax.ByteToFloat()]
    if augment: ops += [
        augmax.Solarization(p=0.1),
        augmax.ColorJitter(),
        # augmax.RandomChannelGamma(p=0.5, range=(0.5, 2.0)),
        augmax.RandomBrightness(p=0.5),
        augmax.RandomContrast(p=0.5),
        augmax.RandomGrayscale(p=0.2),
        augmax.GaussianBlur(sigma=2)
    ]

    if input_types is None:
        input_types = (
            augmax.InputType.IMAGE,
            augmax.InputType.DENSE,
            augmax.InputType.CONTOUR,
          )
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment == False:
        key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, batch[0].shape[0])
    transformation = jax.vmap(chain)
    outputs = list(transformation(subkeys, batch))
    outputs = [
        # Stack img and dem
        jnp.concatenate([outputs[0], outputs[1]], axis=-1),
        # Normalize contour
        2 * (outputs[2] / outputs[0].shape[1]) - 1.0
    ]

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
        check_output(["git", "diff", "--name-only", "HEAD"])
        .decode("utf-8")
        .splitlines()
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
