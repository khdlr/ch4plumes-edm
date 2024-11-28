import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
from . import nnutils as nn
from functools import partial
from flax import nnx


class SnakeHead(nnx.Module):
  def __init__(self, d_in, d_hidden, coord_features=False, *, rngs: nnx.Rngs):
    super().__init__()
    self.coord_features = coord_features

    D = d_in
    C = d_hidden

    if self.coord_features:
      D += 2

    self.blocks = [
      nnx.Conv(D, C, [1], rngs=rngs),
      # simplify nnx.Conv(C, C, [3], rngs=rngs),
      # simplify nnx.Conv(C, C, [3], kernel_dilation=3, rngs=rngs),
      # simplify nnx.Conv(C, C, [3], kernel_dilation=9, rngs=rngs),
      # simplify nnx.Conv(C, C, [3], kernel_dilation=9, rngs=rngs),
      # simplify nnx.Conv(C, C, [3], kernel_dilation=3, rngs=rngs),
      # simplify nnx.Conv(C, C, [3], rngs=rngs),
    ]

    # Initialize offset predictor with 0 -> default to no change
    self.mk_offset = nnx.Conv(
      C, 2, [1], use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs
    )

    self.dropout = nn.ChannelDropout(rngs=rngs)

  def __call__(self, vertices, feature_maps, *, dropout_rate=0.0):
    features = []
    for feature_map in feature_maps:
      features.append(jax.vmap(sample_at_vertices, [0, 0])(vertices, feature_map))
    # For coordinate features
    if self.coord_features:
      diff = vertices[:, 1:] - vertices[:, :-1]
      diff = jnp.pad(diff, [(0, 0), (1, 1), (0, 0)])
      features.append(diff[:, 1:])
      features.append(diff[:, :-1])
    x = jnp.concatenate(features, axis=-1)

    for block in self.blocks:
      x = jax.nn.relu(block(x))
      x = self.dropout(x, dropout_rate)
    offsets = self.mk_offset(x)
    return offsets


def random_bezier(key, vertices):
  t = jnp.linspace(0, 1, vertices).reshape(1, -1, 1)
  points = jax.random.uniform(key, [5, 1, 2], minval=-1, maxval=1)
  while points.shape[0] != 1:
    points = points[1:] * t + points[:-1] * (1 - t)
  return points[0]


def subdivide_polyline(polyline):
  _, T, C = polyline.shape
  T_new = T * 2 - 1
  resized = jax.vmap(partial(jax.image.resize, shape=(T_new, C), method="linear"))(
    polyline
  )
  return resized


def sample_at_vertices(vertices: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
  H, W, _ = features.shape
  vertices = (vertices + 1.0) * (jnp.array([[H - 1, W - 1]]) / 2.0)

  def resample_feature(feature_map: jnp.ndarray):
    return jnd.map_coordinates(feature_map, list(vertices.T), order=1, mode="constant")

  resampled = jax.vmap(resample_feature, in_axes=-1, out_axes=-1)(features)

  return resampled
