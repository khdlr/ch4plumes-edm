import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
from . import nnutils as nn
from functools import partial
from flax import nnx


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
