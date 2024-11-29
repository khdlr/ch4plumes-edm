import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import nnx
import optax

from abc import abstractmethod
from .utils import pad_inf, distance_matrix


class LossFunction(nnx.Module):
  def __init__(self):
    self.metrics = []

  def __call__(self, terms, metric_scale=None):
    loss, out = self.impl(terms)

    detached = jax.tree.map(jax.lax.stop_gradient, terms)
    if metric_scale is not None:
      detached = jax.tree.map(lambda x: x * metric_scale, detached)

    for metric in self.metrics:
      _, metric_terms = metric(detached)
      out.update(metric_terms)
    return loss, out

  @abstractmethod
  def impl(self, terms):
    raise NotImplementedError()

  def add_metric(self, metric):
    self.metrics.append(metric)


def metric(name):
  def decorator(fun):
    class LossModule(LossFunction):
      def impl(self, terms):
        metric = jax.vmap(fun)(terms["snake"], terms["contour"])
        metric = jnp.mean(metric)
        return metric, {name: metric}

    return LossModule

  return decorator


class StepwiseLoss(LossFunction):
  def __init__(self, loss_fn):
    super().__init__()
    self.loss_fn = loss_fn

  def impl(self, terms):
    out = {}
    totals = []
    for step in range(1, len(terms["snake_steps"])):
      total, out_terms = self.loss_fn(
        {"snake": terms["snake_steps"][step], "contour": terms["contour"]}
      )
      totals.append(total)
      out.update({f"{k}_step{step}": out_terms[k] for k in out_terms})
    return sum(totals), out


##### Generic Loss Functions ####
@metric("l2")
def L2(snake, contour):
  loss = jnp.sum(jnp.square(snake - contour), axis=-1)
  loss = jnp.mean(loss)
  return loss


@metric("l1")
def L1(snake, contour):
  loss = jnp.sum(jnp.abs(snake - contour), axis=-1)
  return loss


@metric("huber")
def Huber(snake, contour):
  loss = optax.huber_loss(snake, contour, delta=0.05)
  return loss


def min_min_loss(snake, contour):
  D = distance_matrix(snake, contour)
  min1 = D.min(axis=0)
  min2 = D.min(axis=1)
  min_min = 0.5 * (jnp.mean(min1) + jnp.mean(min2))
  return min_min


##### METRICS #####
@metric("mae")
def MAE(snake, contour):
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.mean(jnp.sqrt(squared_distances))


@metric("rmse")
def RMSE(snake, contour):
  squared_distances = jnp.sum(jnp.square(snake - contour), axis=-1)
  return jnp.sqrt(jnp.mean(squared_distances))


def forward_mae(snake, contour):
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.mean(jnp.sqrt(squared_dist))


def backward_mae(snake, contour):
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.mean(jnp.sqrt(squared_dist))


def forward_rmse(snake, contour):
  squared_dist = squared_distance_points_to_curve(snake, contour)
  return jnp.sqrt(jnp.mean(squared_dist))


def backward_rmse(snake, contour):
  squared_dist = squared_distance_points_to_curve(contour, snake)
  return jnp.sqrt(jnp.mean(squared_dist))


@metric("symmetric_mae")
def SymmetricMAE(snake, contour):
  return 0.5 * forward_mae(snake, contour) + 0.5 * backward_mae(snake, contour)


@metric("symmetric_rmse")
def SymmetricRMSE(snake, contour):
  return 0.5 * forward_rmse(snake, contour) + 0.5 * backward_rmse(snake, contour)


##### Architecture Specific Loss Functions #####


def stepwise_softdtw_and_aux(snake_steps, segmentation, offsets, mask, contour):
  loss_terms = stepwise(SoftDTW(0.001))(snake_steps, contour)
  loss_terms["segmentation_loss"] = bce(segmentation, mask)
  loss_terms["offset_loss"] = offset_field_loss(offsets, mask)

  return loss_terms


class AbstractDTW(nnx.Module):
  def __init__(self, bandwidth=None):
    self.bandwidth = bandwidth

  @abstractmethod
  def minimum(self, args):
    raise NotImplementedError()

  def __call__(self, terms):
    loss = jax.vmap(self.dtw)(terms["snake"], terms["contour"])
    loss = jnp.mean(loss)
    return loss, {"dtw": loss}

  def dtw(self, snake, contour):
    D = distance_matrix(snake, contour)
    # wlog: H >= W
    if D.shape[0] < D.shape[1]:
      D = D.T
    H, W = D.shape

    if self.bandwidth is not None:
      i, j = jnp.mgrid[0:H, 0:W]
      D = jnp.where(jnp.abs(i - j) > self.bandwidth, jnp.inf, D)

    y, x = jnp.mgrid[0 : W + H - 1, 0:H]
    indices = y - x
    model_matrix = jnp.where(
      (indices < 0) | (indices >= W),
      jnp.inf,
      jnp.take_along_axis(D.T, indices, axis=0),
    )

    init = (
      pad_inf(model_matrix[0], 1, 0),
      pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0),
    )

    def scan_step(carry, current_antidiagonal):
      two_ago, one_ago = carry

      diagonal = two_ago[:-1]
      right = one_ago[:-1]
      down = one_ago[1:]
      best = self.minimum(jnp.stack([diagonal, right, down], axis=-1))

      next_row = best + current_antidiagonal
      next_row = pad_inf(next_row, 1, 0)

      return (one_ago, next_row), next_row

    carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
    return carry[1][-1]


class DTW(AbstractDTW):
  __name__ = "DTW"

  def minimum(self, args):
    return jnp.min(args, axis=-1)


def make_softmin(gamma, custom_grad=True):
  """
  We need to manually define the gradient of softmin
  to ensure (1) numerical stability and (2) prevent nans from
  propagating over valid values.
  """

  def softmin_raw(array):
    return -gamma * logsumexp(array / -gamma, axis=-1)

  if not custom_grad:
    return softmin_raw

  softmin = jax.custom_vjp(softmin_raw)

  def softmin_fwd(array):
    return softmin(array), (array / -gamma,)

  def softmin_bwd(res, g):
    (scaled_array,) = res
    grad = jnp.where(
      jnp.isinf(scaled_array),
      jnp.zeros(scaled_array.shape),
      jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1),
    )
    return (grad,)

  softmin.defvjp(softmin_fwd, softmin_bwd)
  return softmin


class SoftDTW(AbstractDTW):
  """
  SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
  by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)
  """

  __name__ = "SoftDTW"

  def __init__(self, gamma=0.001, bandwidth=None):
    super().__init__(bandwidth)
    assert gamma > 0, "Gamma needs to be positive."
    self.gamma = gamma
    self.__name__ = f"SoftDTW({self.gamma})"
    self.minimum_impl = make_softmin(gamma)

  def minimum(self, args):
    return self.minimum_impl(args)


# Internals...
def squared_distance_point_to_linesegment(point, linestart, lineend):
  p = point
  b = lineend
  a = linestart

  b_a = b - a
  p_a = p - a

  t = jnp.dot(b_a, p_a) / jnp.dot(b_a, b_a)
  t = jnp.nan_to_num(jnp.clip(t, 0, 1), nan=0.0, posinf=0.0, neginf=0.0)

  dist2 = jnp.sum(jnp.square((1 - t) * a + t * b - p))

  return dist2


def squared_distance_point_to_curve(point, polyline):
  startpoints = polyline[:-1]
  endpoints = polyline[1:]

  get_squared_distances = jax.vmap(
    squared_distance_point_to_linesegment, in_axes=[None, 0, 0]
  )
  squared_distances = get_squared_distances(point, startpoints, endpoints)

  min_dist = jnp.nanmin(squared_distances)
  return jnp.where(jnp.isnan(min_dist), 0, min_dist)


def squared_distance_points_to_curve(points, polyline):
  get_point_to_curve = jax.vmap(squared_distance_point_to_curve, in_axes=[0, None])
  return get_point_to_curve(points, polyline)
