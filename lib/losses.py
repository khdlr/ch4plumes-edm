import jax
import jax.numpy as jnp
from flax import nnx
import optax

from abc import abstractmethod


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
        metric = jax.vmap(fun)(terms["pred"], terms["target"])
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
    for step in range(1, len(terms["pred_steps"])):
      total, out_terms = self.loss_fn(
        {"pred": terms["pred_steps"][step], "target": terms["target"]}
      )
      totals.append(total)
      out.update({f"{k}_step{step}": out_terms[k] for k in out_terms})
    return sum(totals), out


##### Generic Loss Functions ####
@metric("l2")
def L2(pred, target):
  loss = jnp.sum(jnp.square(pred - target), axis=-1)
  loss = jnp.mean(loss)
  return loss


@metric("l1")
def L1(pred, target):
  loss = jnp.sum(jnp.abs(pred - target), axis=-1)
  return loss


@metric("huber")
def Huber(pred, target):
  loss = optax.huber_loss(pred, target, delta=0.05)
  return loss
