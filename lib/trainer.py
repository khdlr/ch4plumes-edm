import jax
import jax.numpy as jnp
from flax import nnx
import optax
from .config_mod import config
from .utils import prep
from . import losses
from .models import Model
import pandas as pd
from einops import rearrange


class Trainer:
  def __init__(self, key):
    init_key, self.trn_key, self.val_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(init_key)
    model = Model(config, rngs=model_rngs)
    model.train()
    opt = optax.adamw(1e-3, weight_decay=1e-5)
    self.state = nnx.Optimizer(model, opt)

    self.loss_fn = getattr(losses, config.loss_function)
    if config.loss_stepwise:
      self.loss_fn = losses.StepwiseLoss(self.loss_fn)

    self.loss_fn.add_metric(losses.SymmetricMAE())
    self.loss_fn.add_metric(losses.SymmetricRMSE())
    self.loss_fn.add_metric(losses.L1())
    self.loss_fn.add_metric(losses.L2())
    self.loss_fn.add_metric(losses.Huber())

  def train_step(self, batch):
    self.state.model.train()
    self.trn_key, key = jax.random.split(self.trn_key)
    data = (batch["image"], batch["dem"], batch["contour"])
    return _train_step_jit(self.state, data, key, self.loss_fn)

  def test_step(self, batch):
    self.state.model.eval()
    self.val_key, key = jax.random.split(self.val_key)
    data = (batch["image"], batch["dem"], batch["contour"])
    return _test_step_jit(self.state, data, key, self.loss_fn)


@nnx.jit
def _train_step_jit(state, batch, key, loss_fn):
  aug_key, model_key = jax.random.split(key)
  img, contour = prep(batch, aug_key)

  batch = prep(batch, key)

  def get_loss(model):
    terms = model(img, model_key, dropout_rate=0.5)
    terms["contour"] = contour
    loss, metrics = loss_fn(terms)
    metrics["loss"] = loss
    return loss, metrics

  gradients, metrics = nnx.grad(get_loss, has_aux=True)(state.model)
  state.update(gradients)
  return metrics


@nnx.jit
def _test_step_jit(state, batch, key, loss_fn):
  img, contour = prep(batch)
  terms = state.model(img, key, dropout_rate=0.5)
  terms["contour"] = contour
  _, metrics = loss_fn(terms)

  return terms, metrics
