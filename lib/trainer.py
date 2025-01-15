import jax
import jax.numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from . import losses
from .config_mod import config
from .models import COBRA
from .utils import prep


class Trainer:
  def __init__(self, key):
    init_key, self.trn_key, self.val_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(init_key)
    model = COBRA(config.model, rngs=model_rngs)
    model.train()
    opt = optax.adam(1e-4, b1=0.9, b2=0.99)
    self.state = nnx.Optimizer(model, opt)

    self.loss_fn = getattr(losses, config.loss_function)()
    if config.loss_stepwise:
      self.loss_fn = losses.StepwiseLoss(self.loss_fn)

    self.loss_fn.add_metric(losses.SymmetricMAE())
    self.loss_fn.add_metric(losses.SymmetricRMSE())
    self.loss_fn.add_metric(losses.L1())
    self.loss_fn.add_metric(losses.L2())
    self.loss_fn.add_metric(losses.Huber())

    self.checkpointer = ocp.PyTreeCheckpointer()

    if config.resume_from is not None:
      print(f"Resuming from {config.resume_from}")
      self.load_state(config.resume_from)

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

  def save_state(self, path):
    graphdef, state = nnx.split(self.state)
    state = state.flat_state()

    for key_path in list(state.keys()):
      if state[key_path].type == nnx.RngKey:
        # Convert the RNG key into an array of uint32 numbers
        uint32_array = jax.random.key_data(state[key_path].value)
        # Replace the RNG key in the model state with the array
        state[key_path] = nnx.VariableState(type=nnx.Param, value=uint32_array)

    self.checkpointer.save(path, state)

  def load_state(self, path):
    graphdef, state = nnx.split(self.state)
    loaded_state = self.checkpointer.restore(path)
    flat_state = state.flat_state()

    for key_path in list(flat_state.keys()):
      if flat_state[key_path].type == nnx.RngKey:
        # Convert the RNG key into an array of uint32 numbers
        loaded_state[key_path] = nnx.RngKey(loaded_state[key_path].value)

    self.state = nnx.merge(graphdef, state.from_flat_state(loaded_state))


@nnx.jit
def _train_step_jit(state, batch, key, loss_fn):
  img, contour = prep(batch, key)

  batch = prep(batch, key)

  def get_loss(model):
    terms = model(img, dropout_rate=0.5)
    terms["contour"] = contour
    loss, metrics = loss_fn(terms, metric_scale=img.shape[1] / 2)
    metrics["loss"] = loss
    return loss, metrics

  gradients, metrics = nnx.grad(get_loss, has_aux=True)(state.model)
  state.update(gradients)
  return metrics


@nnx.jit
def _test_step_jit(state, batch, key, loss_fn):
  img, contour = prep(batch)
  terms = state.model(img, dropout_rate=0.5)
  terms["contour"] = contour
  _, metrics = loss_fn(terms, metric_scale=img.shape[1] / 2)

  return terms, metrics
