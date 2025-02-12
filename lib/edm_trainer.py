import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from functools import partial
from einops import repeat
from pathlib import Path

from . import losses, models
from .config_mod import config
from .utils import prep

num_steps = 18


class EDMTrainer:
  def __init__(self, key):
    init_key, self.trn_key, self.val_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(init_key)
    model = getattr(models, config.model_type)(in_dim=1, out_dim=1, rngs=model_rngs)
    model.train()
    EPOCH = 1024 * 1024 // config.batch_size
    lr_schedule = optax.warmup_constant_schedule(
      init_value=1e-7,
      peak_value=1e-5,
      warmup_steps=EPOCH,
    )
    opt = optax.chain(
      optax.clip(1.0), optax.adamw(lr_schedule, b1=0.9, b2=0.99, weight_decay=1e-4)
    )
    self.state = nnx.Optimizer(model, opt)

    self.loss_fn = getattr(losses, config.loss_function)()
    self.loss_fn.add_metric(losses.L1())
    self.loss_fn.add_metric(losses.L2())
    self.loss_fn.add_metric(losses.Huber())

    self.edm_params = {
      "P_mean": -1.2,
      "P_std": 1.2,
      "sigma_data": 0.5,
      "sigma_min": 0.002,
      "sigma_max": 80.0,
      "rho": 7,
      "S_churn": 0.0,
      "S_min": 0.0,
      "S_noise": 1.0,
    }

    self.checkpointer = ocp.PyTreeCheckpointer()

    if config.resume_from is not None:
      print(f"Resuming from {config.resume_from}")
      self.load_state(config.resume_from)

  def train_step(self, batch):
    self.state.model.train()
    self.trn_key, key = jax.random.split(self.trn_key)
    data = (batch["CH4PLUME"], batch["U10"], batch["V10"])
    return _train_step_jit(self.state, data, key, self.loss_fn, self.edm_params)

  def test_step(self, batch):
    self.state.model.eval()
    self.val_key, key = jax.random.split(self.val_key)
    data = (batch["CH4PLUME"], batch["U10"], batch["V10"])
    img, contour = prep(data)
    sampling_terms = self.sample(img)

    terms = {
      "contour": contour,
      "snake": sampling_terms["prediction"],
      "snake_steps": list(sampling_terms["steps"]),
    }
    loss, metrics = jax.jit(self.loss_fn)(terms)
    metrics["loss"] = loss
    return terms, metrics

  # Adapted from https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/sampling.py
  # Original Author: YiYi Xu (https://github.com/yiyixuxu)
  def sample(self, imagery, *, key=None):
    self.state.model.eval()
    if key is None:
      self.val_key, key = jax.random.split(self.val_key)
    return _sample_jit(self.state, imagery, self.edm_params, key)

  def save_state(self, path):
    keys, state = nnx.state(self.state, nnx.RngKey, ...)
    self.checkpointer.save(Path(path).absolute(), state)

  def load_state(self, path):
    graphdef, state = nnx.split(self.state)
    keys, state = nnx.state(self.state, nnx.RngKey, ...)
    loaded = self.checkpointer.restore(Path(path).absolute(), state)
    self.state = nnx.merge(graphdef, loaded, keys)


# Adapted from https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/train.py
# Original Author: YiYi Xu (https://github.com/yiyixuxu)
@nnx.jit
def _train_step_jit(state, batch, key, loss_fn, edm_params):
  aug_key, t_key, t_key_u, noise_key = jax.random.split(key, 4)
  plume, u10, v10 = prep(batch, aug_key)
  B = plume.shape[0]

  rnd_normal = jax.random.normal(t_key, shape=(B, 1, 1, 1))
  sigma = jnp.exp(rnd_normal * edm_params["P_std"] + edm_params["P_mean"])
  weight = (sigma**2 + edm_params["sigma_data"] ** 2) / (
    sigma * edm_params["sigma_data"]
  ) ** 2
  noise = jax.random.normal(noise_key, plume.shape) * sigma

  def get_loss(model):
    # Conditional Generation
    D_yn = model(plume + noise, sigma=sigma)
    loss = jnp.mean(weight * ((D_yn - plume) ** 2))
    terms = {"target": plume, "pred": D_yn}
    _, metrics = loss_fn(terms)
    metrics["loss"] = loss
    return metrics["loss"], metrics

  gradients, metrics = nnx.grad(get_loss, has_aux=True)(state.model)
  state.update(gradients)
  return metrics


@nnx.jit
def _sample_step(state, vertices, features, step_data, edm_params):
  # Main sampling loop from EDM code, transferred into a scan-able jax function
  S_min = edm_params["S_min"]
  S_churn = edm_params["S_churn"]
  S_noise = edm_params["S_noise"]

  key, t_cur, t_next, do_2nd = step_data
  x_cur = vertices

  # Increase noise temporarily
  gamma = jnp.where(
    S_min <= t_cur, jnp.minimum(S_churn / num_steps, jnp.sqrt(2) - 1.0), 0.0
  )
  t_hat = t_cur + gamma * t_cur
  x_hat = x_cur + jnp.sqrt(t_hat**2 - t_cur**2) * S_noise * jax.random.normal(
    key, x_cur.shape
  )

  # Euler step.
  denoised = state.model.head(x_hat, features=features, sigma=t_cur)
  d_cur = (x_hat - denoised) / t_hat
  x_next = x_hat + (t_next - t_hat) * d_cur

  # 2nd order correction (without branching in python, to remain scan-able)
  denoised = state.model.head(x_next, features=features, sigma=t_next)
  d_prime = (x_next - denoised) / t_next
  x_next = jnp.where(
    do_2nd, x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime), x_next
  )
  return x_next, x_next  # Double-return needed for scan


@nnx.jit
def _sample_jit(state, imagery, edm_params, key):
  B, *_ = imagery.shape
  # Extract config:
  sigma_max = edm_params["sigma_max"]
  sigma_min = edm_params["sigma_min"]
  rho = edm_params["rho"]

  step_indices = jnp.arange(num_steps)
  t_steps = (
    sigma_max ** (1 / rho)
    + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
  ) ** rho
  t_steps = jnp.concatenate([t_steps, jnp.zeros((1,))])  # t_N = 0
  t_steps = repeat(t_steps, "T -> T B 1 1", B=B)
  # Main sampling loop.
  init_key, sample_key = jax.random.split(key)
  x_init = normal(init_key, [B, 128, 2])

  sample_keys = jax.random.split(sample_key, num_steps)
  features = jax.jit(state.model.backbone)(imagery)

  def scan_step(x, key_ts_do_2nd):
    return _sample_step(state, x, features, key_ts_do_2nd, edm_params)

  t_cur = t_steps[:-1]
  t_next = t_steps[1:]
  do_2nd = t_next != t_next[-1:, :]
  step_info = (sample_keys, t_cur, t_next, do_2nd)
  _, steps = jax.lax.scan(scan_step, x_init, step_info)

  # sample step
  return {"prediction": steps[-1], "steps": steps}
