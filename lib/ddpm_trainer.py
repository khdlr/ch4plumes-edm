import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from functools import partial

from . import losses
from .config_mod import config
from .models import COBRA
from .models.snake_utils import random_bezier
from .utils import prep

ddpm_betas = []


class DDPMTrainer:
  def __init__(self, key):
    init_key, self.trn_key, self.val_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(init_key)
    model = COBRA(config.model, rngs=model_rngs)
    model.train()
    opt = optax.chain(optax.clip(1.0), optax.adam(1e-4, b1=0.9, b2=0.99))
    self.state = nnx.Optimizer(model, opt)

    self.loss_fn = getattr(losses, config.loss_function)()
    self.loss_fn.add_metric(losses.L1())
    self.loss_fn.add_metric(losses.L2())
    self.loss_fn.add_metric(losses.Huber())

    self.checkpointer = ocp.PyTreeCheckpointer()

    self.n_vertices = config.model.vertices
    self.timesteps = 200
    # DDPM Params initialization taken from https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/utils.py
    # Original Author: YiYi Xu (https://github.com/yiyixuxu)
    s = 0.008
    max_beta = 0.999
    ts = jnp.linspace(0, 1, self.timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = jnp.clip(betas, 0, max_beta)

    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar = jnp.sqrt(1.0 - alphas_bar)
    # p2_loss_weight = (1 + alphas_bar / (1 - alphas_bar)) ** -0.0

    self.ddpm_params = {
      "betas": betas,
      "alphas": alphas,
      "alphas_bar": alphas_bar,
      "sqrt_alphas_bar": sqrt_alphas_bar,
      "sqrt_1m_alphas_bar": sqrt_1m_alphas_bar,
      # "p2_loss_weight": p2_loss_weight,
    }

  def train_step(self):
    self.state.model.train()
    self.trn_key, key = jax.random.split(self.trn_key)
    # data = (batch["image"], batch["dem"], batch["contour"])
    # return _train_step_jit(self.state, data, key, self.loss_fn, self.ddpm_params)
    return _train_step_jit(self.state, key, self.loss_fn, self.ddpm_params)

  def test_step(self, batch):
    self.state.model.eval()
    self.val_key, key = jax.random.split(self.val_key)
    data = (batch["image"], batch["dem"], batch["contour"])
    img, contour = prep(data, key)
    sampling_terms = self.sample(img)

    terms = {
      "contour": contour,
      "snake": sampling_terms["prediction"],
      "snake_steps": list(sampling_terms["steps"][::20]),
    }
    loss, metrics = jax.jit(self.loss_fn)(terms, metric_scale=img.shape[1] / 2)
    metrics["loss"] = loss
    return terms, metrics

  # Adapted from https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/sampling.py
  # Original Author: YiYi Xu (https://github.com/yiyixuxu)
  def sample(self, imagery, *, key=None):
    self.state.model.eval()
    if key is None:
      self.val_key, key = jax.random.split(self.val_key)
    return _sample_jit(self.state, imagery, self.ddpm_params, key)

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


# Adapted from https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/train.py
# Original Author: YiYi Xu (https://github.com/yiyixuxu)
@nnx.jit
def _train_step_jit(state, key, loss_fn, ddpm_params):
  aug_key, t_key, noise_key = jax.random.split(key, 3)
  # img, contour = prep(batch, aug_key)
  B = config.batch_size
  keys = jax.random.split(aug_key, B)
  contour = jax.vmap(partial(random_bezier, vertices=128))(keys)

  batched_t = jax.random.randint(
    t_key, shape=(B,), dtype=jnp.int32, minval=0, maxval=len(ddpm_betas)
  )
  noise = jax.random.normal(noise_key, contour.shape)

  sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][batched_t, None, None]
  sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][batched_t, None, None]
  x_t = sqrt_alpha_bar * contour + sqrt_1m_alpha_bar * noise

  # TODO: Check conditioning code in https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/train.py#L266C1-L266C23
  def get_loss(model):
    # features = model.backbone(img)
    pred = model.head(x_t, features=None)
    terms = {"contour": contour, "snake": pred}
    loss, metrics = loss_fn(terms, metric_scale=1)
    metrics["loss"] = loss
    return loss, metrics

  gradients, metrics = nnx.grad(get_loss, has_aux=True)(state.model)
  state.update(gradients)
  return metrics


@nnx.jit
def _sample_step(state, vertices, features, key, ddpm_params, t):
  batched_t = jnp.ones((vertices.shape[0],), dtype=jnp.int32) * t

  pred = state.model.head(vertices, features)
  noise_pred = x0_to_noise(pred, vertices, batched_t, ddpm_params)

  # Recover x0
  sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][batched_t, None, None]
  alpha_bar = ddpm_params["alphas_bar"][batched_t, None, None]
  x0 = 1.0 / sqrt_alpha_bar * vertices - jnp.sqrt(1.0 / alpha_bar - 1) * noise_pred
  x0 = jnp.clip(x0, -1.0, 1.0)

  beta = ddpm_params["betas"][batched_t, None, None]
  alpha = ddpm_params["alphas"][batched_t, None, None]
  alpha_bar_last = ddpm_params["alphas_bar"][batched_t - 1, None, None]
  sqrt_alpha_bar_last = ddpm_params["sqrt_alphas_bar"][batched_t - 1, None, None]

  # only needed when t > 0
  coef_x0 = beta * sqrt_alpha_bar_last / (1.0 - alpha_bar)
  coef_xt = (1.0 - alpha_bar_last) * jnp.sqrt(alpha) / (1 - alpha_bar)
  posterior_mean = coef_x0 * x0 + coef_xt * vertices

  posterior_variance = beta * (1 - alpha_bar_last) / (1.0 - alpha_bar)
  x = posterior_mean + jnp.sqrt(posterior_variance) * jax.random.normal(key, x0.shape)

  return x, x


@nnx.jit
def _sample_jit(state, imagery, ddpm_params, key):
  (T,) = ddpm_params["betas"].shape
  init_key, sample_key = jax.random.split(key)
  sample_keys = jax.random.split(sample_key, T - 1)
  B, *_ = imagery.shape
  x = jax.random.normal(init_key, [B, 128, 2])
  features = None
  # features = jax.jit(state.model.backbone)(imagery)

  def scan_step(x, key_t):
    key, t = key_t
    return _sample_step(state, x, features, key, ddpm_params, t)

  _, steps = jax.lax.scan(scan_step, x, (sample_keys, jnp.arange(1, T)[::-1]))

  # sample step
  return {"prediction": steps[-1], "steps": steps}


def x0_to_noise(x0, xt, batched_t, ddpm):
  assert (
    batched_t.shape[0] == xt.shape[0] == x0.shape[0]
  )  # make sure all has batch dimension
  sqrt_alpha_bar = ddpm["sqrt_alphas_bar"][batched_t, None, None]
  alpha_bar = ddpm["alphas_bar"][batched_t, None, None]
  noise = (1.0 / sqrt_alpha_bar * xt - x0) / jnp.sqrt(1.0 / alpha_bar - 1)
  return noise
