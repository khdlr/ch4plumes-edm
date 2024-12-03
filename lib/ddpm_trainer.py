import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from . import losses
from .config_mod import config
from .models import COBRA
from .utils import prep

ddpm_betas = []


def q_sample(x, t, noise, ddpm_params):
  sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][t, None, None, None]
  sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][t, None, None, None]
  x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

  return x_t


class DDPMTrainer:
  def __init__(self, key):
    init_key, self.trn_key, self.val_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(init_key)
    model = COBRA(config.model, rngs=model_rngs)
    model.train()
    opt = optax.adamw(1e-3, weight_decay=1e-5)
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

  def train_step(self, batch):
    self.state.model.train()
    self.trn_key, key = jax.random.split(self.trn_key)
    data = (batch["image"], batch["dem"], batch["contour"])
    return _train_step_jit(self.state, data, key, self.loss_fn, self.ddpm_params)

  def test_step(self, batch):
    self.state.model.eval()
    self.val_key, key = jax.random.split(self.val_key)
    data = (batch["image"], batch["dem"], batch["contour"])
    img, contour = prep(data, key)
    sampling_terms = self.sample(img)

    terms = {
      "contour": contour,
      "snake": sampling_terms["prediction"],
      "snake_steps": sampling_terms["sampling_steps"],
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
    init_key, *sample_keys = jax.random.split(key, self.timesteps + 1)

    B, *_ = imagery.shape
    x = jax.random.normal(init_key, [B, 128, 2])
    features = jax.jit(self.state.model.backbone)(imagery)
    sampling_steps = []
    # sample step
    for key, t in zip(sample_keys, reversed(jnp.arange(self.timesteps))):
      x, x0 = _sample_step(self.state, x, features, key, self.ddpm_params, t)
      sampling_steps.append(x0)
    return {"prediction": sampling_steps[-1], "sampling_steps": sampling_steps}

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
def _train_step_jit(state, batch, key, loss_fn, ddpm_params):
  aug_key, t_key, noise_key = jax.random.split(key, 3)
  img, contour = prep(batch, aug_key)
  B, H, W, C = img.shape
  B, V, _ = contour.shape

  batched_t = jax.random.randint(
    t_key, shape=(B,), dtype=jnp.int32, minval=0, maxval=len(ddpm_betas)
  )
  target = noise = jax.random.normal(noise_key, contour.shape)

  sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][batched_t, None, None]
  sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][batched_t, None, None]
  x_t = sqrt_alpha_bar * contour + sqrt_1m_alpha_bar * noise

  # TODO: Check conditioning code in https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/train.py#L266C1-L266C23
  def get_loss(model):
    pred = model.ddpm_forward(img, x_t)
    terms = {"contour": contour, "snake": pred}
    loss, metrics = loss_fn(terms, metric_scale=img.shape[1] / 2)
    metrics["loss"] = loss
    return loss, metrics

  gradients, metrics = nnx.grad(get_loss, has_aux=True)(state.model)
  state.update(gradients)
  return metrics


@nnx.jit
def _sample_step(state, vertices, features, key, ddpm_params, t):
  batched_t = jnp.ones((vertices.shape[0],), dtype=jnp.int32) * t

  pred = state.model.head(vertices, features)

  # Recover x0
  sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][batched_t, None, None]
  alpha_bar = ddpm_params["alphas_bar"][batched_t, None, None]
  x0 = 1.0 / sqrt_alpha_bar * vertices - jnp.sqrt(1.0 / alpha_bar - 1) * pred

  beta = ddpm_params["betas"][batched_t, None, None]
  alpha = ddpm_params["alphas"][batched_t, None, None]
  alpha_bar_last = ddpm_params["alphas_bar"][batched_t - 1, None, None]
  sqrt_alpha_bar_last = ddpm_params["sqrt_alphas_bar"][batched_t - 1, None, None]

  # only needed when t > 0
  coef_x0 = beta * sqrt_alpha_bar_last / (1.0 - alpha_bar)
  coef_xt = (1.0 - alpha_bar_last) * jnp.sqrt(alpha) / (1 - alpha_bar)
  posterior_mean = coef_x0 * x0 + coef_xt * vertices

  posterior_variance = beta * (1 - alpha_bar_last) / (1.0 - alpha_bar)
  posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))

  x = posterior_mean + jnp.exp(0.5 * posterior_log_variance) * jax.random.normal(
    key, x0.shape
  )

  return x, x0
