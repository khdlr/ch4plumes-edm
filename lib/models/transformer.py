## Transformer functions from https://github.com/cgarciae/nnx/blob/main/examples/07_transformer.py
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from ..config_mod import Config


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""

  def init_fn(rngs: nnx.Rngs, shape, in_axis, out_axis) -> jax.Array:
    fn = jax.nn.initializers.variance_scaling(
      scale, mode, distribution, in_axis, out_axis
    )
    return fn(rngs(), shape)

  return init_fn


dense_init = nd_dense_init(1.0, "fan_in", "truncated_normal")
embed_init = nd_dense_init(1.0, "fan_in", "normal")


def rotate_half(x):
  x1, x2 = jnp.split(x, 2, axis=-1)
  x = jnp.concatenate([-x2, x1], axis=-1)
  return x


def sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, bfloat16 rounding is catastrophic.
  sinusoid_inp = jnp.einsum(
    "i,j->ij",
    jnp.arange(length),
    rotational_frequency,
    precision=jax.lax.Precision.HIGHEST,
  )
  sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rms_norm(scale, x):
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + 1.0e-7))
  return y * scale[None, None, :]


def apply_rotary_embedding(q, k, cos, sin, n_heads, index=None):
  """Helper function to apply Rotary Embeddings."""
  q = rearrange(q, "B (T H) C -> B T H C", H=n_heads)
  k = rearrange(k, "B (T H) C -> B T H C", H=n_heads)
  batch, qlen, qheads, d = q.shape
  kbatch, klen, kheads, kd = k.shape
  if index is not None:
    qcos = jax.lax.broadcast_in_dim(cos[index, :], (batch, qlen, qheads, d), (3,))
    qsin = jax.lax.broadcast_in_dim(sin[index, :], (batch, qlen, qheads, d), (3,))
  else:
    qcos = jax.lax.broadcast_in_dim(cos[:qlen, :], (batch, qlen, qheads, d), (1, 3))
    qsin = jax.lax.broadcast_in_dim(sin[:qlen, :], (batch, qlen, qheads, d), (1, 3))
  kcos = jax.lax.broadcast_in_dim(cos[:klen, :], (batch, klen, kheads, d), (1, 3))
  ksin = jax.lax.broadcast_in_dim(sin[:klen, :], (batch, klen, kheads, d), (1, 3))
  out_q = (q * qcos) + (rotate_half(q) * qsin)
  out_k = (k * kcos) + (rotate_half(k) * ksin)

  out_q = rearrange(out_q, "B T H C -> B (T H) C", H=n_heads)
  out_k = rearrange(out_k, "B T H C -> B (T H) C", H=n_heads)
  return out_q, out_k


class MLP(nnx.Module):
  def __init__(self, embed_dim, hidden_dim, *, rngs: nnx.Rngs):
    self.Win1 = nnx.Param(dense_init(rngs, (embed_dim, hidden_dim), 0, 1))
    self.Win2 = nnx.Param(dense_init(rngs, (embed_dim, hidden_dim), 0, 1))
    self.Wout = nnx.Param(dense_init(rngs, (hidden_dim, embed_dim), 0, 1))
    # TODO: Dropout
    # self.dropout = nnx.Dropout(rngs=rngs)

  def __call__(self, x):
    h1 = jnp.einsum("bse,eh->bsh", x, self.Win1)
    h2 = jnp.einsum("bse,eh->bsh", x, self.Win2)
    h = jax.nn.gelu(h1) * h2
    o = jnp.einsum("bsh,he->bse", h, self.Wout)
    return o


class TransformerBlock(nnx.Module):
  def __init__(self, embed_dim, hidden_dim, *, rngs: nnx.Rngs):
    self.attn = nnx.MultiHeadAttention(
      num_heads=8, in_features=embed_dim, qkv_features=embed_dim, rngs=rngs
    )
    self.mlp = MLP(embed_dim, hidden_dim, rngs=rngs)
    self.scale1 = nnx.Param(jnp.ones((embed_dim,)))
    self.scale2 = nnx.Param(jnp.ones((embed_dim,)))

  def __call__(self, input):
    x = rms_norm(self.scale1, input)
    q = k = v = x
    sin, cos = sine_table(q.shape[-1], max(q.shape[1], k.shape[1]))
    q, k = apply_rotary_embedding(q, k, cos, sin, n_heads=8)
    x = self.attn(q, k, v)
    x = x + input
    y = rms_norm(self.scale2, x)
    y = self.mlp(y)
    return y + x


class Transformer(nnx.Module):
  def __init__(self, embed_dim, n_blocks, *, rngs: nnx.Rngs):
    self.blocks = [
      TransformerBlock(embed_dim, 3 * embed_dim, rngs=rngs) for _ in range(n_blocks)
    ]

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
    return x
