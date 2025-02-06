import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
from einops import rearrange, repeat

from . import backbones, heads, snake_utils
from . import nnutils as nn


class COBRA(nnx.Module):
  def __init__(
    self,
    config,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.backbone = getattr(backbones, config.backbone)(c_in=3, rngs=rngs)
    self.vertices = config.vertices
    head_model = getattr(heads, config.head)
    self.head = SnakeHead(
      feature_dims=self.backbone.feature_dims(),
      head_model=head_model,
      rngs=rngs,
    )
    self.dropout = nn.ChannelDropout(rngs=rngs)
    self.rngs = rngs


class SnakeHead(nnx.Module):
  def __init__(self, feature_dims, head_model, *, rngs: nnx.Rngs):
    super().__init__()

    self.model = head_model(rngs=rngs)
    C = self.model.get_dim()

    self.init_coords = nnx.Conv(2, C, [1], rngs=rngs)
    # Initialize offset predictor with 0 -> default to no change
    self.mk_offset = nnx.Conv(
      C,
      2,
      [1],
      use_bias=False,
      kernel_init=nnx.initializers.zeros,
      rngs=rngs,
    )

    self.local_cond_proj = nnx.Linear(feature_dims[0], C, rngs=rngs)
    self.cond_attn = CrossAttentionConditioning(feature_dims[1], 8, rngs=rngs)
    self.cond_proj = nnx.Linear(feature_dims[1], C, rngs=rngs)

  def __call__(self, vertices, features, sigma, *, dropout_rate=0.0):
    sigma = jnp.log(sigma)  # Back to log scale for sigma
    # Start with coord features
    x = self.init_coords(vertices)

    if features:
      f_local, f_global = features
      x += self.cond_proj(self.cond_attn(vertices, f_global))
      feat = jax.vmap(snake_utils.sample_at_vertices, [0, 0])(vertices, f_local)
      x += self.local_cond_proj(feat)

    x = jax.nn.silu(x)
    x = self.model(x)
    offsets = self.mk_offset(x)

    return vertices + offsets


class CrossAttentionConditioning(nnx.Module):
  def __init__(self, dim, num_heads, *, rngs: nnx.Rngs):
    self.num_heads = num_heads
    self.attention = nnx.MultiHeadAttention(
      num_heads=num_heads, in_features=dim, rngs=rngs
    )
    self.q_emb = nnx.Linear(4, dim, rngs=rngs)
    self.k_emb = nnx.Linear(dim, dim, rngs=rngs)

  def __call__(self, vertex_coordinates, feature_map):
    """
    vertex_coordinates: [B T 2]
    feature_map: [B H W C]
    """
    B, T, _ = vertex_coordinates.shape
    q_x = vertex_coordinates
    t = repeat(jnp.linspace(-jnp.pi / 2, jnp.pi / 2, T), "T -> B T 1", B=B, T=T)
    q = jnp.concatenate([q_x, jnp.sin(t), jnp.cos(t)], axis=-1)
    q = self.q_emb(q)
    v = feature_map
    k = rope2d(self.k_emb(feature_map), self.num_heads)
    v = rearrange(v, "B H W C -> B (H W) C")
    k = rearrange(k, "B H W C -> B (H W) C")
    return self.attention(q, k, v, decode=False)


def rope2d(feature_map, n_heads):
  "feature_maps: (B H W C), applies rotary embedding for both dims"
  f1, f2, f3, f4 = rearrange(
    feature_map, "B H W (K S C) -> S B H W K C", K=n_heads, S=4
  )
  B, H, W, K, C = f1.shape
  xs = jnp.linspace(-1, 1, W, endpoint=False).reshape(1, 1, W, 1, 1) + 1 / W
  ys = jnp.linspace(-1, 1, H, endpoint=False).reshape(1, H, 1, 1, 1) + 1 / H

  cos_x = jnp.cos(jnp.pi / 2 * xs)
  sin_x = jnp.sin(jnp.pi / 2 * xs)
  cos_y = jnp.cos(jnp.pi / 2 * ys)
  sin_y = jnp.sin(jnp.pi / 2 * ys)

  o1 = cos_x * f1 + sin_x * f2
  o2 = -sin_x * f1 + cos_x * f2
  o3 = cos_y * f3 + sin_y * f4
  o4 = -sin_y * f3 + cos_y * f4
  out = jnp.concatenate([o1, o2, o3, o4], axis=-1)
  out = rearrange(out, "B H W K (S C) -> B H W (K S C)", B=B, H=H, K=K, C=C)
  return out
