import jax
import jax.numpy as jnp
from flax import nnx
import flax.nnx.nn.linear
from flax.nnx import nn
from einops import rearrange
from ..nnutils import MCDropout

flax.nnx.nn.linear.default_kernel_init = nnx.initializers.truncated_normal(0.02)


## Ported from pytorch:
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py


class Mlp(nnx.Module):
  def __init__(
    self,
    in_features,
    hidden_features=None,
    out_features=None,
    act_layer=jax.nn.gelu,
    drop=0.0,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
    self.act = act_layer
    self.fc2 = nnx.Linear(hidden_features, out_features, rngs=rngs)
    self.drop = MCDropout(rngs=rngs)

  def __call__(self, x, dropout_rate):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x, dropout_rate=dropout_rate)
    x = self.fc2(x)
    x = self.drop(x, dropout_rate=dropout_rate)
    return x


def window_partition(x, window_size):
  """
  Args:
      x: (B, H, W, C)
      window_size (int): window size

  Returns:
      windows: (num_windows*B, window_size, window_size, C)
  """
  windows = rearrange(
    x, "B (H Hw) (W Ww) C -> (B H W) Hw Ww C", Hw=window_size, Ww=window_size
  )
  return windows


def window_reverse(windows, window_size, H, W):
  """
  Args:
      windows: (num_windows*B, window_size, window_size, C)
      window_size (int): Window size
      H (int): Height of image
      W (int): Width of image

  Returns:
      x: (B, H, W, C)
  """
  x = rearrange(
    windows,
    "(B H W) Hw Ww C -> B (H Hw) (W Ww) C",
    Hw=window_size,
    Ww=window_size,
    H=H // window_size,
    W=W // window_size,
  )
  return x


class WindowAttention(nnx.Module):
  r"""Window based multi-head self attention (W-MSA) module with relative position bias.
  It supports both of shifted and non-shifted window.

  Args:
      dim (int): Number of input channels.
      window_size (tuple[int]): The height and width of the window.
      num_heads (int): Number of attention heads.
      qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
      attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
      proj_drop (float, optional): Dropout ratio of output. Default: 0.0
      pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
  """

  def __init__(
    self,
    dim,
    window_size,
    num_heads,
    qkv_bias=True,
    attn_drop=0.0,
    proj_drop=0.0,
    pretrained_window_size=[0, 0],
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.dim = dim
    self.window_size = window_size  # Wh, Ww
    self.pretrained_window_size = pretrained_window_size
    self.num_heads = num_heads

    self.logit_scale = nnx.Param(
      jnp.log(10 * jnp.ones((1, num_heads, 1, 1))), requires_grad=True
    )

    # mlp to generate continuous relative position bias
    self.cpb_mlp = nnx.Sequential(
      nnx.Linear(2, 512, use_bias=True, rngs=rngs),
      jax.nn.relu,
      nnx.Linear(512, num_heads, use_bias=False, rngs=rngs),
    )

    # get relative_coords_table
    relative_coords_h = jnp.arange(
      -(self.window_size[0] - 1), self.window_size[0], dtype=jnp.float32
    )
    relative_coords_w = jnp.arange(
      -(self.window_size[1] - 1), self.window_size[1], dtype=jnp.float32
    )
    relative_coords_table = jnp.stack(
      jnp.meshgrid(relative_coords_h, relative_coords_w), axis=-1
    )[jnp.newaxis]  # 1, 2*Wh-1, 2*Ww-1, 2
    if pretrained_window_size[0] > 0:
      relative_coords_table = relative_coords_table / rearrange(
        jnp.array(pretrained_window_size) - 1, "C -> 1 1 1 C"
      )
    else:
      relative_coords_table = relative_coords_table / rearrange(
        jnp.array(self.window_size) - 1, "C -> 1 1 1 C"
      )
    relative_coords_table *= 8  # normalize to -8, 8
    relative_coords_table = (
      jnp.sign(relative_coords_table)
      * jnp.log2(jnp.abs(relative_coords_table) + 1.0)
      / jnp.log2(8)
    )

    self.relative_coords_table = nnx.BatchStat(relative_coords_table)

    # get pair-wise relative position index for each token inside the window
    coords_h = jnp.arange(self.window_size[0])
    coords_w = jnp.arange(self.window_size[1])
    coords = jnp.stack(jnp.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
    coords_flatten = rearrange(coords, "C H W -> C (H W)")  # 2, Wh*Ww
    relative_coords = (
      coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww
    # relative_coords = rearrange(relative_coords, "C H W -> H W C")  # Wh*Ww, Wh*Ww, 2
    ry, rx = relative_coords
    ry = ry + (self.window_size[0] - 1)
    rx = rx + (self.window_size[1] - 1)
    ry = ry * (2 * self.window_size[1] - 1)
    relative_coords = jnp.stack([ry, rx], axis=-1)
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    self.relative_position_index = nnx.BatchStat(relative_position_index)

    self.qkv = nnx.Linear(dim, dim * 3, use_bias=False, rngs=rngs)
    if qkv_bias:
      self.q_bias = nnx.Param(jnp.zeros(dim))
      self.v_bias = nnx.Param(jnp.zeros(dim))
    else:
      self.q_bias = None
      self.v_bias = None
    self.attn_drop = MCDropout(rngs=rngs)
    self.proj = nnx.Linear(dim, dim, rngs=rngs)
    self.proj_drop = MCDropout(rngs=rngs)

  def __call__(self, x, dropout_rate, mask=None):
    """
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
    """
    B_, N, C = x.shape
    qkv_bias = None
    if self.q_bias is not None:
      qkv_bias = jnp.concatenate(
        [self.q_bias.value, jnp.zeros_like(self.v_bias.value), self.v_bias.value]
      )
    qkv = self.qkv(x) + qkv_bias[jnp.newaxis, jnp.newaxis, :]
    q, k, v = rearrange(qkv, "B N (tr H C) -> tr B H N C", tr=3, H=self.num_heads)

    # cosine attention
    attn = (q / jnp.linalg.norm(q, axis=-1, keepdims=True)) @ (
      k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    ).swapaxes(-2, -1)
    logit_scale = jnp.exp(
      jnp.clip(self.logit_scale.value, max=jnp.log(jnp.array(1.0 / 0.01)))
    )
    attn = attn * logit_scale

    relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).reshape(
      -1, self.num_heads
    )
    relative_position_bias = relative_position_bias_table[
      self.relative_position_index.reshape(-1)
    ].reshape(
      self.window_size[0] * self.window_size[1],
      self.window_size[0] * self.window_size[1],
      -1,
    )  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.transpose(2, 0, 1)
    relative_position_bias = 16 * jax.nn.sigmoid(relative_position_bias)
    attn = attn + relative_position_bias[jnp.newaxis]

    if mask is not None:
      nW = mask.shape[0]
      attn = (
        attn.reshape(B_ // nW, nW, self.num_heads, N, N)
        + mask[jnp.newaxis, :, jnp.newaxis]
      )
      attn = attn.reshape(-1, self.num_heads, N, N)
      attn = jax.nn.softmax(attn, axis=-1)
    else:
      attn = jax.nn.softmax(attn, axis=-1)

    attn = self.attn_drop(attn, dropout_rate=dropout_rate)

    x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x, dropout_rate=dropout_rate)
    return x

  def extra_repr(self) -> str:
    return (
      f"dim={self.dim}, window_size={self.window_size}, "
      f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
    )


class SwinTransformerBlock(nnx.Module):
  r"""Swin Transformer Block.

  Args:
      dim (int): Number of input channels.
      input_resolution (tuple[int]): Input resulotion.
      num_heads (int): Number of attention heads.
      window_size (int): Window size.
      shift_size (int): Shift size for SW-MSA.
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
      qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
      drop (float, optional): Dropout rate. Default: 0.0
      attn_drop (float, optional): Attention dropout rate. Default: 0.0
      act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
      norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
      pretrained_window_size (int): Window size in pre-training.
  """

  def __init__(
    self,
    dim,
    input_resolution,
    num_heads,
    window_size=8,
    shift_size=0,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop=0.0,
    attn_drop=0.0,
    act_layer=jax.nn.gelu,
    norm_layer=nnx.LayerNorm,
    pretrained_window_size=0,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.dim = dim
    self.input_resolution = input_resolution
    self.num_heads = num_heads
    self.window_size = window_size
    self.shift_size = shift_size
    self.mlp_ratio = mlp_ratio
    if min(self.input_resolution) <= self.window_size:
      # if window size is larger than input resolution, we don't partition windows
      self.shift_size = 0
      self.window_size = min(self.input_resolution)
    assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

    self.norm1 = norm_layer(dim, rngs=rngs, scale_init=nnx.initializers.zeros)
    self.attn = WindowAttention(
      dim,
      window_size=(self.window_size, self.window_size),
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      attn_drop=attn_drop,
      proj_drop=drop,
      pretrained_window_size=(pretrained_window_size, pretrained_window_size),
      rngs=rngs,
    )

    # TODO: Ignoring DropPath for now...
    # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    self.norm2 = norm_layer(dim, rngs=rngs, scale_init=nnx.initializers.zeros)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(
      in_features=dim,
      hidden_features=mlp_hidden_dim,
      act_layer=act_layer,
      drop=drop,
      rngs=rngs,
    )

    if self.shift_size > 0:
      # calculate attention mask for SW-MSA
      H, W = self.input_resolution
      img_mask = jnp.zeros((1, H, W, 1))  # 1 H W 1
      h_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      w_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      cnt = 0
      for h in h_slices:
        for w in w_slices:
          img_mask = img_mask.at[:, h, w, :].set(cnt)
          cnt += 1

      mask_windows = window_partition(
        img_mask, self.window_size
      )  # nW, window_size, window_size, 1
      mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
      attn_mask = mask_windows[:, jnp.newaxis] - mask_windows[:, :, jnp.newaxis]
      attn_mask = jnp.where(attn_mask == 0, 0.0, -100.0)
      self.attn_mask = nnx.BatchStat(attn_mask)
    else:
      self.attn_mask = None

  def __call__(self, x, dropout_rate):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = x.reshape(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
      shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
    else:
      shifted_x = x

    # partition windows
    x_windows = window_partition(
      shifted_x, self.window_size
    )  # nW*B, window_size, window_size, C
    x_windows = x_windows.reshape(
      -1, self.window_size * self.window_size, C
    )  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(
      x_windows,
      mask=self.attn_mask,
      dropout_rate=dropout_rate,
    )  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
      x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
    else:
      x = shifted_x
    x = x.reshape(B, H * W, C)
    x = shortcut + self.norm1(x)

    # FFN
    x = x + self.norm2(self.mlp(x, dropout_rate=dropout_rate))

    return x


class PatchMerging(nnx.Module):
  r"""Patch Merging Layer.

  Args:
      input_resolution (tuple[int]): Resolution of input feature.
      dim (int): Number of input channels.
      norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
  """

  def __init__(
    self, input_resolution, dim, norm_layer=nnx.LayerNorm, *, rngs: nnx.Rngs
  ):
    super().__init__()
    self.input_resolution = input_resolution
    self.dim = dim
    self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False, rngs=rngs)
    self.norm = norm_layer(2 * dim, rngs=rngs)

  def __call__(self, x):
    """
    x: B, H*W, C
    """
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

    x = x.reshape(B, H, W, C)

    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = jnp.concatenate([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
    x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

    x = self.reduction(x)
    x = self.norm(x)

    return x


class BasicLayer(nnx.Module):
  """A basic Swin Transformer layer for one stage.

  Args:
      dim (int): Number of input channels.
      input_resolution (tuple[int]): Input resolution.
      depth (int): Number of blocks.
      num_heads (int): Number of attention heads.
      window_size (int): Local window size.
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
      qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
      drop (float, optional): Dropout rate. Default: 0.0
      attn_drop (float, optional): Attention dropout rate. Default: 0.0
      norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
      downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
      use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
      pretrained_window_size (int): Local window size in pre-training.
  """

  def __init__(
    self,
    dim,
    input_resolution,
    depth,
    num_heads,
    window_size,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop=0.0,
    attn_drop=0.0,
    norm_layer=nnx.LayerNorm,
    downsample=None,
    use_checkpoint=False,
    pretrained_window_size=0,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    self.dim = dim
    self.input_resolution = input_resolution
    self.depth = depth
    self.use_checkpoint = use_checkpoint

    # build blocks
    self.blocks = [
      SwinTransformerBlock(
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0 if (i % 2 == 0) else window_size // 2,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop,
        attn_drop=attn_drop,
        norm_layer=norm_layer,
        pretrained_window_size=pretrained_window_size,
        rngs=rngs,
      )
      for i in range(depth)
    ]

    # patch merging layer
    if downsample is not None:
      self.downsample = downsample(
        input_resolution, dim=dim, norm_layer=norm_layer, rngs=rngs
      )
    else:
      self.downsample = None

  def __call__(self, x, dropout_rate):
    for blk in self.blocks:
      if self.use_checkpoint:
        blk = jax.remat(blk)
      x = blk(x, dropout_rate=dropout_rate)

    if self.downsample is not None:
      x = self.downsample(x)
    return x

  def extra_repr(self) -> str:
    return (
      f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    )

  def flops(self):
    flops = 0
    for blk in self.blocks:
      flops += blk.flops()
    if self.downsample is not None:
      flops += self.downsample.flops()
    return flops


class PatchEmbed(nnx.Module):
  r"""Image to Patch Embedding

  Args:
      img_size (int): Image size.  Default: 224.
      patch_size (int): Patch token size. Default: 4.
      in_chans (int): Number of input image channels. Default: 3.
      embed_dim (int): Number of linear projection output channels. Default: 96.
      norm_layer (nn.Module, optional): Normalization layer. Default: None
  """

  def __init__(
    self,
    img_size=512,
    patch_size=4,
    in_chans=4,
    embed_dim=96,
    norm_layer=None,
    *,
    rngs: nnx.Rngs,
  ):
    super().__init__()
    img_size = (img_size, img_size)
    patch_size = (patch_size, patch_size)
    patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    self.img_size = img_size
    self.patch_size = patch_size
    self.patches_resolution = patches_resolution
    self.num_patches = patches_resolution[0] * patches_resolution[1]

    self.in_chans = in_chans
    self.embed_dim = embed_dim

    self.proj = nnx.Conv(
      in_chans,
      embed_dim,
      kernel_size=patch_size,
      strides=patch_size,
      rngs=rngs,
    )
    if norm_layer is not None:
      self.norm = norm_layer(embed_dim, rngs=rngs)
    else:
      self.norm = None

  def __call__(self, x):
    B, H, W, C = x.shape
    # FIXME look at relaxing size constraints
    assert (
      H == self.img_size[0] and W == self.img_size[1]
    ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = rearrange(self.proj(x), "B H W C -> B (H W) C")  # B Ph*Pw C
    if self.norm is not None:
      x = self.norm(x)
    return x


class SwinTransformerV2(nnx.Module):
  r"""Swin Transformer
      A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/pdf/2103.14030

  Args:
      img_size (int | tuple(int)): Input image size. Default 224
      patch_size (int | tuple(int)): Patch size. Default: 4
      in_chans (int): Number of input image channels. Default: 3
      num_classes (int): Number of classes for classification head. Default: 1000
      embed_dim (int): Patch embedding dimension. Default: 96
      depths (tuple(int)): Depth of each Swin Transformer layer.
      num_heads (tuple(int)): Number of attention heads in different layers.
      window_size (int): Window size. Default: 7
      mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
      qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
      drop_rate (float): Dropout rate. Default: 0
      attn_drop_rate (float): Attention dropout rate. Default: 0
      norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
      ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
      patch_norm (bool): If True, add normalization after patch embedding. Default: True
      use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
      pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
  """

  def __init__(
    self,
    img_size=512,
    patch_size=4,
    in_chans=4,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    norm_layer=nnx.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    pretrained_window_sizes=[0, 0, 0, 0],
    *,
    rngs: nnx.Rngs,
    **kwargs,
  ):
    super().__init__()

    self.img_size = img_size
    self.num_classes = num_classes
    self.num_layers = len(depths)
    self.embed_dim = embed_dim
    self.ape = ape
    self.patch_norm = patch_norm
    self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
    self.mlp_ratio = mlp_ratio
    trunc_normal = nnx.initializers.truncated_normal(0.02)

    # split image into non-overlapping patches
    self.patch_embed = PatchEmbed(
      img_size=img_size,
      patch_size=patch_size,
      in_chans=in_chans,
      embed_dim=embed_dim,
      norm_layer=norm_layer if self.patch_norm else None,
      rngs=rngs,
    )
    num_patches = self.patch_embed.num_patches
    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution

    # absolute position embedding
    if self.ape:
      self.absolute_pos_embed = nnx.Param(
        trunc_normal(rngs(), [1, num_patches, embed_dim])
      )

    self.pos_drop = MCDropout(rngs=rngs)

    # build layers
    self.layers = []
    for i_layer in range(self.num_layers):
      layer = BasicLayer(
        dim=int(embed_dim * 2**i_layer),
        input_resolution=(
          patches_resolution[0] // (2**i_layer),
          patches_resolution[1] // (2**i_layer),
        ),
        depth=depths[i_layer],
        num_heads=num_heads[i_layer],
        window_size=window_size,
        mlp_ratio=self.mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
        norm_layer=norm_layer,
        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        use_checkpoint=use_checkpoint,
        pretrained_window_size=pretrained_window_sizes[i_layer],
        rngs=rngs,
      )
      self.layers.append(layer)

    self.norm = norm_layer(self.num_features, rngs=rngs)

  def __call__(self, x, dropout_rate=0.0):
    x = self.patch_embed(x)
    if self.ape:
      x = x + self.absolute_pos_embed
    x = self.pos_drop(x, dropout_rate=dropout_rate)

    for i, layer in enumerate(self.layers):
      with jax.profiler.TraceAnnotation(f"swint_layer{i}"):
        x = layer(x, dropout_rate=dropout_rate)
        if i == 1:
          skip = x

    x = self.norm(x)  # B L C
    x = rearrange(
      x, "B (H W) C -> B H W C", H=self.img_size // 32, W=self.img_size // 32
    )
    skip = rearrange(
      skip, "B (H W) C -> B H W C", H=self.img_size // 16, W=self.img_size // 16
    )
    return [skip, x]
