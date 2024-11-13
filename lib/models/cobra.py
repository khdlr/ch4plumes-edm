import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from . import backbones
from . import nnutils as nn
from . import snake_utils


class COBRA:
    def __init__(
        self,
        backbone,
        vertices=64,
        model_dim=64,
        iterations=5,
        head="SnakeHead",
        coord_features=False,
        weight_sharing=True,
        stop_grad=True,
    ):
        super().__init__()
        self.backbone = getattr(backbones, backbone)
        self.model_dim = model_dim
        self.iterations = iterations
        self.coord_features = coord_features
        self.vertices = vertices
        self.stop_grad = stop_grad
        self.weight_sharing = weight_sharing
        self.head = head

    def __call__(self, imagery, is_training=False, dropout_rate=0.0):
        backbone = self.backbone()
        feature_maps = backbone(imagery, is_training, dropout_rate=dropout_rate)

        if is_training:
          feature_maps = [nn.channel_dropout(f, dropout_rate) for f in feature_maps]

        init_keys = jax.random.split(hk.next_rng_key(), imagery.shape[0])
        init_fn = partial(snake_utils.random_bezier, vertices=self.vertices)
        vertices = jax.vmap(init_fn)(init_keys)
        # vertices = jnp.zeros([imagery.shape[0], self.vertices, 2])
        steps = [vertices]

        if self.weight_sharing:
            _head = getattr(snake_utils, self.head)(self.model_dim, self.coord_features)
            head = lambda x, y: _head(x, y, dropout_rate=dropout_rate)
        else:
            head_cls = getattr(snake_utils, self.head)
            head = lambda x, y: head_cls(self.model_dim, self.coord_features)(x, y, dropout_rate=dropout_rate)

        for _ in range(self.iterations):
            if self.stop_grad:
                vertices = jax.lax.stop_gradient(vertices)
            vertices = vertices + head(vertices, feature_maps)
            steps.append(vertices)

        return {"snake_steps": steps, "snake": vertices}
