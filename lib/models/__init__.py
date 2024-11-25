from .cobra import COBRA

import jax
from inspect import signature


def get_model(config, dummy_in, seed=jax.random.PRNGKey(39)):
  model_args = config["model_args"]
  modelclass = globals()[config["model"]]
  if "vertices" in signature(modelclass).parameters:
    model_args["vertices"] = config["vertices"]
  model = modelclass(
    **model_args,
  )
  params, buffers = model.init(seed, dummy_in[:1], is_training=True)

  return model, params, buffers
