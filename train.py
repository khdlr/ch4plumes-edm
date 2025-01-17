from pathlib import Path

from functools import partial
from itertools import islice
from data_loading import get_loader
import yaml

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm

from lib import logging, config, Trainer, EDMTrainer
from lib.models.snake_utils import random_bezier

jax.config.update("jax_numpy_rank_promotion", "raise")


def main() -> None:
  run_name = config.name or None

  trainer = EDMTrainer(jax.random.PRNGKey(config.seed))
  train_loader = get_loader(config.batch_size, "train")
  val_loader = get_loader(4, "val")

  wandb.init(project="Diffusion-Cobra", config=config, name=run_name)

  assert wandb.run is not None
  config.wandb_id = wandb.run.id
  run_dir = Path("runs") / wandb.run.id
  run_dir.mkdir(exist_ok=True, parents=True)
  with open(run_dir / "config.yml", "w") as f:
    f.write(yaml.dump(config, default_flow_style=False))

  for epoch in range(1, 501):
    wandb.log({"epoch": epoch}, step=epoch)
    trn_metrics = defaultdict(list)
    for batch in tqdm(train_loader, desc=f"Trn {epoch:3d}", ncols=80):
      metrics = trainer.train_step(batch)
      for k, v in metrics.items():
        trn_metrics[k].append(v)

    logging.log_metrics(trn_metrics, "trn", epoch)

    if epoch % 1 != 0:
      continue

    trainer.save_state((run_dir / f"{epoch}.ckpt").absolute())
    trainer.val_key = jax.random.PRNGKey(0)  # Re-seed val key
    B, H, _, _ = batch["image"].shape
    predictions = []
    for _ in range(5):
      out, metrics = trainer.test_step(batch)
      out = jax.tree.map(lambda x: (x + 1) * (H / 2), out)
      out.update(batch)
      predictions.append(out)

    for i in range(B):
      # jax.tree.map(lambda x: x[i], predictions)
      out = {
        k: np.stack(jax.tree.map(lambda x: x[i], [p[k] for p in predictions]))
        for k in predictions[0]
      }
      logging.log_anim_multi(out, f"TrnAnim/{i}", epoch)

    val_metrics = defaultdict(list)
    for i, batch in enumerate(
      tqdm(islice(val_loader, 8), desc=f"Val {epoch:3d}", ncols=80, total=8)
    ):
      B, H, _, _ = batch["image"].shape
      predictions = []
      for _ in range(5):
        out, metrics = trainer.test_step(batch)
        out = jax.tree.map(lambda x: (x + 1) * (H / 2), out)
        out.update(batch)
        predictions.append(jax.tree.map(lambda x: x[0], out))
        for m in metrics:
          val_metrics[m].append(metrics[m])
      out = {k: np.stack([p[k] for p in predictions]) for k in predictions[0]}
      try:
        filename = batch["filename"][0].decode("utf8").removesuffix(".tif")
        name = f"{batch['year'][0]}_{filename}"
      except:
        name = batch["filename"][0].decode("utf8").removesuffix(".tif")
      logging.log_anim_multi(out, f"Animated/{name}", epoch)
    logging.log_metrics(val_metrics, "val", epoch)


if __name__ == "__main__":
  main()
