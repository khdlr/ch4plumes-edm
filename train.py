from pathlib import Path

from data_loading import get_loader
import yaml

import jax
import orbax.checkpoint as ocp
from collections import defaultdict
import wandb
from tqdm import tqdm

from lib import logging, config, load_config, Trainer

jax.config.update("jax_numpy_rank_promotion", "raise")


def main() -> None:
  load_config()
  run_name = config.name or None

  train_loader = get_loader(config.batch_size, "train")
  val_loader = get_loader(4, "val")
  trainer = Trainer(jax.random.PRNGKey(config.seed))

  wandb.init(
    project="COBRA Zakynthos", config=config, name=run_name, group=config.group
  )

  assert wandb.run is not None
  config.wandb_id = wandb.run.id
  run_dir = Path("runs") / wandb.run.id
  with open(run_dir / "config.yml", "w") as f:
    f.write(yaml.dump(config, default_flow_style=False))
  checkpointer = ocp.PyTreeCheckpointer()
  trn_metrics = defaultdict(list)

  for epoch in range(1, 501):
    wandb.log({"epoch": epoch}, step=epoch)
    trn_metrics = {}
    for batch in tqdm(train_loader, desc=f"Trn {epoch:3d}"):
      metrics = trainer.train_step(batch)
      for k, v in metrics.items():
        trn_metrics[k].append(v)

    logging.log_metrics(trn_metrics, "trn", epoch)

    if epoch % 10 != 0:
      continue

    checkpointer.save((run_dir / f"{epoch}.ckpt").absolute(), trainer.state)

    trainer.val_key = jax.random.PRNGKey(0)  # Re-seed val key
    val_metrics = defaultdict(list)
    for step, batch in tqdm(enumerate(val_loader), desc=f"Val {epoch:3d}"):
      pass
