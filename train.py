from pathlib import Path

from itertools import islice
import yaml

import jax
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm

from lib import logging, config, EDMTrainer, get_loader

jax.config.update("jax_numpy_rank_promotion", "raise")


def main() -> None:
  run_name = config.name or None

  trainer = EDMTrainer(jax.random.PRNGKey(config.seed))
  data_loader = get_loader(config.batch_size, "train")
  wandb.init(project="Plumes EDM", config=config, name=run_name)

  assert wandb.run is not None
  config.wandb_id = wandb.run.id
  run_dir = Path("runs") / wandb.run.id
  run_dir.mkdir(exist_ok=True, parents=True)
  with open(run_dir / "config.yml", "w") as f:
    f.write(yaml.dump(config, default_flow_style=False))

  val_frequency = 1

  for epoch in range(1, 501):
    wandb.log({"epoch": epoch}, step=epoch)
    trn_metrics = defaultdict(list)
    for batch in tqdm(data_loader, desc=f"Trn {epoch:3d}", ncols=80):
      metrics = trainer.train_step(batch)
      for k, v in metrics.items():
        trn_metrics[k].append(v)

    logging.log_metrics(trn_metrics, "trn", epoch)

    if epoch % val_frequency != 0:
      continue

    trainer.save_state((run_dir / f"{epoch}.ckpt").absolute())

    out = trainer.sample(key=jax.random.key(0))
    steps = out["steps"]
    B = steps[0].shape[0]
    for i in range(B):
      sample_steps = [steps[t][i] for t in range(18)]
      logging.log_steps(sample_steps, f"{i:02d}", epoch, run_dir)


if __name__ == "__main__":
  main()
