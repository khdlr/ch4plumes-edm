from pathlib import Path

from data_loading import get_loader
import jax
import geopandas as gpd
import shapely as shp
from tqdm import tqdm

from lib import config, EDMTrainer
from lib.models.snake_utils import random_bezier

jax.config.update("jax_numpy_rank_promotion", "raise")


def main() -> None:
  run_name = config.name or None
  trainer = EDMTrainer(jax.random.PRNGKey(config.seed))
  test_loader = get_loader(16, "test")
  out_dir = Path(f"inference/{config.resume_from}")
  out_dir.mkdir(exist_ok=True, parents=True)

  # val_metrics = defaultdict(list)
  results = []
  for batch in tqdm(test_loader, desc=f"Inference", ncols=80):
    B, H, _, _ = batch["image"].shape
    predictions = []
    for _ in range(5):
      out, metrics = trainer.test_step(batch)
      out = jax.tree.map(lambda x: (x + 1) * (H / 2), out)
      out.update(batch)
      predictions.append(out["snake"])

    for i in range(B):
      tx = batch["transform"][i]
      a, b, c, d, e, f = tx
      for s in range(5):
        snake = predictions[s][i]
        snake = snake[..., ::-1]
        snake = shp.LineString(snake)
        snake = shp.affinity.affine_transform(snake, [a, b, d, e, c, f])

        results.append(
          {
            "filename": batch["filename"][i].decode("utf8"),
            "year": batch["year"][i],
            "geom": snake,
          }
        )

  results = gpd.GeoDataFrame(results, crs="EPSG:2100", geometry="geom")
  results.to_parquet(out_dir / "predictions.parquet")


if __name__ == "__main__":
  main()
