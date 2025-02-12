import jax
import jax.numpy as jnp
import numpy as np
import wandb
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CM = mpl.colormaps["inferno"]


def log_metrics(metrics, prefix, epoch, do_print=True, do_wandb=True):
  metrics = {m: np.mean(metrics[m]) for m in metrics}

  if do_wandb:
    wandb.log({f"{prefix}/{m}": metrics[m] for m in metrics}, step=epoch)
  if do_print:
    print(f"{prefix}/metrics")
    print(", ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))


def to_rgb(values):
  values = values[..., 0]
  mm = values / (values.max() / 3)
  rgb = np.clip(CM(mm)[..., :3] * 255, 0, 255).astype(np.uint8)
  return Image.fromarray(rgb)


def log_steps(steps, tag, step, base_path):
  out_dir = base_path / "gifs" / str(step)
  out_dir.mkdir(exist_ok=True, parents=True)
  imgs = [to_rgb(step) for step in steps]
  imgs[0].save(
    out_dir / f"{tag}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=1
  )
  wandb.log(
    {
      f"Animated/{tag}": wandb.Image(str(out_dir / f"{tag}.gif")),
      f"Samples/{tag}": wandb.Image(imgs[-1]),
    },
    step=step,
  )
