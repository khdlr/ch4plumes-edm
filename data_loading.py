import tensorflow as tf
import tensorflow_datasets as tfds
from lib.config_mod import config


def purge_dem(batch):
  batch = dict(**batch)
  batch["dem"] = batch["dem"] * 0
  return batch


def get_loader(batch_size, mode):
  name = config.dataset
  print(f"loading {name}")
  ds = tfds.load(name, split=mode, shuffle_files=(mode == "train"))
  if mode == "train":
    if name == "zakynthos":
      ds = ds.repeat(50)
    ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)
  ds = ds.map(purge_dem)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  return tfds.as_numpy(ds)
