import tensorflow as tf
import tensorflow_datasets as tfds
from lib.config_mod import config


def normalize(batch):
  batch = dict(**batch)
  # Drop elevation data (if any)
  batch["dem"] = tf.zeros_like(batch["image"][..., :1])
  if "trace_class" not in batch:
    batch["trace_class"] = tf.cast(batch["dem"][..., 0, 0, 0], tf.int32)
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
  ds = ds.map(normalize)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  return tfds.as_numpy(ds)
