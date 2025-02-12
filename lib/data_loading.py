import tensorflow as tf
import tensorflow_datasets as tfds
from lib.config_mod import config


def get_loader(batch_size, mode):
  name = config.dataset
  ds = tfds.load(name, split=mode, shuffle_files=(mode == "train"))
  if mode == "train":
    ds = ds.repeat(10)
    ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)
