import tensorflow as tf
import tensorflow_datasets as tfds


def pseudo_dem(batch):
  batch = dict(**batch)
  batch["dem"] = batch["image"][..., :1]
  return batch


def prepare_coastlines(batch):
  batch = dict(**batch)
  m = batch["mask"]
  batch["image"] = 255 * tf.cast(tf.concat([m, m, m], axis=-1), tf.uint8)
  batch["dem"] = 300 * tf.cast(m, tf.float32)
  batch["contour"] = tf.reverse(batch["contour"], axis=[-1])
  batch["filename"] = tf.strings.reduce_join(
    tf.strings.as_string(batch["xyz"]), separator="/", axis=-1
  )
  return batch


def get_loader(batch_size, mode):
  name = "coastlines"
  ds = tfds.load(name, split=mode)
  if name == "coastlines":
    ds = ds.filter(lambda x: x["xyz"][2] < 10)

  if mode == "train":
    if name == "zakynthos":
      ds = ds.repeat(50)
    ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)
  if name == "synthetic_contours":
    ds = ds.map(pseudo_dem)
  elif name == "coastlines":
    ds = ds.map(prepare_coastlines)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  return tfds.as_numpy(ds)
