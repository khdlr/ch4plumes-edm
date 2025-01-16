import tensorflow as tf
import tensorflow_datasets as tfds


def pseudo_dem(batch):
  batch = dict(**batch)
  batch["dem"] = batch["image"][..., :1]
  return batch


def prepare_coastlines(batch):
  batch = dict(**batch)

  # Train on the mask
  # m = batch["mask"]
  # batch["image"] = tf.cast(tf.concat([m, m, m], axis=-1), tf.uint8)
  # batch["dem"] = tf.cast(m, tf.float32)

  # Train on the image
  dem = tf.cast(batch["image"], tf.float32)
  dem = tf.reduce_mean(dem, axis=-1, keepdims=True)
  batch["dem"] = dem
  batch["contour"] = tf.reverse(batch["contour"], axis=[-1])
  batch["filename"] = tf.strings.reduce_join(
    tf.strings.as_string(batch["xyz"]), separator="/", axis=-1
  )
  return batch


def get_loader(batch_size, mode):
  name = "coastlines"
  ds = tfds.load(name, split=mode)
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
