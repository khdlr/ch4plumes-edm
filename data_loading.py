import tensorflow_datasets as tfds


def get_loader(batch_size, mode):
  ds = tfds.load("zakynthos/all", split="train")
  if mode == "train":
    ds = ds.repeat(30)
    ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)

  return tfds.as_numpy(ds)
