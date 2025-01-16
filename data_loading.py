import tensorflow_datasets as tfds


def pseudo_dem(batch):
  batch = dict(**batch)
  batch["dem"] = batch["image"][..., :1]
  return batch


def get_loader(batch_size, mode):
  name = "zakynthos"
  ds = tfds.load(name, split=mode)
  if mode == "train":
    ds = ds.repeat(50)
    ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)
  if name == "synthetic_contours":
    ds = ds.map(pseudo_dem)

  return tfds.as_numpy(ds)
