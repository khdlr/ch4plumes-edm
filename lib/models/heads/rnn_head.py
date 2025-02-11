from flax import nnx

from functools import partial


class RNNHead(nnx.Module):
  def __init__(self, dim, num_layers, *, rngs: nnx.Rngs):
    self.layers = []
    self.dim = dim
    for _ in range(3):
      forward_cell = nnx.GRUCell(in_features=dim, hidden_features=dim // 2, rngs=rngs)
      backward_cell = nnx.GRUCell(in_features=dim, hidden_features=dim // 2, rngs=rngs)
      self.layers.append(
        nnx.Bidirectional(nnx.RNN(forward_cell), nnx.RNN(backward_cell))
      )

  def __call__(self, x):
    for ly in self.layers:
      x = ly(x)
    return x

  def get_dim(self):
    return self.dim


RNNHead_T = partial(RNNHead, dim=512, num_layers=2)
