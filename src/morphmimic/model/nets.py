"""MLP builder shared by the world model, policy and critic."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
  "silu": nn.SiLU,
  "elu": nn.ELU,
  "relu": nn.ReLU,
  "gelu": nn.GELU,
  "tanh": nn.Tanh,
}


def mlp(
  in_dim: int,
  hidden: tuple[int, ...],
  out_dim: int,
  act: str = "elu",
  layer_norm: bool = False,
) -> nn.Sequential:
  """Linear/activation pairs for each width in `hidden`, then a linear output layer
  (no output activation). `act` is one of silu, elu, relu, gelu, tanh; ELU by default
  (mjlab/BeyondMimic's default). `layer_norm` adds a LayerNorm after each
  hidden activation, as in SHAC's reference networks."""
  if act not in _ACTIVATIONS:
    raise ValueError(
      f"Unknown activation {act!r}; expected one of {list(_ACTIVATIONS)}."
    )
  layers: list[nn.Module] = []
  dim = in_dim
  for width in hidden:
    layers += [nn.Linear(dim, width), _ACTIVATIONS[act]()]
    if layer_norm:
      layers.append(nn.LayerNorm(width))
    dim = width
  layers.append(nn.Linear(dim, out_dim))
  return nn.Sequential(*layers)


def run_mlp(
  net: nn.Module | Callable[[Tensor], Tensor], x: Tensor, bf16: bool = False
) -> Tensor:
  """`net(x)` (a module or any function of x), with its matmuls in bfloat16 autocast when `bf16` and `x` is on CUDA.
  The result is float32 (parameters stay float32; LayerNorm runs in float32 under
  autocast), so everything downstream of the MLP keeps full precision."""
  if bf16 and x.is_cuda:
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return net(x).float()
  return net(x)


def fill_missing_buffers(
  state_dict: dict[str, Tensor], prefix: str, defaults: Mapping[str, Tensor]
) -> None:
  """For `_load_from_state_dict` overrides: inserts `defaults` (buffer name -> value)
  for the buffers a state dict saved before they existed lacks, so older checkpoints
  still load strictly. Only buffers whose default is the no-op value belong here
  (identity statistics, size-0 vectors)."""
  for name, value in defaults.items():
    state_dict.setdefault(prefix + name, value)
