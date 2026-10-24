"""Data-parallel training helpers (paper Table A.5: "Multi-GPU: data parallel; gradients
averaged across GPUs").

One process per GPU, each with its own env, replay, collector and normaliser copies.
All ranks start from rank 0's networks and normalisers (`broadcast_`), every gradient
is averaged over the ranks before its optimiser step (`all_reduce_grads_`: one
all-reduce of the flattened gradients, before the paper's clipping), and the running
input normalisers pool every rank's batch (`RunningNormalizer(sync=True)`), so the
parameters and normalisers stay identical on every rank. These are plain
`torch.distributed` collectives outside the compiled regions, not DDP wrappers, so the
compiled (CUDA-graph) steps see ordinary modules. `param_checksums` lets the trainer
verify at each checkpoint that the copies (parameters and buffers) are still
identical.

Multi-robot mode (`TrainCfg.robots`: rank r simulates robot r) shares only some
modules and only their parameters: `broadcast_params_` starts them from rank 0's
parameters without touching the buffers (each robot's normaliser, joint limits,
default pose, residual scale, height statistics and morphology vector stay its own),
`param_checksums(..., buffers=False)` compares the parameters alone, and
`latent_alignment_loss` pulls each rank's motion latents towards the mean over the
ranks.

Without a process group (one GPU) none of this runs: `rank_and_world_size()` is (0, 1).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def rank_and_world_size() -> tuple[int, int]:
  """(rank, world size) of the default process group; (0, 1) without one."""
  if dist.is_available() and dist.is_initialized():
    return dist.get_rank(), dist.get_world_size()
  return 0, 1


@torch.no_grad()
def broadcast_(*modules: nn.Module, src: int = 0) -> None:
  """Overwrites every rank's parameters and buffers (`state_dict` tensors) with rank
  `src`'s, in place (the tensors keep their storage)."""
  for module in modules:
    for tensor in module.state_dict().values():
      if tensor.is_contiguous():
        dist.broadcast(tensor, src)
      else:
        buf = tensor.contiguous()
        dist.broadcast(buf, src)
        tensor.copy_(buf)


@torch.no_grad()
def broadcast_params_(*modules: nn.Module, src: int = 0) -> None:
  """Overwrites every rank's parameters (not the buffers) with rank `src`'s, in place
  (the tensors keep their storage). Multi-robot mode: the shared modules start
  identical while each rank keeps its own robot's buffers."""
  for module in modules:
    for param in module.parameters():
      tensor = param.data
      if tensor.is_contiguous():
        dist.broadcast(tensor, src)
      else:
        buf = tensor.contiguous()
        dist.broadcast(buf, src)
        tensor.copy_(buf)


@torch.no_grad()
def all_reduce_grads_(params: Iterable[Tensor]) -> None:
  """Replaces each gradient by its mean over the ranks: one all-reduce (sum, then
  / world size) of all gradients flattened into one buffer. Parameters without a
  gradient are skipped; the ranks run the same graphs, so the same ones on every
  rank."""
  grads = [p.grad for p in params if p.grad is not None]
  if not grads:
    return
  flat = _flatten_dense_tensors(grads)
  dist.all_reduce(flat)
  flat.div_(dist.get_world_size())
  torch._foreach_copy_(grads, _unflatten_dense_tensors(flat, grads))


@torch.no_grad()
def param_checksums(
  *modules: nn.Module, buffers: bool = True
) -> tuple[Tensor, Tensor]:
  """Per-module float64 sum of all parameters and buffers (the `state_dict` tensors,
  e.g. the world model's normaliser and residual scale; with `buffers=False` the
  parameters only), as (max, min) over the ranks, each [len(modules)] on the
  parameters' device: one all-reduce (MAX of the sums and of their negatives).
  Identical copies give max == min."""

  def tensors(module: nn.Module) -> list[Tensor]:
    if buffers:
      return list(module.state_dict().values())
    return [p.detach() for p in module.parameters()]

  sums = torch.stack(
    [
      torch.stack([t.sum(dtype=torch.float64) for t in tensors(m)]).sum()
      for m in modules
    ]
  )
  both = torch.cat([sums, -sums])
  dist.all_reduce(both, op=dist.ReduceOp.MAX)
  n = len(modules)
  return both[:n], -both[n:]


def latent_alignment_loss(z: Tensor) -> tuple[Tensor, Tensor]:
  """Cross-robot latent alignment of this rank's motion latents z [B, L],
  every rank's row b encoding the same reference (clip, frame) for its own robot:

    loss = mean_b ||z_b - stopgrad(mean_r z_{r,b})||^2,

  with the mean over the ranks of the all-gathered latents (gathered without
  gradient, so the gradient flows only through this rank's own z). Also returns the
  mean cross-robot latent distance mean_{r<s} mean_b ||z_{r,b} - z_{s,b}|| (no
  gradient). Collective when a process group of more than one rank exists (every rank
  calls it with the same B); without one, or on one rank, both are 0."""
  _, world = rank_and_world_size()
  if world == 1:
    zero = (z - z.detach()).square().sum(-1).mean()  # 0, in z's graph
    return zero, zero.detach()
  with torch.no_grad():
    gathered = [torch.empty_like(z) for _ in range(world)]
    dist.all_gather(gathered, z.detach().contiguous())
    stacked = torch.stack(gathered)  # [W, B, L]
    center = stacked.mean(0)
    pairs = [
      (stacked[r] - stacked[s]).norm(dim=-1).mean()
      for r in range(world)
      for s in range(r + 1, world)
    ]
    distance = torch.stack(pairs).mean()
  loss = (z - center).square().sum(-1).mean()
  return loss, distance
