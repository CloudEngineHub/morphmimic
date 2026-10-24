"""Clips common to every robot of a multi-robot run, and the (clip, frame) draws of the
cross-robot latent alignment.

Every rank reads all the robots' subset yamls, so every rank knows the same list of
common clips (`common_clip_names`: the SEED clip names, i.e. the npz stems, present in
every subset, sorted). `CommonClipSampler` maps them to this rank's library and draws,
for a given iteration, the same (clip, frame) pairs on every rank from a generator
seeded by the iteration, so no indices are exchanged. Frames are drawn within the
shortest of the robots' versions of a clip (one all-reduce MIN at construction), so
every robot has each drawn frame.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from morphmimic.data.manifest import load_subset
from morphmimic.train.distributed import rank_and_world_size

# Generator seed of the draws of iteration `it`: seed * SEED_STRIDE + it.
SEED_STRIDE = 1_000_003


class _Library(Protocol):
  """What the sampler reads of a `MotionLibrary`."""

  clip_names: list[str]
  clip_start: Tensor
  clip_len: Tensor


def common_clip_names(subset_yamls: Sequence[str]) -> list[str]:
  """Clip names (npz stems) present in every subset yaml, sorted. Raises ValueError
  when a yaml lists a name more than once (the name would not identify one clip)."""
  common: set[str] | None = None
  for path in subset_yamls:
    stems = [Path(p).stem for p in load_subset(path)]
    dups = sorted(name for name, n in Counter(stems).items() if n > 1)
    if dups:
      raise ValueError(f"{path} lists clip names more than once: {dups[:5]}")
    common = set(stems) if common is None else common & set(stems)
  return sorted(common or ())


class CommonClipSampler:
  """Draws of reference frames from the clips `names` (the same list, in the same
  order, on every rank), in this rank's `library`.

  `clips` [C] are the clips' indices in `library` (on its device), `min_len` [C]
  (CPU, long) the shortest length of each clip over the ranks (an all-reduce MIN under
  a process group, so every rank must construct it at the same point). Raises
  ValueError when `names` is empty or a name is not in the library.
  """

  def __init__(self, names: Sequence[str], library: _Library, seed: int) -> None:
    if not names:
      raise ValueError(
        "no clip common to all robots' subsets: the latent alignment needs at least"
        " one (set latent_align_weight 0 to train without it)"
      )
    index = {name: i for i, name in enumerate(library.clip_names)}
    missing = [n for n in names if n not in index]
    if missing:
      raise ValueError(f"clips {missing[:5]} are not in this rank's library")
    self.names = list(names)
    self.seed = seed
    device = library.clip_len.device
    self.clips = torch.tensor([index[n] for n in names], device=device)
    lengths = library.clip_len[self.clips].clone()
    if rank_and_world_size()[1] > 1:
      dist.all_reduce(lengths, op=dist.ReduceOp.MIN)
    self.min_len = lengths.cpu()
    self._start = library.clip_start[self.clips]

  def draw(self, it: int, n: int) -> tuple[Tensor, Tensor]:
    """(common-clip index [n], frame offset in the clip [n]) of iteration `it`, CPU
    long tensors, identical on every rank: a CPU generator seeded
    `seed * 1_000_003 + it` draws the clips uniformly, then each offset uniformly in
    [0, min_len)."""
    gen = torch.Generator().manual_seed(self.seed * SEED_STRIDE + it)
    clip = torch.randint(len(self.names), (n,), generator=gen)
    length = self.min_len[clip]
    offset = (torch.rand(n, generator=gen) * length).long().minimum(length - 1)
    return clip, offset

  def frames(self, it: int, n: int) -> Tensor:
    """Global frame indices [n] in this rank's library (on its device) of the draws
    of iteration `it` (`draw`)."""
    clip, offset = self.draw(it, n)
    clip = clip.to(self._start.device)
    return self._start[clip] + offset.to(self._start.device)
