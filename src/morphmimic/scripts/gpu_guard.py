"""GPU selection shared by the scripts: the physical GPUs of a run via
CUDA_VISIBLE_DEVICES (one GPU becomes cuda:0; a data-parallel run's i-th GPU is
cuda:i of its rank-i process). Every GPU id is allowed; `--allow-any-gpu` is still
accepted and does nothing.

CUDA_VISIBLE_DEVICES is read when CUDA initialises, not when torch is imported, so
it can be set after `import morphmimic` (whose imports do not touch the GPU) as long
as nothing has initialised CUDA yet; `select_gpu` checks that.
"""

from __future__ import annotations

import os
import sys


def select_gpu(gpu: int, allow_any_gpu: bool = False) -> None:
  """Exposes only physical GPU `gpu` to this process (`allow_any_gpu` is ignored)."""
  select_gpus([gpu], allow_any_gpu)


def select_gpus(gpus: list[int], allow_any_gpu: bool = False) -> None:
  """Exposes only the physical GPUs `gpus` (distinct, in this order) to this process
  and the processes it starts (`allow_any_gpu` is ignored, kept for old command
  lines)."""
  del allow_any_gpu
  if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
    raise ValueError(f"need distinct non-negative GPU ids, got {gpus}")
  torch = sys.modules.get("torch")
  if torch is not None and torch.cuda.is_initialized():
    raise RuntimeError("CUDA is already initialised; select the GPU before using it.")
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
