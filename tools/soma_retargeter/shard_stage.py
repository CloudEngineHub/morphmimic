"""Split a staged BVH tree into size-balanced shards for parallel converter runs.

Each shard `<out>/s<NN>/<date>/<stem>.bvh` links to the clip's real BVH file. Clips
whose CSV under `--done-csv-dir` is already complete (`validate_csv.complete`) are
skipped, so a rerun after an interruption only queues the unfinished clips. Clips are
dealt round-robin in decreasing file size, so every shard gets a similar amount of
motion (the converter's time is proportional to frames). `<out>` is replaced.

  uv run python tools/soma_retargeter/shard_stage.py --staged $WORK/seed_stage/all \
    --out $WORK/seed_stage/rem_x2 --shards 18 --done-csv-dir $DATA/seed_x2_csv_full_v3
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import tyro
from validate_csv import complete


def shard(staged: Path, out: Path, shards: int, done_csv_dir: Path | None = None) -> int:
  """Writes the shards and returns the number of clips queued."""
  if shards < 1:
    raise ValueError(f"shards must be >= 1, got {shards}")
  staged, out = Path(staged), Path(out)
  clips = sorted(p.relative_to(staged) for p in staged.rglob("*.bvh"))
  real = {c: Path(os.path.realpath(staged / c)) for c in clips}
  if done_csv_dir is not None:
    done = Path(done_csv_dir)
    clips = [c for c in clips if not complete(done / c.with_suffix(".csv"), real[c])]
  clips.sort(key=lambda c: (-real[c].stat().st_size, str(c)))
  if out.exists():
    shutil.rmtree(out)
  for i, c in enumerate(clips):
    link = out / f"s{i % shards:02d}" / c
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real[c])
  print(f"queued {len(clips)} clips in {shards} shards -> {out}")
  return len(clips)


def main(staged: Path, out: Path, shards: int, done_csv_dir: Path | None = None) -> None:
  """Shard a staged tree.

  Args:
    staged: folder written by stage_subset.py.
    out: shard root (replaced).
    shards: number of shards (one converter process each).
    done_csv_dir: the run's export folder; clips with a complete CSV there are skipped.
  """
  shard(staged, out, shards, done_csv_dir)


if __name__ == "__main__":
  tyro.cli(main)
