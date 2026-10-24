"""Check retargeted CSVs against their BVH sources: a CSV is complete when it ends in a
newline, has one row per BVH frame (the converter keeps the 120 fps frame count) and
the robot's 38 columns (`Frame`, 6 root, 31 joints) in its first, middle and last
rows. A converter killed or paused mid-export can leave a truncated file.

  uv run python tools/soma_retargeter/validate_csv.py --staged $WORK/seed_stage/all \
    --csv-dir $DATA/seed_x2_csv_full_v3
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import tyro

NUM_COLUMNS = 38


def bvh_frames(bvh: Path) -> int:
  """The `Frames:` count of a BVH file's MOTION header (-1 if absent)."""
  with open(bvh, "rb") as f:
    for line in f:
      if line.startswith(b"Frames:"):
        return int(line.split()[1])
  return -1


def problem(csv: Path, bvh: Path) -> str | None:
  """Why `csv` is not a complete export of `bvh`, or None if it is."""
  if not csv.is_file():
    return "missing"
  data = csv.read_bytes()
  if not data.endswith(b"\n"):
    return "no trailing newline"
  lines = data[:-1].split(b"\n")
  frames = bvh_frames(bvh)
  if len(lines) - 1 != frames:
    return f"rows {len(lines) - 1} != frames {frames}"
  for line in (lines[0], lines[len(lines) // 2], lines[-1]):
    if line.count(b",") != NUM_COLUMNS - 1:
      return "column count"
  return None


def complete(csv: Path, bvh: Path) -> bool:
  return problem(csv, bvh) is None


def _check(args: tuple[Path, Path]) -> tuple[Path, str | None]:
  return args[0], problem(*args)


def main(staged: Path, csv_dir: Path, workers: int = 32) -> None:
  """Validate every staged clip's CSV; exits 1 if any is missing or incomplete.

  Args:
    staged: folder written by stage_subset.py (`<date>/<stem>.bvh` links).
    csv_dir: the converter's export folder (`<date>/<stem>.csv`).
    workers: processes.
  """
  staged, csv_dir = Path(staged), Path(csv_dir)
  jobs = [
    (csv_dir / p.relative_to(staged).with_suffix(".csv"), Path(os.path.realpath(p)))
    for p in sorted(staged.rglob("*.bvh"))
  ]
  with Pool(workers) as pool:
    bad = [(c, why) for c, why in pool.map(_check, jobs, chunksize=64) if why]
  print(f"checked {len(jobs)} clips, {len(bad)} missing or incomplete")
  for c, why in bad[:20]:
    print(f"  {c}: {why}")
  if bad:
    sys.exit(1)


if __name__ == "__main__":
  tyro.cli(main)
