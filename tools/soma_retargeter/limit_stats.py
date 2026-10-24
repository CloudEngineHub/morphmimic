"""Per-joint limit statistics of retargeted motions against the simulator's limits.

For every native joint of a robot: the fraction of frames at the hard limit (within
1e-3 rad of the MJCF range, either end), at the soft limit (within 1e-3 rad of either
soft end), below the soft lower and above the soft upper limit by more than 1e-4 rad
(mjlab's soft limits: the range shrunk about its midpoint by the robot's
`soft_joint_pos_limit_factor`), and the 1st/5th/50th percentiles. mjlab clips the
reference to the soft limits at reset and penalises the policy beyond them, so every
frame outside them is a built-in tracking mismatch. The 1e-4 rad slack absorbs the
float32 and CSV rounding of frames the retargeter clamped onto a soft limit (~2e-6).

  # npz motions of a subset (clips relative to the subset's npz root):
  uv run python tools/soma_retargeter/limit_stats.py --robot t2 \
    --subset-yaml $DATA/seed_t2_npz50/subsets/1h.yaml
  # SEED-format CSVs, every *.csv below the folder:
  uv run python tools/soma_retargeter/limit_stats.py --robot t2 --csv-dir $DATA/seed_t2_csv
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import tyro
import yaml
from assemble_robots import limit_factor

from morphmimic.data.seed_convert import make_model, parse_csv
from morphmimic.robots import get_robot

AT_LIMIT_TOL = 1e-3  # "at a limit": within this of it (rad)
SOFT_TOL = 1e-4  # "outside the soft limits": beyond them by more than this (rad)


class JointStats(NamedTuple):
  name: str
  lo: float
  hi: float
  soft_lo: float
  soft_hi: float
  at_hard: float  # fraction of frames within AT_LIMIT_TOL of lo or hi
  at_soft: float  # fraction of frames within AT_LIMIT_TOL of soft_lo or soft_hi
  below_soft: float  # fraction of frames < soft_lo - SOFT_TOL
  above_soft: float  # fraction of frames > soft_hi + SOFT_TOL
  p1: float
  p5: float
  p50: float


def hard_limits(robot: str) -> dict[str, tuple[float, float]]:
  """`{native joint: (lower, upper)}` of the robot's MJCF, in `joint_names` order."""
  model = make_model(robot)
  return {n: tuple(float(v) for v in model.jnt_range[model.joint(n).id]) for n in get_robot(robot).joint_names}


def joint_stats(robot: str, q: np.ndarray, factor: float | None = None) -> list[JointStats]:
  """Statistics of `q` [T, J] (radians, `get_robot(robot).joint_names` order)."""
  factor = limit_factor(robot) if factor is None else factor
  out = []
  for j, (name, (lo, hi)) in enumerate(hard_limits(robot).items()):
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    soft_lo, soft_hi = mid - factor * half, mid + factor * half
    v = q[:, j]
    at = np.mean((v <= lo + AT_LIMIT_TOL) | (v >= hi - AT_LIMIT_TOL))
    at_soft = np.mean((np.abs(v - soft_lo) <= AT_LIMIT_TOL) | (np.abs(v - soft_hi) <= AT_LIMIT_TOL))
    p1, p5, p50 = np.percentile(v, [1, 5, 50])
    out.append(JointStats(
      name, lo, hi, soft_lo, soft_hi, float(at), float(at_soft),
      float(np.mean(v < soft_lo - SOFT_TOL)), float(np.mean(v > soft_hi + SOFT_TOL)),
      float(p1), float(p5), float(p50),
    ))  # fmt: skip
  return out


def load_subset(robot: str, subset_yaml: Path, npz_root: Path | None = None) -> tuple[np.ndarray, int]:
  """All frames of a subset's npz clips as [T, J] in `joint_names` order, and the clip count.

  Clips are relative to `npz_root` (default: the folder above the yaml's `subsets/`).
  """
  subset_yaml = Path(subset_yaml).expanduser()
  root = subset_yaml.parent.parent if npz_root is None else Path(npz_root).expanduser()
  with open(subset_yaml) as f:
    clips = yaml.safe_load(f)["clips"]
  names = list(get_robot(robot).joint_names)
  frames = []
  for clip in clips:
    z = np.load(root / clip)
    have = [str(n) for n in z["joint_names"]]
    frames.append(z["joint_pos"][:, [have.index(n) for n in names]].astype(np.float64))
  return np.concatenate(frames), len(clips)


def load_csvs(robot: str, csv_dir: Path) -> tuple[np.ndarray, int]:
  """All frames of the SEED-format CSVs under `csv_dir`, and the file count."""
  spec = get_robot(robot)
  paths = sorted(Path(csv_dir).expanduser().rglob("*.csv"))
  if not paths:
    raise FileNotFoundError(f"no *.csv under {csv_dir}")
  return np.concatenate([parse_csv(str(p), spec)[2] for p in paths]), len(paths)


def format_stats(stats: list[JointStats]) -> str:
  head = (
    f"{'joint':28s} {'hard lo':>8s} {'hard hi':>8s} {'soft lo':>8s} {'soft hi':>8s} "
    f"{'at hard':>8s} {'at soft':>8s} {'<soft':>8s} {'>soft':>8s} "
    f"{'p1':>8s} {'p5':>8s} {'p50':>8s}"
  )
  rows = [head]
  for s in stats:
    rows.append(
      f"{s.name:28s} {s.lo:8.4f} {s.hi:8.4f} {s.soft_lo:8.4f} {s.soft_hi:8.4f} "
      f"{s.at_hard:8.2%} {s.at_soft:8.2%} {s.below_soft:8.2%} {s.above_soft:8.2%} "
      f"{s.p1:8.4f} {s.p5:8.4f} {s.p50:8.4f}"
    )
  return "\n".join(rows)


def main(
  robot: str,
  subset_yaml: Path | None = None,
  npz_root: Path | None = None,
  csv_dir: Path | None = None,
  factor: float | None = None,
) -> None:
  """Print per-joint limit statistics.

  Args:
    robot: registry name (g1, x2, t2).
    subset_yaml: subset yaml of npz clips (`clips: [<date>/<stem>.npz, ...]`).
    npz_root: folder the clips are relative to (default: the yaml's parent's parent).
    csv_dir: read every SEED-format *.csv below this folder instead of a subset.
    factor: soft-limit factor (default: the robot's soft_joint_pos_limit_factor).
  """
  if (subset_yaml is None) == (csv_dir is None):
    raise SystemExit("give exactly one of --subset-yaml and --csv-dir")
  if subset_yaml is not None:
    q, n = load_subset(robot, subset_yaml, npz_root)
    src = subset_yaml
  else:
    q, n = load_csvs(robot, csv_dir)
    src = csv_dir
  factor = limit_factor(robot) if factor is None else factor
  print(
    f"{robot}: {src}: {n} clips, {len(q)} frames, soft factor {factor}, "
    f"at-limit tol {AT_LIMIT_TOL}, outside-soft tol {SOFT_TOL}"
  )
  print(format_stats(joint_stats(robot, q, factor)))


if __name__ == "__main__":
  tyro.cli(main)
