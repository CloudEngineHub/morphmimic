"""Conversion manifest and dataset subsets.

The manifest (`<npz root>/manifest.csv`) has one row per clip with MANIFEST_COLUMNS;
`path` is relative to the npz root and the reject columns are "" for kept clips. A
subset is a yaml file `{"root": <abs npz root>, "hours": h, "clips": [rel. paths]}`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MANIFEST_COLUMNS: tuple[str, ...] = (
  "name",
  "path",
  "session",
  "frames",
  "seconds",
  "name_reject",
  "feas_reject",
  "category",
  "package",
)
_STR_COLUMNS = tuple(c for c in MANIFEST_COLUMNS if c not in ("frames", "seconds"))


def load_manifest(path: str) -> pd.DataFrame:
  """Reads a manifest; `df.attrs["root"]` is the absolute npz root (its directory)."""
  df = pd.read_csv(
    path, dtype=dict.fromkeys(_STR_COLUMNS, str), keep_default_na=False
  )
  df.attrs["root"] = str(Path(path).resolve().parent)
  return df


def filtered(df: pd.DataFrame) -> pd.DataFrame:
  """Clips that pass both the name and the feasibility filter."""
  return df[(df["name_reject"] == "") & (df["feas_reject"] == "")]


def build_subset(
  df: pd.DataFrame,
  hours: float | None,
  seed: int,
  out_yaml: str,
  clips_from: str | None = None,
) -> list[str]:
  """Seeded random subset of `df` with at least `hours` of motion (all clips if None).

  Clips are sorted by path, shuffled with `seed` and taken greedily until the total
  duration reaches `hours`, so subsets with the same seed are nested. With
  `clips_from` (a subset yaml, e.g. another robot's), the clips are instead that
  yaml's clips present in `df["path"]`, in its order, and `hours` and `seed` are
  ignored. `df` must come from `load_manifest` (it provides the npz root). Writes
  `out_yaml` (its `hours` is the total of the kept clips) and returns its relative
  clip paths.
  """
  root = df.attrs.get("root")
  if root is None:
    raise ValueError("df has no npz root; load it with load_manifest()")
  if clips_from is not None:
    with open(clips_from) as f:
      wanted = yaml.safe_load(f)["clips"]
    seconds_of = dict(zip(df["path"], df["seconds"].astype(np.float64)))
    clips = [str(p) for p in wanted if p in seconds_of]
    total_hours = float(sum(seconds_of[p] for p in clips) / 3600.0)
    _write_subset(out_yaml, root, total_hours, clips)
    return clips
  df = df.sort_values("path")
  order = np.random.default_rng(seed).permutation(len(df))
  paths = df["path"].to_numpy()[order]
  seconds = df["seconds"].to_numpy(dtype=np.float64)[order]
  if hours is None:
    count = len(df)
  else:
    reached = np.cumsum(seconds) >= hours * 3600.0 - 1e-6
    count = int(np.argmax(reached)) + 1 if reached.any() else len(df)
  clips = [str(p) for p in paths[:count]]
  total_hours = float(seconds[:count].sum() / 3600.0)
  _write_subset(out_yaml, root, total_hours, clips)
  return clips


def _write_subset(out_yaml: str, root: str, hours: float, clips: list[str]) -> None:
  Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
  with open(out_yaml, "w") as f:
    spec = {"root": root, "hours": hours, "clips": clips}
    yaml.safe_dump(spec, f, sort_keys=False)


def load_subset(yaml_path: str) -> list[str]:
  """Absolute npz paths of a subset yaml."""
  with open(yaml_path) as f:
    spec = yaml.safe_load(f)
  root = Path(spec["root"])
  return [str(root / p) for p in spec["clips"]]
