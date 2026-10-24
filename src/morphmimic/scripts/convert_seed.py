"""Convert SEED-format robot CSVs to 50 Hz npz motions and write the manifest.

Example:
  uv run python -m morphmimic.scripts.convert_seed --csv-dir $DATA/seed/g1/csv \
    --out-dir $DATA/seed_g1_npz50 --metadata $DATA/seed/seed_metadata_v003.csv \
    --workers 120

For the X2 / T2, pass `--robot x2` / `--robot t2` with their retargeted CSVs.
"""

from __future__ import annotations

import os

import tyro

from morphmimic.data.manifest import filtered
from morphmimic.data.seed_convert import convert_all


def main(
  csv_dir: str,
  out_dir: str,
  metadata: str,
  workers: int = os.cpu_count() or 1,
  limit: int | None = None,
  overwrite: bool = False,
  robot: str = "g1",
) -> None:
  """Converts `csv_dir/<session>/<clip>.csv` into `out_dir` and writes manifest.csv.

  Args:
    csv_dir: SEED-format CSV root of `robot` (one sub-directory per session).
    out_dir: npz output root.
    metadata: SEED metadata CSV (seed_metadata_v003.csv).
    workers: number of conversion processes.
    limit: convert only the first `limit` clips (sorted by path).
    overwrite: reconvert clips whose npz already exists.
    robot: robot registry name (g1, x2, t2).
  """
  df = convert_all(
    csv_dir,
    out_dir,
    metadata,
    workers=workers,
    limit=limit,
    overwrite=overwrite,
    robot=robot,
  )
  named = df[df["name_reject"] == ""]
  kept = filtered(df)
  feas = named.loc[named["feas_reject"] != "", "feas_reject"]
  print(f"clips: {len(df)} ({df['seconds'].sum() / 3600:.2f} h)")
  print(f"after name filter: {len(named)}")
  print(f"after feasibility filter: {len(kept)} ({kept['seconds'].sum() / 3600:.2f} h)")
  print(f"feasibility rejects after name filter: {feas.value_counts().to_dict()}")
  print(f"manifest: {os.path.join(out_dir, 'manifest.csv')}")


if __name__ == "__main__":
  tyro.cli(main)
