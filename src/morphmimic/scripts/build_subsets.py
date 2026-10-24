"""Build the seeded tiny / 1h / 7h / 40h / all subset yamls.

Example:
  uv run python -m morphmimic.scripts.build_subsets \
    --manifest $DATA/seed_g1_npz50/manifest.csv \
    --out-dir $DATA/seed_g1_npz50/subsets

With `--clips-from`, rebuilds one given subset (e.g. the G1's 1h.yaml) on another
robot's manifest, keeping the clips that survive that robot's filters:
  uv run python -m morphmimic.scripts.build_subsets \
    --manifest $DATA/seed_x2_npz50/manifest.csv \
    --out-dir $DATA/seed_x2_npz50/subsets \
    --clips-from $DATA/seed_g1_npz50/subsets/1h.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import tyro
import yaml

from morphmimic.data.manifest import build_subset, filtered, load_manifest

# Subset name -> target hours (None = every clip that passes the filters).
SUBSET_HOURS: dict[str, float | None] = {
  "tiny": 0.1,
  "1h": 1.0,
  "7h": 7.0,
  "40h": 40.0,
  "all": None,
}


def main(
  manifest: str, out_dir: str, seed: int = 0, clips_from: str | None = None
) -> None:
  """Writes `<out_dir>/<subset>.yaml`; subsets share `seed` and are therefore nested.

  Args:
    manifest: manifest.csv written by convert_seed.
    out_dir: directory for the subset yamls.
    seed: shuffle seed.
    clips_from: a subset yaml; if given, writes only `<out_dir>/<its stem>.yaml` with
      its clips that pass this manifest's filters, in its order, and prints the
      dropped clips.
  """
  df = filtered(load_manifest(manifest))
  print(f"filtered manifest: {len(df)} clips ({df['seconds'].sum() / 3600:.2f} h)")
  if clips_from is not None:
    out_yaml = os.path.join(out_dir, f"{Path(clips_from).stem}.yaml")
    clips = build_subset(df, None, seed, out_yaml, clips_from=clips_from)
    with open(out_yaml) as f:
      hours = yaml.safe_load(f)["hours"]
    with open(clips_from) as f:
      source = yaml.safe_load(f)["clips"]
    kept = set(clips)
    dropped = [c for c in source if c not in kept]
    print(f"kept {len(clips)} of {len(source)} clips, {hours:.3f} h -> {out_yaml}")
    print(f"dropped {len(dropped)} clips:")
    for clip in dropped:
      print(f"  {clip}")
    return
  for name, hours in SUBSET_HOURS.items():
    out_yaml = os.path.join(out_dir, f"{name}.yaml")
    clips = build_subset(df, hours, seed, out_yaml)
    total = df.set_index("path").loc[clips, "seconds"].sum() / 3600
    print(f"{name:>4}: {len(clips)} clips, {total:.3f} h -> {out_yaml}")


if __name__ == "__main__":
  tyro.cli(main)
