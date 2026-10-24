"""Stage a subset's SOMA BVH files as a folder tree of symlinks for the retargeter.

For every clip `<date>/<stem>.npz` of the subset yaml, looks the stem up in the
BONES-SEED metadata (`filename` column) and links `<out>/<date>/<stem>.bvh` to
`<bvh_root>/<date>/<stem>.bvh`, `<date>` taken from `move_soma_uniform_path`
(`soma_uniform/bvh/<date>/<stem>.bvh`). The retargeter's batch converter then reads
`<out>` as its `import_folder`.

  uv run python tools/soma_retargeter/stage_subset.py \
    --subset-yaml $DATA/seed_g1_npz50/subsets/1h.yaml \
    --metadata $DATA/seed/metadata/seed_metadata_v004.csv \
    --bvh-root $DATA/seed/soma_uniform/bvh --out $WORK/seed_stage/1h
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import tyro
import yaml


def stage(subset_yaml: Path, metadata_csv: Path, bvh_root: Path, out: Path) -> list[Path]:
  """Creates the links and returns them; exits 1 listing the missing stems, if any.

  A stem is missing when it has no metadata row or its BVH file does not exist; the
  links of the stems found are created first. Existing links are replaced, and every
  other symlink under `out` (e.g. from an earlier, different subset, or a stem now
  missing) is removed, so `out` holds exactly this subset. Regular files are left alone.
  """
  with open(subset_yaml) as f:
    stems = [Path(clip).stem for clip in yaml.safe_load(f)["clips"]]
  with open(metadata_csv, newline="") as f:
    paths = {row["filename"]: row["move_soma_uniform_path"] for row in csv.DictReader(f)}

  links: list[Path] = []
  missing: list[str] = []
  for stem in stems:
    if stem not in paths:
      missing.append(stem)
      continue
    rel = Path(*Path(paths[stem]).parts[-2:])  # <date>/<stem>.bvh
    target = Path(bvh_root).resolve() / rel
    if not target.is_file():
      missing.append(stem)
      continue
    link = Path(out) / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
      link.unlink()
    link.symlink_to(target)
    links.append(link)

  keep = set(links)
  stale = [p for p in Path(out).rglob("*") if p.is_symlink() and p not in keep]
  for p in stale:
    p.unlink()
  for d in sorted((p for p in Path(out).rglob("*") if p.is_dir()), reverse=True):
    if not any(d.iterdir()):
      d.rmdir()

  print(
    f"staged {len(links)} clips, {len(missing)} missing, removed {len(stale)} stale links"
    f" -> {out}",
    flush=True,
  )
  if missing:
    print(f"missing {len(missing)} clips:", file=sys.stderr)
    for stem in missing:
      print(f"  {stem}", file=sys.stderr)
    raise SystemExit(1)
  return links


def main(subset_yaml: Path, metadata: Path, bvh_root: Path, out: Path) -> None:
  """Stage a subset's BVH files.

  Args:
    subset_yaml: subset yaml (e.g. the G1's 1h.yaml); its `clips` are `<date>/<stem>.npz`.
    metadata: BONES-SEED seed_metadata_v004.csv.
    bvh_root: `<seed>/soma_uniform/bvh`.
    out: stage folder; becomes `<out>/<date>/<stem>.bvh` symlinks.
  """
  stage(subset_yaml, metadata, bvh_root, out)


if __name__ == "__main__":
  tyro.cli(main)
