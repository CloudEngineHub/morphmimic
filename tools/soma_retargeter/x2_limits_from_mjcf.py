"""Hinge-joint limits of the X2 Ultra MJCF, for the retargeter's X2 URDF.

The SOMA Retargeter reads joint limits from its URDF's `<limit lower upper>`; ours
come from the MJCF we simulate, which differs on the waist, head and wrists.

  uv run python tools/soma_retargeter/x2_limits_from_mjcf.py   # prints them as JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco


def limits(xml_path: str | Path) -> dict[str, tuple[float, float]]:
  """`{joint name: (lower, upper)}` in radians for every hinge joint, in model order."""
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  out = {}
  for i in range(model.njnt):
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    if not model.jnt_limited[i]:
      raise ValueError(f"{xml_path}: hinge {model.joint(i).name} has no range")
    lo, hi = model.jnt_range[i]
    out[model.joint(i).name] = (float(lo), float(hi))
  return out


if __name__ == "__main__":
  from morphmimic.robots.agibot_x2_ultra.x2_ultra_constants import X2_ULTRA_XML

  print(json.dumps(limits(X2_ULTRA_XML), indent=2))
