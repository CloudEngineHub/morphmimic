"""Assemble the SOMA Retargeter `extra_robot_paths` folder for the X2 and the T2.

  <out>/agibot/x2   the retargeter's own X2 Ultra target (manifest, configs, URDF), its
                    URDF joint limits replaced by our MJCF's (x2_limits_from_mjcf) and
                    its feet and left hand re-targeted to the distal links
                    (`fix_x2_effectors`); same registry name, so it overrides the
                    bundled `agibot_x2ultra`.
  <out>/booster/t2  our Booster T2 target (tools/soma_retargeter/robotics/booster/t2).

Both URDFs get the simulator's SOFT joint limits: every revolute range [lo, hi] is
shrunk about its midpoint by the robot's `soft_joint_pos_limit_factor` f (mjlab's
rule, from our registry): mid -+ f * (hi - lo) / 2. mjlab clips the reference to
these at reset and penalises the policy beyond them, so the IK must stay inside.

`desc/meshes` is a symlink in both: to the clone's X2 meshes (real after `git lfs pull`)
and to the T2 STL files vendored in src/morphmimic/robots/booster_t2/xmls/assets.
Rebuilding replaces both robot folders; nothing else under <out> is touched.

  uv run python tools/soma_retargeter/assemble_robots.py \
    --retargeter <soma-retargeter clone> --out $WORK/soma_robots
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tyro
from x2_limits_from_mjcf import limits

from morphmimic.robots import get_robot
from morphmimic.robots.agibot_x2_ultra.x2_ultra_constants import X2_ULTRA_XML
from morphmimic.robots.booster_t2.t2_constants import T2_XML

T2_SRC = Path(__file__).resolve().parent / "robotics" / "booster" / "t2"
T2_MESHES = T2_XML.parent / "assets"
X2_REL = Path("soma_retargeter") / "assets" / "robotics" / "agibot" / "x2"
X2_URDF = "agibot_x2ultra.urdf"
T2_URDF = "T2_31dof.urdf"

Limits = dict[str, tuple[float, float]]

X2_RETARGETER_CFG = "soma_to_agibot_x2ultra_retargeter_config.json"
X2_SCALER_CFG = "soma_to_agibot_x2ultra_scaler_config.json"
X2_POST_CFG = "agibot_x2ultra_post_processing_config.json"
# SOMA joint -> (link the shipped X2 config targets, distal link to target instead).
# Shipped, the feet target `*_ankle_pitch_link` and the left hand
# `left_wrist_pitch_link` (the right hand already targets `right_wrist_roll_link`),
# so no effector constrains the ankle-roll or left wrist-roll joints and the IK leaves
# them at their reference values in every frame (G1's config targets the distal links).
X2_EFFECTOR_FIX = {
  "LeftFoot": ("left_ankle_pitch_link", "left_ankle_roll_link"),
  "RightFoot": ("right_ankle_pitch_link", "right_ankle_roll_link"),
  "LeftHand": ("left_wrist_pitch_link", "left_wrist_roll_link"),
}


def limit_factor(robot: str) -> float:
  """The robot's `soft_joint_pos_limit_factor` (mjlab articulation cfg)."""
  return float(get_robot(robot).entity_cfg_fn().articulation.soft_joint_pos_limit_factor)


def soft_limits(lim: Limits, factor: float) -> Limits:
  """mjlab's soft limits: each range shrunk about its midpoint by `factor`."""
  out = {}
  for name, (lo, hi) in lim.items():
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    out[name] = (mid - factor * half, mid + factor * half)
  return out


def urdf_limits(urdf: Path) -> Limits:
  """`{revolute joint: (lower, upper)}` in URDF order."""
  out = {}
  for j in ET.parse(urdf).iter("joint"):
    if j.get("type") == "revolute":
      lim = j.find("limit")
      out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
  return out


def _copy_target(src: Path, dst: Path, meshes: Path) -> None:
  """Copies a robot folder without its `desc/meshes` and links that to `meshes`."""
  if dst.is_symlink() or dst.is_file():
    dst.unlink()
  elif dst.exists():
    shutil.rmtree(dst)
  shutil.copytree(src, dst, ignore=lambda d, names: ["meshes"] if Path(d).name == "desc" else [])
  (dst / "desc" / "meshes").symlink_to(meshes.resolve(), target_is_directory=True)


def _override_limits(urdf: Path, lim: Limits) -> None:
  """Sets `<limit lower upper>` of every revolute joint (effort, velocity kept)."""
  tree = ET.parse(urdf, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
  revolute = [j for j in tree.iter("joint") if j.get("type") == "revolute"]
  names = {j.get("name") for j in revolute}
  if names != lim.keys():
    raise ValueError(
      f"{urdf} revolute joints != given limits: {sorted(names ^ lim.keys())}"
    )
  for joint in revolute:
    lo, hi = lim[joint.get("name")]
    joint.find("limit").set("lower", str(lo))
    joint.find("limit").set("upper", str(hi))
  tree.write(urdf, encoding="utf-8", xml_declaration=True)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of (x, y, z, w) quaternions (Warp's layout)."""
  (ax, ay, az, aw), (bx, by, bz, bw) = a, b
  return np.array(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ]
  )


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotates v by the (x, y, z, w) quaternion q."""
  conj = q * np.array([-1.0, -1.0, -1.0, 1.0])
  return _qmul(_qmul(q, np.append(v, 0.0)), conj)[:3]


def _child_joint(urdf: Path, link: str) -> ET.Element:
  for j in ET.parse(urdf).iter("joint"):
    if j.find("child").get("link") == link:
      return j
  raise ValueError(f"{urdf}: no joint has child link {link!r}")


def rebase_offset(offset: list, d_p: np.ndarray, d_q: np.ndarray) -> list:
  """A scaler offset re-expressed for a target link moved by (d_p, d_q) in the old one.

  The scaler's effector for a SOMA joint is t = base + rot(Q, p), Q = q_human * q
  (human_to_robot_scaler._compute_scaled_effectors), i.e. the old link's pose. The new
  link's pose is that pose composed with (d_p, d_q): Q' = Q d_q and t' = t + rot(Q, d_p),
  so q' = q d_q and p' = rot(d_q^-1, p + d_p).
  """
  p, q = np.asarray(offset[0], float), np.asarray(offset[1], float)
  d_inv = d_q * np.array([-1.0, -1.0, -1.0, 1.0])
  return [_qrot(d_inv, p + d_p).tolist(), _qmul(q, d_q).tolist()]


def fix_x2_effectors(x2: Path) -> None:
  """Re-targets the X2's feet and left hand to the distal links (`X2_EFFECTOR_FIX`).

  Edits the copied configs in place: the IK match table, the scaler offsets (rebased so
  the effectors still land where the shipped calibration put them, using each link's
  joint origin and the scaler's reference angle) and the limb stabiliser's foot
  effectors (G1 uses `*_ankle_roll_link` there too).
  """
  cfg_dir, urdf = x2 / "configs", x2 / "desc" / X2_URDF
  ret = json.loads((cfg_dir / X2_RETARGETER_CFG).read_text())
  scl = json.loads((cfg_dir / X2_SCALER_CFG).read_text())
  post = json.loads((cfg_dir / X2_POST_CFG).read_text())
  revolute = [
    j.get("name") for j in ET.parse(urdf).iter("joint") if j.get("type") == "revolute"
  ]
  ref_q = scl["reference_joint_q"][7:]  # after the free joint (xyz, quat)
  if len(ref_q) != len(revolute):
    raise ValueError(f"reference_joint_q has {len(ref_q)} joints, URDF {len(revolute)}")
  for soma, (old, new) in X2_EFFECTOR_FIX.items():
    entry = ret["ik_match_table"][soma]
    if (entry["t_body"], entry["r_body"]) != (old, old):
      raise ValueError(f"{soma} targets {entry['t_body']}/{entry['r_body']}, expected {old}")
    joint = _child_joint(urdf, new)
    if joint.find("parent").get("link") != old:
      raise ValueError(f"{new} is not a child of {old}")
    origin = joint.find("origin")
    if any(float(v) != 0.0 for v in origin.get("rpy", "0 0 0").split()):
      raise ValueError(f"{joint.get('name')}: non-zero origin rpy is not handled")
    d_p = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
    angle = ref_q[revolute.index(joint.get("name"))]
    d_q = np.append(axis * np.sin(angle / 2), np.cos(angle / 2))
    entry["t_body"] = entry["r_body"] = new
    scl["human_joint_offsets"][soma] = rebase_offset(
      scl["human_joint_offsets"][soma], d_p, d_q
    )
  effectors = post["limb_stabilizer"]["effectors"]
  renamed = {f"{s}_ankle_pitch_link": f"{s}_ankle_roll_link" for s in ("left", "right")}
  post["limb_stabilizer"]["effectors"] = {renamed.get(k, k): v for k, v in effectors.items()}
  for name, cfg in ((X2_RETARGETER_CFG, ret), (X2_SCALER_CFG, scl), (X2_POST_CFG, post)):
    (cfg_dir / name).write_text(json.dumps(cfg, indent=4) + "\n")


def _print_tree(root: Path) -> None:
  print(root)

  def walk(d: Path, indent: str) -> None:
    for p in sorted(d.iterdir()):
      if p.is_symlink():
        print(f"{indent}{p.name} -> {p.readlink()}")
      elif p.is_dir():
        print(f"{indent}{p.name}/")
        walk(p, indent + "  ")
      else:
        print(f"{indent}{p.name}")

  walk(root, "  ")


def assemble(retargeter_root: Path, out: Path, factor: float | None = None) -> None:
  """Builds `<out>/agibot/x2` and `<out>/booster/t2` from the retargeter clone.

  `factor` overrides both robots' soft-limit factor (default: from the registry).
  """
  x2_src = Path(retargeter_root) / X2_REL
  if not (x2_src / "manifest.json").is_file():
    raise FileNotFoundError(f"no X2 target at {x2_src}; is --retargeter the clone root?")
  out = Path(out)
  x2 = out / "agibot" / "x2"
  t2 = out / "booster" / "t2"
  x2.parent.mkdir(parents=True, exist_ok=True)
  t2.parent.mkdir(parents=True, exist_ok=True)

  f_x2 = limit_factor("x2") if factor is None else factor
  f_t2 = limit_factor("t2") if factor is None else factor
  print(f"soft-limit factor: x2 {f_x2}, t2 {f_t2}")

  _copy_target(x2_src, x2, x2_src / "desc" / "meshes")
  _override_limits(x2 / "desc" / X2_URDF, soft_limits(limits(X2_ULTRA_XML), f_x2))
  fix_x2_effectors(x2)
  _copy_target(T2_SRC, t2, T2_MESHES)
  # A copy, so the hard ranges come from the vendored URDF (== our MJCF to 5e-6 rad).
  t2_urdf = t2 / "desc" / T2_URDF
  _override_limits(t2_urdf, soft_limits(urdf_limits(T2_SRC / "desc" / T2_URDF), f_t2))
  _print_tree(out)


def main(retargeter: Path, out: Path, soft_factor: float | None = None) -> None:
  """Assemble the extra robot folder.

  Args:
    retargeter: root of the SOMA Retargeter clone (the folder with soma_retargeter/).
    out: folder to pass as the batch config's `extra_robot_paths` entry.
    soft_factor: experiments only: soft-limit factor for both robots (1.0 = hard
      limits). Default: each robot's `soft_joint_pos_limit_factor` from the registry.
  """
  assemble(retargeter, out, soft_factor)


if __name__ == "__main__":
  tyro.cli(main)
