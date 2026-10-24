"""Derive the Booster T2 scaler calibration (robotics/booster/t2/configs/*_scaler_config.json).

1. Heights: MuJoCo forward kinematics of our MJCFs at the canonical zero pose, with
   the lowest foot-geom point placed at z = 0 (`heights`).
2. `human_scale_ratio`: the configurator's "Pelvis" auto-scale, i.e. the height of the
   body mapped to `Hips` (the T2's `waist_yaw_link`) over the SOMA `Hips` height, taken
   relative to the X2: `X2 ratio * T2 waist_yaw_link / X2 pelvis`.
3. `reference_joint_q`: the free joint at the T2 trunk height (identity, xyzw), then the
   canonical zero pose in URDF joint order, clamped to the soft joint limits the IK
   uses (`seed_pose`): the URDF ranges shrunk by the T2's soft_joint_pos_limit_factor,
   as assemble_robots.py writes them. The knees are not clamped: they keep the
   canonical 0 so that a straight human knee maps to a straight robot knee.
4. `human_joint_offsets`: the configurator's `_compute_local_offset`, between the scaled
   SOMA zero pose and the T2 URDF links at the seed pose. The SOMA zero pose (at scale 1,
   `soma_zero_pose.json`) is recovered by inverting the retargeter's shipped X2
   calibration; `--retargeter <clone>` recomputes that file and checks the recovery by
   reproducing the shipped G1 offsets.

  uv run python tools/soma_retargeter/derive_t2.py            # print the derivation
  uv run python tools/soma_retargeter/derive_t2.py --write    # rewrite the scaler config
  uv run python tools/soma_retargeter/derive_t2.py --retargeter <soma-retargeter clone>
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from assemble_robots import limit_factor, soft_limits
from scipy.spatial.transform import Rotation as Rot

from morphmimic.robots import get_robot

HERE = Path(__file__).resolve().parent
T2_DIR = HERE / "robotics" / "booster" / "t2"
T2_URDF = T2_DIR / "desc" / "T2_31dof.urdf"
T2_SCALER = T2_DIR / "configs" / "soma_to_booster_t2_scaler_config.json"
T2_RETARGETER = T2_DIR / "configs" / "soma_to_booster_t2_retargeter_config.json"
SOMA_ZERO_POSE = HERE / "soma_zero_pose.json"
ROBOTICS = Path("soma_retargeter") / "assets" / "robotics"
X2_CFG = ROBOTICS / "agibot" / "x2" / "configs"
X2_URDF = ROBOTICS / "agibot" / "x2" / "desc" / "agibot_x2ultra.urdf"
G1_CFG = ROBOTICS / "unitree" / "g1" / "configs"
# The retargeter's shipped X2 scale (soma_to_agibot_x2ultra_scaler_config.json @ cbdb2b4).
X2_RATIO = 0.6686341166496277
# `human_joint_offsets` keys in the T1 scaler's order.
OFFSET_JOINTS = (
  "Hips", "Head", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot",
  "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "Chest",
)  # fmt: skip

Pose = dict[str, tuple[np.ndarray, Rot]]


def _json(path: Path) -> dict:
  with open(path) as f:
    return json.load(f)


def zero_pose(name: str) -> dict[str, float]:
  """The robot's canonical zero pose as `{native joint: angle}`."""
  robot = get_robot(name)
  native = robot.translator.to_native(torch.zeros(29), torch.zeros(len(robot.joint_names)))
  return dict(zip(robot.joint_names, native.tolist(), strict=True))


def _lowest_z(m: mujoco.MjModel, d: mujoco.MjData, gid: int) -> float:
  t, p, s = m.geom_type[gid], d.geom_xpos[gid], m.geom_size[gid]
  rot = d.geom_xmat[gid].reshape(3, 3)
  if t == mujoco.mjtGeom.mjGEOM_SPHERE:
    return p[2] - s[0]
  if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return p[2] - abs(rot[2, 2]) * s[1] - s[0]
  if t == mujoco.mjtGeom.mjGEOM_BOX:
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return float((p + (corners * s) @ rot.T)[:, 2].min())
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = m.geom_dataid[gid]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return float((p + m.mesh_vert[a : a + n] @ rot.T)[:, 2].min())
  raise ValueError(f"unsupported foot geom type {t}")


def _mj_pose(name: str, z: float, qpos: dict[str, float]) -> tuple[mujoco.MjModel, mujoco.MjData]:
  m = get_robot(name).entity_cfg_fn().spec_fn().compile()
  d = mujoco.MjData(m)
  d.qpos[:] = 0.0
  d.qpos[2] = z
  d.qpos[3] = 1.0
  for joint, v in qpos.items():
    d.qpos[m.jnt_qposadr[m.joint(joint).id]] = v
  mujoco.mj_kinematics(m, d)
  return m, d


def heights(name: str, bodies: list[str]) -> dict[str, float]:
  """Heights of `bodies` over the lowest foot-geom point, at the canonical zero pose."""
  m, d = _mj_pose(name, 0.0, zero_pose(name))
  foot = re.compile(get_robot(name).foot_geom_regex)
  feet = [g for g in range(m.ngeom) if foot.match(m.geom(g).name)]
  ground = min(_lowest_z(m, d, g) for g in feet)
  return {b: float(d.body(b).xpos[2] - ground) for b in bodies}


def _floats(text: str | None) -> np.ndarray:
  return np.array([float(v) for v in (text or "0 0 0").split()])


def revolute_limits(urdf: Path) -> dict[str, tuple[float, float]]:
  """`{revolute joint: (lower, upper)}` in URDF order."""
  out = {}
  for j in ET.parse(urdf).iter("joint"):
    if j.get("type") == "revolute":
      lim = j.find("limit")
      out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
  return out


def urdf_fk(urdf: Path, joint_q: list[float]) -> Pose:
  """World `{link: (position, rotation)}` for a floating-base URDF at Newton `joint_q`."""
  root = ET.parse(urdf).getroot()
  joints = list(root.iter("joint"))
  revolute = [j.get("name") for j in joints if j.get("type") == "revolute"]
  if len(revolute) != len(joint_q) - 7:
    raise ValueError(f"{urdf}: {len(revolute)} revolute joints, {len(joint_q) - 7} values")
  q = dict(zip(revolute, joint_q[7:], strict=True))
  children = {j.find("child").get("link") for j in joints}
  (base,) = [link.get("name") for link in root.iter("link") if link.get("name") not in children]
  pose: Pose = {base: (np.array(joint_q[:3], dtype=float), Rot.from_quat(joint_q[3:7]))}
  todo = list(joints)
  while todo:
    j = todo.pop(0)
    parent = j.find("parent").get("link")
    if parent not in pose:
      todo.append(j)
      continue
    origin = j.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None)
    rpy = _floats(origin.get("rpy") if origin is not None else None)
    rot = Rot.from_euler("xyz", rpy)  # URDF rpy: Rz(y) Ry(p) Rx(r)
    if j.get("type") == "revolute":
      axis = _floats(j.find("axis").get("xyz"))
      rot = rot * Rot.from_rotvec(axis / np.linalg.norm(axis) * q[j.get("name")])
    pp, pr = pose[parent]
    pose[j.find("child").get("link")] = (pp + pr.apply(xyz), pr * rot)
  return pose


def local_offset(soma: tuple[np.ndarray, Rot], robot: tuple[np.ndarray, Rot]) -> tuple:
  """The configurator's `_compute_local_offset(source=soma, target=robot)`."""
  (sp, sr), (rp, rr) = soma, robot
  return rr.inv().apply(rp - sp), sr.inv() * rr


def _apply_inverse(robot: tuple[np.ndarray, Rot], offset: list) -> tuple[np.ndarray, Rot]:
  """The SOMA joint transform that `offset` maps onto `robot`."""
  (rp, rr), (op, oq) = robot, offset
  return rp - rr.apply(op), rr * Rot.from_quat(oq).inv()


def recover_soma_zero_pose(retargeter: Path) -> Pose:
  """SOMA zero pose at scale 1, from the retargeter's shipped X2 calibration."""
  scaler = _json(retargeter / X2_CFG / "soma_to_agibot_x2ultra_scaler_config.json")
  table = _json(retargeter / X2_CFG / "soma_to_agibot_x2ultra_retargeter_config.json")
  links = urdf_fk(retargeter / X2_URDF, scaler["reference_joint_q"])
  ratio = scaler["human_scale_ratio"]
  if ratio != X2_RATIO:
    raise ValueError(f"X2 human_scale_ratio {ratio} != X2_RATIO {X2_RATIO}; update derive_t2.py")
  out = {}
  for joint, v in table["ik_match_table"].items():
    p, r = _apply_inverse(links[v["t_body"]], scaler["human_joint_offsets"][joint])
    out[joint] = (p / ratio, r)
  return out


def save_soma_zero_pose(pose: Pose, path: Path = SOMA_ZERO_POSE) -> None:
  doc = {
    "source": "SOMA Retargeter cbdb2b4, agibot/x2 scaler + URDF, inverted (derive_t2.py)",
    "joints": {k: {"p": p.tolist(), "q": r.as_quat().tolist()} for k, (p, r) in pose.items()},
  }
  path.write_text(json.dumps(doc, indent=2) + "\n")


def load_soma_zero_pose(path: Path = SOMA_ZERO_POSE) -> Pose:
  doc = _json(path)["joints"]
  return {k: (np.array(v["p"]), Rot.from_quat(v["q"])) for k, v in doc.items()}


def check_g1(retargeter: Path, soma: Pose) -> dict[str, tuple[float, float]]:
  """`{joint: (|dp| m, dq deg)}` between G1 offsets from `soma` and the shipped ones."""
  scaler = _json(retargeter / G1_CFG / "soma_to_g1_scaler_config.json")
  table = _json(retargeter / G1_CFG / "soma_to_g1_retargeter_config.json")
  q = scaler["reference_joint_q"]
  g1 = get_robot("g1")
  _, d = _mj_pose("g1", q[2], dict(zip(g1.joint_names, q[7:], strict=True)))
  ratio = scaler["human_scale_ratio"]
  out = {}
  for joint, v in table["ik_match_table"].items():
    body = d.body(v["t_body"])
    robot = (body.xpos.copy(), Rot.from_quat(body.xquat[[1, 2, 3, 0]]))
    p, r = local_offset((soma[joint][0] * ratio, soma[joint][1]), robot)
    sp, sq = scaler["human_joint_offsets"][joint]
    dq = (Rot.from_quat(sq).inv() * r).magnitude()
    out[joint] = (float(np.linalg.norm(p - sp)), float(np.degrees(dq)))
  return out


# Seed joints that keep the canonical zero instead of the soft clamp (see `seed_pose`).
SEED_UNCLAMPED = ("left_knee_pitch_joint", "right_knee_pitch_joint")


def seed_pose(urdf: Path = T2_URDF) -> list[float]:
  """The T2 canonical zero in URDF joint order, clamped to the IK's soft limits.

  Every joint is clamped to its soft limit except the knees (`SEED_UNCLAMPED`), which
  keep the canonical 0 rad (the hard lower limit, below the soft lower limit 0.129).
  The offsets are computed at this seed, so a clamped seed knee would map a straight
  human knee to a 0.129 rad robot knee and bend every retargeted frame by that much
  more (v2 data: T2 knee median 0.375 rad vs 0.259 for the G1). With the seed knee at
  0 the IK still keeps every frame inside the soft limits (the URDF limits it solves
  with), and a standing T2 holds its knees at about 0.15 rad, like the X2.
  """
  zero = zero_pose("t2")
  lim = soft_limits(revolute_limits(urdf), limit_factor("t2"))
  return [
    zero[n] if n in SEED_UNCLAMPED else min(max(zero[n], lo), hi) for n, (lo, hi) in lim.items()
  ]


def derive(soma: Pose | None = None) -> dict:
  """The T2 scaler values (rounded as written) and the heights they come from."""
  soma = load_soma_zero_pose() if soma is None else soma
  t2 = heights("t2", ["trunk", "waist_yaw_link"])
  x2 = heights("x2", ["pelvis"])
  ratio = X2_RATIO * t2["waist_yaw_link"] / x2["pelvis"]
  ref = [0.0, 0.0, t2["trunk"], 0.0, 0.0, 0.0, 1.0, *seed_pose()]
  links = urdf_fk(T2_URDF, ref)
  table = _json(T2_RETARGETER)["ik_match_table"]
  scaled = {k: (p * ratio, r) for k, (p, r) in soma.items()}
  # Place the scaled SOMA root horizontally on the body mapped to Hips.
  shift = links[table["Hips"]["t_body"]][0] - scaled["Hips"][0]
  shift[2] = 0.0
  offsets = {}
  for joint in OFFSET_JOINTS:
    p, r = local_offset((scaled[joint][0] + shift, scaled[joint][1]), links[table[joint]["t_body"]])
    offsets[joint] = [[round(float(x), 4) for x in p], [round(float(x), 4) for x in r.as_quat()]]
  return {
    "heights": {
      "t2_trunk": t2["trunk"],
      "t2_waist_yaw_link": t2["waist_yaw_link"],
      "x2_pelvis": x2["pelvis"],
    },
    "soma_hips_height": float(soma["Hips"][0][2]),
    "soma_shift_x": float(shift[0]),
    "human_scale_ratio": round(ratio, 6),
    "reference_joint_q": [round(v, 10) for v in ref],
    "human_joint_offsets": offsets,
  }


def write_scaler(values: dict, path: Path = T2_SCALER) -> None:
  doc = {
    "human_type": "soma",
    "human_root": "Hips",
    "robot_type": "booster_t2",
    "robot_root": "trunk",
    "human_scale_ratio": values["human_scale_ratio"],
    "reference_joint_q": values["reference_joint_q"],
    "human_joint_offsets": values["human_joint_offsets"],
  }
  path.write_text(json.dumps(doc, indent=4) + "\n")


def main(write: bool = False, retargeter: Path | None = None) -> None:
  """Print the T2 scaler derivation.

  Args:
    write: rewrite robotics/booster/t2/configs/soma_to_booster_t2_scaler_config.json.
    retargeter: SOMA Retargeter clone root; if given, recover soma_zero_pose.json from
      its X2 calibration (rewritten with --write) and check it against the G1's.
  """
  soma = None
  if retargeter is not None:
    soma = recover_soma_zero_pose(Path(retargeter).expanduser())
    print("G1 check (|dp| m, dq deg; the G1 hand/foot translations are hand-zeroed):")
    for joint, (dp, dq) in check_g1(Path(retargeter).expanduser(), soma).items():
      print(f"  {joint:13s} {dp:.4f} {dq:.3f}")
    if write:
      save_soma_zero_pose(soma)
      print(f"wrote {SOMA_ZERO_POSE}")
  v = derive(soma)
  h = v["heights"]
  print(f"T2 trunk height        {h['t2_trunk']:.6f}")
  print(f"T2 waist_yaw_link      {h['t2_waist_yaw_link']:.6f}")
  print(f"X2 pelvis height       {h['x2_pelvis']:.6f}")
  print(f"SOMA Hips height       {v['soma_hips_height']:.4f}")
  print(f"human_scale_ratio      {v['human_scale_ratio']}  (by trunk it would be "
        f"{X2_RATIO * h['t2_trunk'] / h['x2_pelvis']:.4f})")
  print(f"SOMA shift onto pelvis {v['soma_shift_x']:+.4f} m in x")
  zero = zero_pose("t2")
  for name, q in zip(revolute_limits(T2_URDF), v["reference_joint_q"][7:], strict=True):
    if not math.isclose(q, zero[name], abs_tol=1e-9):
      print(f"seed clamped: {name} {zero[name]:+.4f} -> {q:+.4f}")
  print("human_joint_offsets:")
  for joint, off in v["human_joint_offsets"].items():
    print(f"  {joint:13s} {off}")
  if write:
    write_scaler(v)
    print(f"wrote {T2_SCALER}")


if __name__ == "__main__":
  tyro.cli(main)
