"""Fit GMR's PICO-to-robot IK configs for the X2 and T2 (dev box; `gmr_retarget.py`).

GMR's IK targets each robot body at `R_human(joint) * offset` (and scaled positions).
Its PICO-to-G1 config (`xrobot_to_g1.json`) holds those offsets for the G1; we have no
PICO recordings, but our dataset has the same BONES-SEED clips retargeted to all three
robots. So:

1. A G1 clip gives the operator's PICO joint poses by inverting GMR's G1 config:
   R_human = R_g1(body) * offset_g1^-1, positions unscaled by its scale table.
2. For each PICO joint, the X2 (T2) offset is the mean of R_human^-1 * R_robot(body)
   over the frames of the same clips retargeted to that robot (the body mapping is our
   SOMA retargeter configs'); its spread says how well one constant offset fits. The
   scale of each joint is the least-squares ratio of robot to human positions
   relative to the root.
3. Validation on held-out clips: GMR with the fitted config on the synthetic PICO
   poses, against the dataset's joint angles; GMR on the G1 gives the IK's own error
   level.

The ankles: GMR's G1 config targets a toe body only its mocap model has; ours target
PICO's ankle joint at the ankle roll link (where our SOMA configs put the foot), with
the G1 leg offset for the synthesis.

Also the neutral stand that seeds the IK (`gmr_configs/stand.json`): each robot's
retarget of one SEED standing frame (the still, upright, two-feet-down frame nearest
the median pose; it is all of that on all three).

0. Arm posture (`gmr_retarget.ArmAbductionMap`): the robot's upper-arm abduction
   against the human's in the SEED BVHs the robot motions were retargeted from (both
   in a torso frame from positions), fitted as `robot = slope * human + offset` over
   every frame. Steps 2 and 3 use the mapped human.

  uv run --group teleop python -m deploy.vr_teleop.fit_gmr_configs --g1-data <dir> \\
    --x2-data <dir> --t2-data <dir> --bvh <dir> --clips <list.txt> [--validate 5] \\
    [--extra-fit <clip> ...] [--extra-validate <clip> ...]

`--bvh` mirrors the clip paths (`<date>/<stem>.bvh`, e.g.
`$DATA/seed/soma_uniform/bvh`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.vr_teleop.gmr_retarget import (
  CONFIG_DIR,
  REPO,
  STAND_FILE,
  ArmAbductionMap,
  GmrRetargeter,
  abduction_deg,
  torso_axes,
)

G1_CONFIG = REPO / "third_party/GMR/general_motion_retargeting/ik_configs/xrobot_to_g1.json"
SIDES = ("Left", "Right")
# PICO joint -> (G1 body in our data, G1 config body for its offset and weights,
# X2 body, T2 body). X2/T2 bodies: our SOMA retargeter configs' ik_match_table.
_BASE = {
  "Pelvis": ("pelvis", "pelvis", "pelvis", "waist_yaw_link"),
  "Spine3": ("torso_link", "torso_link", "torso_link", "trunk"),
}
_LIMBS = {
  "Hip": ("{s}_hip_yaw_link", "{s}_hip_yaw_link", "{s}_hip_yaw_link", "{s}_hip_yaw_link"),
  "Knee": ("{s}_knee_link", "{s}_knee_link", "{s}_knee_link", "{s}_knee_pitch_link"),
  "Ankle": ("{s}_ankle_roll_link", "{s}_toe_link", "{s}_ankle_roll_link", "{s}_ankle_roll_link"),
  # The G1 config targets the upper arm after its twist (shoulder yaw); the X2 has that
  # body, the T2 none (its arm twist is at the elbow yaw), so its shoulder rotation
  # target gets a low weight (T2_SHOULDER_ROT_WEIGHT).
  "Shoulder": ("{s}_shoulder_yaw_link", "{s}_shoulder_yaw_link", "{s}_shoulder_yaw_link",
               "{s}_shoulder_roll_link"),
  "Elbow": ("{s}_elbow_link", "{s}_elbow_link", "{s}_elbow_link", "{s}_elbow_yaw_link"),
  "Wrist": ("{s}_wrist_yaw_link", "{s}_wrist_yaw_link", "{s}_wrist_roll_link",
            "{s}_wrist_roll_link"),
}
JOINTS = dict(_BASE)
for _side in SIDES:
  for _part, _bodies in _LIMBS.items():
    JOINTS[f"{_side}_{_part}"] = tuple(b.format(s=_side.lower()) for b in _bodies)
ROBOT_COLUMN = {"x2": 2, "t2": 3}
T2_SHOULDER_ROT_WEIGHT = 1.0
# GMR's G1 config constrains the arms by orientation only. On the X2 (and T2) the same
# upper-arm orientation is reached on another shoulder branch (pitch, roll, yaw all off
# by 1-1.5 rad on a walk clip); a position target on the elbows and wrists rules it out.
ARM_POS_WEIGHT = 5.0
# The G1 config's scale-table key for each PICO joint (its feet are `*_Foot`).
G1_SCALE_KEY = {j: j.replace("Ankle", "Foot") for j in JOINTS}
# The neutral stand's source frame.
STAND_CLIP, STAND_FRAME = "230413/dance_hiphop_scooby_doo_R_002__A320.npz", 36
# Hand adjustments to the stand after the SEED frame (native joint -> rad, added): on the
# real G1 the frame leaned too far forward (2026-10-23; the torso 5.2 cm ahead of the
# ankles, the CoM 6.5 cm), so its ankles pitch 0.067 rad back (+ is toes down: with the
# feet flat, the body tilts back), which puts the torso over the ankles (0.0 cm; the CoM
# 2.3 cm ahead of them, the foot's middle is 4 cm ahead). `offset_stand` keeps the feet
# in place and tilts the body over them (the two ankle axes are not parallel in this
# stance, so the feet end within 0.8 deg of flat).
STAND_OFFSETS = {"g1": {"left_ankle_pitch_joint": 0.067, "right_ankle_pitch_joint": 0.067}}
# Pelvis, shoulder and elbow points of the abduction measurement (`{S}`/`{s}`: Left or
# left): the BVH joints, and per robot the bodies whose origins sit on its shoulder roll
# axis and elbow axis.
BVH_ARM = ("Hips", "{S}Arm", "{S}ForeArm")
ROBOT_ARM = {
  "g1": ("pelvis", "{s}_shoulder_roll_link", "{s}_elbow_link"),
  "x2": ("pelvis", "{s}_shoulder_roll_link", "{s}_elbow_link"),
  "t2": ("waist_yaw_link", "{s}_shoulder_roll_link", "{s}_elbow_pitch_link"),
}


def _load_g1_config() -> dict:
  return json.loads(G1_CONFIG.read_text())


def _g1_offset(cfg: dict, body: str) -> R:
  return R.from_quat(cfg["ik_match_table1"][body][4], scalar_first=True)


def _poses(path: Path, bodies: list[str]) -> tuple[np.ndarray, np.ndarray]:
  d = np.load(path)
  names = [str(n) for n in d["body_names"]]
  idx = [names.index(b) for b in bodies]
  return d["body_pos_w"][:, idx], d["body_quat_w"][:, idx]


def synth_human(cfg: dict, g1_clip: Path) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
  """PICO joint poses (GMR's human frame) of every frame of a G1 clip, from the G1
  config's offsets and scales."""
  return synth_frames(cfg, *_poses(g1_clip, [JOINTS[j][0] for j in JOINTS]))


def synth_frames(cfg: dict, pos: np.ndarray,
                 quat: np.ndarray) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
  """`synth_human` from the G1 body poses ([T, len(JOINTS), 3] and [..., 4] wxyz, the
  G1 bodies of `JOINTS` in its order)."""
  joints = list(JOINTS)
  scale = cfg["human_scale_table"]
  root = joints.index("Pelvis")
  frames = []
  for t in range(len(pos)):
    h_root = pos[t, root] / scale["Pelvis"]
    frame = {}
    for i, j in enumerate(joints):
      rot = R.from_quat(quat[t, i], scalar_first=True) * _g1_offset(cfg, JOINTS[j][1]).inv()
      p = h_root + (pos[t, i] - pos[t, root]) / scale[G1_SCALE_KEY[j]]
      frame[j] = (p, rot.as_quat(scalar_first=True))
    frames.append(frame)
  return frames


def load_bvh(path: Path) -> tuple[np.ndarray, dict[str, int], float]:
  """World joint positions [T, J, 3] of a BVH (forward kinematics of its channels),
  joint name -> index, and the frame time (s)."""
  header, motion = path.read_text().split("MOTION")
  names, parents, offsets, channels, stack = [], [], [], [], []
  for line in header.splitlines():
    tok = line.split()
    if not tok:
      continue
    if tok[0] in ("ROOT", "JOINT", "End"):
      names.append(tok[1] if tok[0] != "End" else None)
      parents.append(stack[-1] if stack else -1)
      offsets.append(np.zeros(3))
      channels.append([])
    elif tok[0] == "{":
      stack.append(len(names) - 1)
    elif tok[0] == "}":
      stack.pop()
    elif tok[0] == "OFFSET":
      offsets[-1] = np.array([float(x) for x in tok[1:4]])
    elif tok[0] == "CHANNELS":
      channels[-1] = tok[2:]
  lines = motion.strip().splitlines()
  frame_time = float(lines[1].split()[-1])
  values = np.array([[float(x) for x in line.split()] for line in lines[2:]])
  pos = np.zeros((len(values), len(names), 3))
  rots: list[R] = []
  col = 0
  for j, chans in enumerate(channels):
    local = np.tile(offsets[j], (len(values), 1))
    rot = R.identity(len(values))
    if chans:
      v = values[:, col:col + len(chans)]
      col += len(chans)
      for k, ch in enumerate(chans):
        if ch.endswith("position"):
          local[:, "XYZ".index(ch[0])] = v[:, k]
      axes = [(k, ch[0]) for k, ch in enumerate(chans) if ch.endswith("rotation")]
      if axes:
        rot = R.from_euler("".join(a for _, a in axes), v[:, [k for k, _ in axes]], degrees=True)
    if parents[j] < 0:
      pos[:, j], world = local, rot
    else:
      pos[:, j] = pos[:, parents[j]] + rots[parents[j]].apply(local)
      world = rots[parents[j]] * rot
    rots.append(world)
  return pos, {n: i for i, n in enumerate(names) if n}, frame_time


def _abduction(points: dict[str, np.ndarray]) -> np.ndarray:
  """[T, 2] left and right abduction (deg) from `{pelvis, Left/Right_{shoulder,elbow}}`."""
  _, left, up = torso_axes(points["pelvis"], points["Left_shoulder"], points["Right_shoulder"])
  return np.stack([abduction_deg(points[f"{s}_elbow"] - points[f"{s}_shoulder"], left, up,
                                 1.0 if s == "Left" else -1.0) for s in SIDES], -1)


def _arm_points(names: tuple[str, str, str], get) -> dict[str, np.ndarray]:
  out = {"pelvis": get(names[0])}
  for s in SIDES:
    out[f"{s}_shoulder"] = get(names[1].format(S=s, s=s.lower()))
    out[f"{s}_elbow"] = get(names[2].format(S=s, s=s.lower()))
  return out


def abduction_pairs(robot: str, clips: list[str], bvh_dir: Path, robot_dir: Path):
  """Human (BVH) and robot (dataset) abduction (deg) on a shared 50 Hz clock, both arms
  pooled: two flat arrays."""
  human, robo = [], []
  for clip in clips:
    pos, idx, frame_time = load_bvh(bvh_dir / Path(clip).with_suffix(".bvh"))
    d = np.load(robot_dir / clip)
    names = [str(n) for n in d["body_names"]]
    h = _abduction(_arm_points(BVH_ARM, lambda n: pos[:, idx[n]]))  # noqa: B023 (called at once)
    r = _abduction(_arm_points(ROBOT_ARM[robot], lambda n: d["body_pos_w"][:, names.index(n)]))  # noqa: B023 (called at once)
    t_h = np.arange(len(h)) * frame_time
    t_r = np.arange(len(r)) / float(d["fps"])
    t_r = t_r[t_r <= t_h[-1]]
    for side in range(2):
      human.append(np.interp(t_r, t_h, h[:, side]))
      robo.append(r[: len(t_r), side])
  return np.concatenate(human), np.concatenate(robo)


def fit_arm_map(robot: str, clips: list[str], bvh_dir: Path, robot_dir: Path) -> dict:
  """Least-squares `robot = slope * human + offset` abduction (deg) and its residual."""
  human, robo = abduction_pairs(robot, clips, bvh_dir, robot_dir)
  slope, offset = np.polyfit(human, robo, 1)
  return {"slope": float(slope), "offset_deg": float(offset),
          "residual_std_deg": float(np.std(robo - (slope * human + offset))),
          "human_median_deg": float(np.median(human)), "robot_median_deg": float(np.median(robo))}


def fit(cfg: dict, robot: str, clips: list[str], g1_dir: Path, robot_dir: Path,
        arm: ArmAbductionMap) -> dict:
  """Rotation offset, its spread (deg) and scale per PICO joint, on the mapped human."""
  col = ROBOT_COLUMN[robot]
  joints = list(JOINTS)
  rel: dict[str, list[R]] = {j: [] for j in joints}
  num = {j: 0.0 for j in joints}
  den = {j: 0.0 for j in joints}
  for clip in clips:
    human = [arm(frame) for frame in synth_human(cfg, g1_dir / clip)]
    pos, quat = _poses(robot_dir / clip, [JOINTS[j][col] for j in joints])
    n = min(len(human), len(pos))
    root = joints.index("Pelvis")
    for t in range(n):
      h_root = human[t]["Pelvis"][0]
      for i, j in enumerate(joints):
        h_pos, h_quat = human[t][j]
        rel[j].append(R.from_quat(h_quat, scalar_first=True).inv()
                      * R.from_quat(quat[t, i], scalar_first=True))
        a = pos[t, i] - pos[t, root] if j != "Pelvis" else pos[t, root]
        b = h_pos - h_root if j != "Pelvis" else h_root
        num[j] += float(a @ b)
        den[j] += float(b @ b)
  out = {}
  for j in joints:
    rots = R.concatenate(rel[j])
    mean = rots.mean()
    spread = np.degrees((mean.inv() * rots).magnitude())
    out[j] = {
      "offset": mean.as_quat(scalar_first=True).tolist(),
      "spread_deg_median": float(np.median(spread)),
      "spread_deg_p90": float(np.percentile(spread, 90)),
      "scale": num[j] / den[j],
    }
  return out


def make_config(cfg: dict, robot: str, fitted: dict, arm: dict) -> dict:
  """GMR config for `robot`: the G1 config's structure and weights, our bodies, fitted
  offsets and scales, and our arm abduction map (GMR ignores the key)."""
  col = ROBOT_COLUMN[robot]
  tables = {}
  for name in ("ik_match_table1", "ik_match_table2"):
    table = {}
    for j, bodies in JOINTS.items():
      g1 = cfg[name][bodies[1]]
      pos_weight, rot_weight = g1[1], g1[2]
      if robot == "t2" and j.endswith("_Shoulder"):
        rot_weight = T2_SHOULDER_ROT_WEIGHT
      if j.endswith(("_Elbow", "_Wrist")):
        pos_weight = ARM_POS_WEIGHT
      table[bodies[col]] = [j, pos_weight, rot_weight, [0.0, 0.0, 0.0], fitted[j]["offset"]]
    tables[name] = table
  return {
    "about": f"PICO (XRoboToolkit) to {robot}, fitted by deploy/vr_teleop/fit_gmr_configs.py"
             " from xrobot_to_g1.json and our dataset's retargets of the same clips.",
    "robot_root_name": JOINTS["Pelvis"][col],
    "human_root_name": "Pelvis",
    "ground_height": cfg["ground_height"],
    "human_height_assumption": cfg["human_height_assumption"],
    "use_ik_match_table1": cfg["use_ik_match_table1"],
    "use_ik_match_table2": cfg["use_ik_match_table2"],
    # The torso's position target has no weight; its fitted ratio (of two short
    # pelvis-torso offsets) is meaningless, so it takes the pelvis scale.
    "human_scale_table": {j: round(fitted["Pelvis" if j == "Spine3" else j]["scale"], 4)
                          for j in JOINTS},
    **tables,
    "arm_abduction_map": {k: round(arm[k], 4) for k in ("slope", "offset_deg")},
  }


def write_stand(data: dict[str, Path]) -> None:
  """`stand.json`: per robot, the STAND_CLIP frame of its retarget, turned to face +x
  with slot 0 (the pelvis-equivalent) at x = y = 0: `joint_pos` (native joint name ->
  angle, rad; GMR's IK seed), the free body's `base_pos` / `base_quat` (wxyz; where the
  sim stands the robot), `ankle_z` (the mean ankle height, which the streamer's
  calibration puts the operator's ankles at) and `state`, the reference state [240]
  of the stand at rest (the policy node's stance in IDLE)."""
  from deploy.sim2sim.obs import Robot, quat_inv, quat_mul, yaw_quat

  stand = {"source": f"{STAND_CLIP} frame {STAND_FRAME}"}
  for robot, path in data.items():
    d = np.load(path / STAND_CLIP)
    spec = Robot.load(robot)
    names = [str(n) for n in d["body_names"]]
    pos = d["body_pos_w"][STAND_FRAME].astype(np.float64)
    quat = d["body_quat_w"][STAND_FRAME].astype(np.float64)
    slot0 = names.index(spec.tracked_bodies[0])
    turn = quat_inv(yaw_quat(quat[slot0]))
    origin = np.array([pos[slot0, 0], pos[slot0, 1], 0.0])
    pos = R.from_quat(turn, scalar_first=True).apply(pos - origin)
    quat = quat_mul(turn[None], quat)
    tracked = [names.index(b) for b in spec.tracked_bodies]
    q = d["joint_pos"][STAND_FRAME].astype(np.float64)
    zeros = np.zeros(3 * len(tracked))
    state = np.concatenate((spec.to_canonical(q), np.zeros(len(spec.perm)),
                            pos[tracked].ravel(), quat[tracked].ravel(), zeros, zeros))
    ankles = [names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
    stand[robot] = {
      "joint_pos": {str(n): round(float(a), 5) for n, a in zip(d["joint_names"], q, strict=True)},
      "base_pos": np.round(pos[0], 5).tolist(),
      "base_quat": np.round(quat[0], 6).tolist(),
      "ankle_z": round(float(pos[ankles, 2].mean()), 5),
      "state": np.round(state, 6).tolist(),
    }
    if robot in STAND_OFFSETS:
      stand[robot] = offset_stand(robot, stand[robot], STAND_OFFSETS[robot])
  STAND_FILE.write_text(json.dumps(stand, indent=1) + "\n")


def _stand_model(robot: str):
  import mujoco

  if robot != "g1":
    raise NotImplementedError(f"no stand model for {robot}")
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

  return mujoco.MjModel.from_xml_path(str(G1_XML))


def offset_stand(robot: str, entry: dict, offsets: dict[str, float],
                 max_feet_err: float = 2e-3) -> dict:
  """The `stand.json` entry with `offsets` added to its joints, the feet kept where they
  were (the free body re-solved by least squares on both feet's poses) and everything
  derived again: `base_pos` / `base_quat`, `ankle_z` and `state`, turned to face +x
  with slot 0 at x = y = 0 as `write_stand` does. Checks that the model reproduces the
  entry's own state first."""
  import mujoco
  from scipy.optimize import least_squares

  from deploy.sim2sim.obs import Robot, quat_inv, quat_mul, yaw_quat

  spec = Robot.load(robot)
  model = _stand_model(robot)
  data = mujoco.MjData(model)
  names = spec.native_joint_names
  adr = [model.joint(n).qposadr[0] for n in names]
  tracked = [model.body(b).id for b in spec.tracked_bodies]
  feet = [model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

  def fk(base_pos, base_quat, q):
    data.qpos[:3], data.qpos[3:7] = base_pos, base_quat
    data.qpos[adr] = q
    mujoco.mj_kinematics(model, data)
    return data.xpos.copy(), data.xquat.copy()

  q0 = np.array([entry["joint_pos"][n] for n in names])
  base_pos, base_quat = np.asarray(entry["base_pos"]), np.asarray(entry["base_quat"])
  pos0, quat0 = fk(base_pos, base_quat, q0)
  old = np.asarray(entry["state"])
  err = np.abs(pos0[tracked].ravel() - old[58:100]).max()
  if err > 1e-3:
    raise ValueError(f"{robot}: the model's FK is {err:.2g} m off the stand's state")

  q = q0.copy()
  for n, d in offsets.items():
    q[names.index(n)] += d
  rot0 = R.from_quat(base_quat, scalar_first=True)
  feet0 = [(pos0[f], R.from_quat(quat0[f], scalar_first=True)) for f in feet]

  def base(x):
    return base_pos + x[:3], (R.from_rotvec(x[3:]) * rot0).as_quat(scalar_first=True)

  def residual(x):
    pos, quat = fk(*base(x), q)
    return np.concatenate([
      np.concatenate((pos[f] - p,
                      0.1 * (R.from_quat(quat[f], scalar_first=True) * r.inv()).as_rotvec()))
      for f, (p, r) in zip(feet, feet0, strict=True)])

  sol = least_squares(residual, np.zeros(6), xtol=1e-12, ftol=1e-12)
  if np.abs(sol.fun).max() > max_feet_err:
    raise ValueError(f"{robot}: the feet moved {np.abs(sol.fun).max():.2g} with {offsets}")
  pos, quat = fk(*base(sol.x), q)
  slot0 = tracked[0]
  turn = quat_inv(yaw_quat(quat[slot0]))
  origin = np.array([pos[slot0, 0], pos[slot0, 1], 0.0])
  pos = R.from_quat(turn, scalar_first=True).apply(pos - origin)
  quat = quat_mul(turn[None], quat)
  zeros = np.zeros(3 * len(tracked))
  state = np.concatenate((spec.to_canonical(q), np.zeros(len(spec.perm)),
                          pos[tracked].ravel(), quat[tracked].ravel(), zeros, zeros))
  root = model.jnt_bodyid[0]  # The free body: `base_pos` / `base_quat`.
  return {
    **entry,
    "joint_pos": {n: round(float(a), 5) for n, a in zip(names, q, strict=True)},
    "base_pos": np.round(pos[root], 5).tolist(),
    "base_quat": np.round(quat[root], 6).tolist(),
    "ankle_z": round(float(pos[feet, 2].mean()), 5),
    "state": np.round(state, 6).tolist(),
    "offsets": offsets,
  }


def validate(cfg: dict, robot: str, clips: list[str], g1_dir: Path, data_dir: Path) -> dict:
  """GMR on the synthetic PICO poses of held-out clips: RMS joint error (rad) against
  the dataset's retarget of each clip, all joints and per limb group, and the median
  upper-arm abduction (deg) of GMR's output and of the dataset."""
  errs: dict[str, list[float]] = {}
  abd_gmr, abd_data = [], []
  for clip in clips:
    gmr = GmrRetargeter(robot)
    data = gmr.gmr.configuration.data
    clip_errs: list[float] = []
    human = synth_human(cfg, g1_dir / clip)
    d = np.load(data_dir / clip)
    names = [str(n) for n in d["joint_names"]]
    bodies = [str(n) for n in d["body_names"]]
    n = min(len(human), len(d["joint_pos"]))
    abd_data.append(_abduction(_arm_points(
      ROBOT_ARM[robot], lambda b: d["body_pos_w"][:n, bodies.index(b)])).ravel())  # noqa: B023 (called at once)
    points = []
    for t, frame in enumerate(human[:n]):
      # GMR's G1 config targets the feet (`*_Foot`, at its toe bodies); the ankle pose
      # stands in for them.
      frame = {**frame, **{f"{s}_Foot": frame[f"{s}_Ankle"] for s in SIDES}}
      pose = gmr.settle(frame) if t == 0 else gmr.retarget(frame)
      points.append(_arm_points(ROBOT_ARM[robot], lambda b: data.xpos[gmr.gmr.model.body(b).id].copy()))  # noqa: B023 (called at once)
      for k, name in enumerate(names):
        if name in pose.joint_pos and "head" not in name:
          group = next((g for g in ("hip", "knee", "ankle", "waist", "shoulder", "elbow",
                                    "wrist") if g in name), "other")
          errs.setdefault(group, []).append(pose.joint_pos[name] - d["joint_pos"][t, k])
          clip_errs.append(pose.joint_pos[name] - d["joint_pos"][t, k])
    abd_gmr.append(_abduction({k: np.array([p[k] for p in points]) for k in points[0]}).ravel())
    print(f"[validate] {robot} {Path(clip).stem[:40]}: RMS {np.sqrt(np.mean(np.square(clip_errs))):.3f} rad")
  out = {g: float(np.sqrt(np.mean(np.square(e)))) for g, e in errs.items()}
  out["all"] = float(np.sqrt(np.mean(np.square(np.concatenate([np.array(e) for e in errs.values()])))))
  out["abduction_gmr_median_deg"] = float(np.median(np.concatenate(abd_gmr)))
  out["abduction_data_median_deg"] = float(np.median(np.concatenate(abd_data)))
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--g1-data", type=Path, required=True)
  parser.add_argument("--x2-data", type=Path, required=True)
  parser.add_argument("--t2-data", type=Path, required=True)
  parser.add_argument("--bvh", type=Path, required=True, help="SEED BVH root (<date>/<stem>.bvh)")
  parser.add_argument("--clips", type=Path, required=True, help="clip list, one per line")
  parser.add_argument("--validate", type=int, default=2, help="held-out clips to validate on")
  parser.add_argument("--fit-clips", type=int, default=40, help="clips to fit on")
  parser.add_argument("--extra-fit", nargs="*", default=[], help="more clips to fit on")
  parser.add_argument("--extra-validate", nargs="*", default=[], help="more held-out clips")
  parser.add_argument("--stand-only", action="store_true", help="only write stand.json")
  args = parser.parse_args()
  if args.stand_only:
    write_stand({"g1": args.g1_data, "x2": args.x2_data, "t2": args.t2_data})
    return
  cfg = _load_g1_config()
  clips = [c for c in args.clips.read_text().split() if c]
  held_out = clips[:: max(1, len(clips) // args.validate)][: args.validate]
  fit_clips = [c for c in clips if c not in held_out][: args.fit_clips] + args.extra_fit
  held_out += args.extra_validate
  CONFIG_DIR.mkdir(exist_ok=True)
  write_stand({"g1": args.g1_data, "x2": args.x2_data, "t2": args.t2_data})
  report = {}
  for robot, data in (("x2", args.x2_data), ("t2", args.t2_data)):
    arm = fit_arm_map(robot, fit_clips, args.bvh, data)
    print(f"[fit] {robot} arm abduction: robot = {arm['slope']:.2f} * human + "
          f"{arm['offset_deg']:.1f} deg (residual {arm['residual_std_deg']:.1f}; medians human "
          f"{arm['human_median_deg']:.1f}, robot {arm['robot_median_deg']:.1f})")
    fitted = fit(cfg, robot, fit_clips, args.g1_data, data,
                 ArmAbductionMap(arm["slope"], arm["offset_deg"]))
    (CONFIG_DIR / f"xrobot_to_{robot}.json").write_text(
      json.dumps(make_config(cfg, robot, fitted, arm), indent=1) + "\n")
    report[robot] = {"arm_abduction_map": arm, "fit": fitted}
    print(f"[fit] {robot}: {len(fit_clips)} clips")
    for j, f in fitted.items():
      print(f"   {j:15s} scale {f['scale']:.3f}  offset spread median {f['spread_deg_median']:5.1f}"
            f" deg, p90 {f['spread_deg_p90']:5.1f} deg")
  for robot, data in (("g1", args.g1_data), ("x2", args.x2_data), ("t2", args.t2_data)):
    report.setdefault(robot, {})["validation_rms_rad"] = v = validate(
      cfg, robot, held_out, args.g1_data, data)
    print(f"[validate] {robot} on {len(held_out)} held-out clips, RMS joint error (rad): "
          + ", ".join(f"{k} {x:.3f}" for k, x in v.items()))
  (CONFIG_DIR / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
  main()
