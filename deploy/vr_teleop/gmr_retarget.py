"""PICO body -> full robot pose with GMR (General Motion Retargeting, `third_party/GMR`).

GMR solves an IK per frame that fits the robot's bodies to the operator's 24 PICO body
joints (its `xrobot` source). It ships a config for the G1; the X2 and T2 configs are
ours (`gmr_configs/xrobot_to_{x2,t2}.json`, fitted by `fit_gmr_configs.py` against our
dataset's retargets of the same clips). `register` adds our robots to GMR's tables, so
the submodule stays unmodified.

`GmrRetargeter(robot).retarget(body)` takes GMR's human frame (`{joint: (pos, quat
wxyz)}`, as `XRobotStreamer.get_processed_body_data` gives it) and returns the robot's
native joint positions and floating base.

Arm posture: our X2 references (NVIDIA's SOMA Retargeter, its bundled X2 config) hold
the upper arms out from the torso: ~42 deg of abduction (median) where the SEED human
has ~16, following it weakly (0.44 * human + 35 deg) until the two meet at ~62 deg. GMR
follows the operator, which on the X2 is near its shoulder-roll limit and outside the
policy's training data. (The G1 and T2 references follow the human.) A config's
`arm_abduction_map` (fitted by `fit_gmr_configs.py` from the SEED BVHs) rotates the
operator's arm about the shoulder to the dataset's abduction before the IK
(`ArmAbductionMap`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent / "gmr_configs"
# The neutral stand that seeds the IK (`GmrRetargeter.settle`): per robot, its retarget
# of one SEED standing frame, written by
# `fit_gmr_configs.py`.
STAND_FILE = CONFIG_DIR / "stand.json"
# GMR's name for each robot, and the MJCF its IK runs on. The G1 uses GMR's own model
# and config (its mocap variant has the toe bodies the config targets); the X2 and T2
# use our training MJCFs.
GMR_ROBOT = {"g1": "unitree_g1", "x2": "agibot_x2_ultra", "t2": "booster_t2"}
ROBOT_XML = {
  "x2": REPO / "src/morphmimic/robots/agibot_x2_ultra/xmls/x2_ultra.xml",
  "t2": REPO / "src/morphmimic/robots/booster_t2/xmls/t2.xml",
}


def register() -> None:
  """Adds the X2 and T2 to GMR's robot and IK-config tables (idempotent)."""
  from general_motion_retargeting import params

  for robot, xml in ROBOT_XML.items():
    params.ROBOT_XML_DICT[GMR_ROBOT[robot]] = xml
    params.IK_CONFIG_DICT["xrobot"][GMR_ROBOT[robot]] = CONFIG_DIR / f"xrobot_to_{robot}.json"


SIDES = (("Left", 1.0), ("Right", -1.0))


def torso_axes(pelvis: np.ndarray, l_shoulder: np.ndarray, r_shoulder: np.ndarray):
  """Forward, left and up unit vectors of a torso frame built from positions alone (up:
  pelvis to mid-shoulders; left: across the shoulders), so a human and a robot
  skeleton share its definition. Inputs [..., 3]."""
  up = (l_shoulder + r_shoulder) / 2 - pelvis
  up = up / np.linalg.norm(up, axis=-1, keepdims=True)
  left = l_shoulder - r_shoulder
  left = left - (left * up).sum(-1, keepdims=True) * up
  left = left / np.linalg.norm(left, axis=-1, keepdims=True)
  return np.cross(left, up), left, up


def abduction_deg(upper_arm: np.ndarray, left: np.ndarray, up: np.ndarray, sign: float):
  """Upper-arm angle out from hanging straight down, in the torso's frontal plane
  (outward positive; `sign` +1 for the left arm, -1 for the right)."""
  return np.degrees(np.arctan2(sign * (upper_arm * left).sum(-1), -(upper_arm * up).sum(-1)))


@dataclass(frozen=True)
class ArmAbductionMap:
  """Moves the operator's upper-arm abduction `a` to `slope * a + offset_deg` where that
  is larger (relaxed arms), and leaves it above the crossing (`a >= offset / (1 -
  slope)`: big raises pass through, as in the references). Across the body
  (a < 0) the shift is the a = 0 one, tapering to none at a = -180 deg (overhead), and
  it fades as the arm points along the torso's forward axis (where a rotation about
  that axis would only twist it). The rotation is about the shoulder: the upper arm
  and forearm (orientations, elbow and wrist positions) turn with it, the hand keeps
  the operator's orientation (our references follow it to ~2 deg)."""

  slope: float
  offset_deg: float

  def shift_deg(self, a: np.ndarray) -> np.ndarray:
    raised = np.maximum(0.0, (self.slope - 1.0) * a + self.offset_deg)
    across = self.offset_deg * (1.0 + a / 180.0)
    return np.where(a >= 0.0, raised, across)

  def __call__(self, body: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    out = dict(body)
    fwd, left, up = torso_axes(body["Pelvis"][0], body["Left_Shoulder"][0],
                               body["Right_Shoulder"][0])
    for side, sign in SIDES:
      shoulder = body[f"{side}_Shoulder"][0]
      upper = body[f"{side}_Elbow"][0] - shoulder
      frontal = np.hypot((upper * left).sum(), (upper * up).sum()) / np.linalg.norm(upper)
      shift = self.shift_deg(abduction_deg(upper, left, up, sign)) * frontal
      turn = R.from_rotvec(np.radians(sign * shift) * fwd)
      for part in ("Shoulder", "Elbow", "Wrist", "Hand"):
        name = f"{side}_{part}"
        if name not in body:
          continue
        pos, quat = body[name]
        pos = shoulder + turn.apply(pos - shoulder)
        if part in ("Shoulder", "Elbow"):
          quat = (turn * R.from_quat(quat, scalar_first=True)).as_quat(scalar_first=True)
        out[name] = (pos, quat)
    return out


def arm_map(robot: str) -> ArmAbductionMap | None:
  """The robot's `arm_abduction_map` (None for the G1, whose config is GMR's own)."""
  path = CONFIG_DIR / f"xrobot_to_{robot}.json"
  if robot not in ROBOT_XML or not path.exists():
    return None
  spec = json.loads(path.read_text()).get("arm_abduction_map")
  return ArmAbductionMap(spec["slope"], spec["offset_deg"]) if spec else None


@dataclass
class RobotPose:
  base_pos: np.ndarray  # [3]
  base_quat: np.ndarray  # [4] wxyz
  joint_pos: dict[str, float]  # native joint name -> angle


class GmrRetargeter:
  """GMR's IK for one robot, warm-started from the previous frame."""

  def __init__(self, robot: str, human_height: float | None = None,
               arm: ArmAbductionMap | None | str = "config") -> None:
    import mujoco

    register()
    from general_motion_retargeting import GeneralMotionRetargeting

    self.robot = robot
    self.arm = arm_map(robot) if arm == "config" else arm
    self.gmr = GeneralMotionRetargeting(
      src_human="xrobot", tgt_robot=GMR_ROBOT[robot], actual_human_height=human_height,
      verbose=False,
    )
    m = self.gmr.model
    self.joints = [
      (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_qposadr[j]))
      for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    stand = json.loads(STAND_FILE.read_text())[robot]["joint_pos"]
    self.stand_q = {adr: stand[name] for name, adr in self.joints if name in stand}

  def settle(self, body: dict[str, tuple[np.ndarray, np.ndarray]], solves: int = 30) -> RobotPose:
    """Solves one pose to convergence. GMR takes a few damped IK steps per call and
    warm-starts from the last solution; from the zero pose a single call can land on a
    wrong branch (a flipped shoulder) that later frames keep. The IK starts from the
    floating base at the pelvis target and the joints at the neutral stand: from the
    world-aligned zero pose, an operator facing away swings the whole robot round and
    winds the shoulders onto a flipped branch; the T2's zero pose has its arms out
    horizontally, from where a hanging arm can end at the roll limit, raised; and from
    its training default (elbows bent the other way) an arm held against the body ends
    25 cm off. Call it on the first frame of a stream (the calibration pose)."""
    gmr = self.gmr
    gmr.update_targets(self.arm(body) if self.arm else body)
    pos, quat = gmr.scaled_human_data[gmr.human_root_name]
    q = np.zeros_like(gmr.configuration.q)
    q[:3], q[3:7] = pos, quat
    for adr, angle in self.stand_q.items():
      q[adr] = angle
    gmr.configuration.update(q)
    for _ in range(solves - 1):
      self.retarget(body)
    return self.retarget(body)

  def retarget(self, body: dict[str, tuple[np.ndarray, np.ndarray]]) -> RobotPose:
    qpos = self.gmr.retarget(self.arm(body) if self.arm else body)
    return RobotPose(
      base_pos=qpos[:3].copy(),
      base_quat=qpos[3:7].copy(),
      joint_pos={name: float(qpos[adr]) for name, adr in self.joints},
    )
