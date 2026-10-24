"""Booster T2 entity config, in the style of mjlab's asset-zoo robot modules.

Removed delays, payloads, unused gain variants and adaptation/randomization helpers.
Uses the live 10 Hz leg articulation, with pinned upstream target_names_expr API.
The XML, mesh assets and LICENSE in xmls/ are vendored unchanged.
Vendored on 2026-10-21; unchanged except FULL_COLLISION, ported from mjlab 1.4.0 to
1.6.0 (explicit contype/conaffinity and a ".*" priority catch-all; same compiled
collision properties).
"""

import math
import re
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

T2_XML = Path(__file__).parent / "xmls" / "t2.xml"
T2_GROUNDED_KNEES_BENT_ROOT_Z = 0.958416
T2_FOOT_COLLISION_EXPR = r"^(left|right)_ankle_collision$"


T2_ACTION_JOINT_NAMES = (
  "aa_head_yaw_joint",
  "head_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_elbow_pitch_joint",
  "left_elbow_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_elbow_pitch_joint",
  "right_elbow_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "waist_yaw_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

T2_LOWER_BODY_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

T2_JOINT_ARMATURE: dict[str, float] = {
  "aa_head_yaw_joint": 0.007209,
  "head_pitch_joint": 0.007209,
  ".*_shoulder_pitch_joint": 0.017,
  ".*_shoulder_roll_joint": 0.0137,
  ".*_elbow_pitch_joint": 0.0137,
  ".*_elbow_yaw_joint": 0.0137,
  ".*_wrist_pitch_joint": 0.007209,
  ".*_wrist_yaw_joint": 0.007209,
  ".*_wrist_roll_joint": 0.007209,
  "waist_pitch_joint": 0.039514279799999996,
  "waist_roll_joint": 0.0524116653,
  "waist_yaw_joint": 0.0281,
  ".*_hip_pitch_joint": 0.0281,
  ".*_hip_roll_joint": 0.0281,
  ".*_hip_yaw_joint": 0.0281,
  ".*_knee_pitch_joint": 0.0281,
  ".*_ankle_pitch_joint": 0.0358941096,
  ".*_ankle_roll_joint": 0.01333560603,
}

T2_JOINT_EFFORT_LIMIT: dict[str, float] = {
  "aa_head_yaw_joint": 22.0,
  "head_pitch_joint": 22.0,
  ".*_shoulder_pitch_joint": 74.0,
  ".*_shoulder_roll_joint": 60.0,
  ".*_elbow_pitch_joint": 60.0,
  ".*_elbow_yaw_joint": 60.0,
  ".*_wrist_pitch_joint": 22.0,
  ".*_wrist_yaw_joint": 22.0,
  ".*_wrist_roll_joint": 22.0,
  "waist_pitch_joint": 138.0,
  "waist_roll_joint": 144.0,
  "waist_yaw_joint": 135.0,
  ".*_hip_pitch_joint": 135.0,
  ".*_hip_roll_joint": 135.0,
  ".*_hip_yaw_joint": 135.0,
  ".*_knee_pitch_joint": 135.0,
  ".*_ankle_pitch_joint": 132.0,
  ".*_ankle_roll_joint": 72.0,
}

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, T2_GROUNDED_KNEES_BENT_ROOT_Z),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_pitch_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_pitch_joint": -0.6,
    "left_shoulder_roll_joint": -1.3,
    "right_shoulder_roll_joint": 1.3,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={T2_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
  priority={T2_FOOT_COLLISION_EXPR: 1, ".*": 0},
  friction={T2_FOOT_COLLISION_EXPR: (0.6,)},
)


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(T2_XML))
  spec.delete(spec.key("init_state"))
  for joint in spec.joints:
    for expr, armature in T2_JOINT_ARMATURE.items():
      if re.fullmatch(expr, joint.name):
        joint.damping[0] = 0.0
        joint.armature = armature
        joint.frictionloss = (
          0.3
          if joint.name in T2_LOWER_BODY_JOINT_NAMES or joint.name.startswith("waist_")
          else 0.0
        )
  # IMU sensors, defined here so the entity works standalone.
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  return spec


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
  actuators = []
  for expr, armature in T2_JOINT_ARMATURE.items():
    lower = any(re.fullmatch(expr, name) for name in T2_LOWER_BODY_JOINT_NAMES)
    omega = 2 * math.pi * 10
    kp = armature * omega**2 if lower else 20.0
    kd = 4 * armature * omega if lower else (4.0 if expr.startswith("waist") else 2.0)
    actuators.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=(expr,),
        stiffness=kp,
        damping=kd,
        effort_limit=T2_JOINT_EFFORT_LIMIT[expr],
        armature=armature,
        # Final actuator override (including waist friction = 0).
        frictionloss=0.3 if lower else 0.0,
      )
    )
  return tuple(actuators)


T2_BEYONDMIMIC_KPKD_10HZ_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_actuators(),
  soft_joint_pos_limit_factor=0.9,
)
T2_BEYONDMIMIC_KPKD_10HZ_ACTION_SCALE = {
  expr: 0.25 * a.effort_limit / a.stiffness
  for a in T2_BEYONDMIMIC_KPKD_10HZ_ARTICULATION.actuators
  for expr in a.target_names_expr
}


def get_t2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    articulation=T2_BEYONDMIMIC_KPKD_10HZ_ARTICULATION,
  )
