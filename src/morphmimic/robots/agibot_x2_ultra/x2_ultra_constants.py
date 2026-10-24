"""AgiBot X2 Ultra entity config, in the style of mjlab's asset-zoo robot modules.

Removed delay wrappers, payload/hand-payload helpers, unused gain variants and
disabled 15 Hz ankle pass. The motor-table joints are DC motors (explicit PD clipped
to the torque-speed curve); the wrist pitch/roll and head joints use built-in PD.
The XML, mesh assets and LICENSE in xmls/ are vendored unchanged.
Vendored on 2026-10-21; unchanged except FULL_COLLISION, ported from mjlab 1.4.0 to
1.6.0 (explicit contype/conaffinity and a ".*" priority catch-all; same compiled
collision properties).
"""

import math
import re
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

X2_ULTRA_XML = Path(__file__).parent / "xmls" / "x2_ultra.xml"
X2_ULTRA_GROUNDED_KNEES_BENT_ROOT_Z = 0.641
X2_ULTRA_FOOT_COLLISION_EXPR = r"^(left|right)_foot([1-9]|1[0-2])_collision$"

# (joint expressions, motor rotor inertia [kg m^2], gear ratio, peak torque [Nm],
# motor no-load speed [rpm]): PF90, PF70, PF52.
X2_ULTRA_MOTOR_TABLE = (
  (
    (
      ".*_hip_pitch_joint",
      ".*_hip_roll_joint",
      ".*_hip_yaw_joint",
      ".*_knee_joint",
      "waist_yaw_joint",
    ),
    0.606e-4,
    21.90581842,
    120.0,
    3000.0,
  ),
  (
    (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_ankle_pitch_joint"),
    0.221e-4,
    20.0,
    36.0,
    3110.0,
  ),
  (
    (
      ".*_shoulder_yaw_joint",
      ".*_elbow_joint",
      ".*_wrist_yaw_joint",
      ".*_ankle_roll_joint",
      "waist_pitch_joint",
      "waist_roll_joint",
    ),
    0.083e-4,
    19.42856,
    24.0,
    3970.0,
  ),
)
X2_ULTRA_FALLBACK_TABLE = (
  ((".*_wrist_pitch_joint", ".*_wrist_roll_joint"), 4.8),
  (("head_yaw_joint",), 2.6),
  (("head_pitch_joint",), 0.6),
)


KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, X2_ULTRA_GROUNDED_KNEES_BENT_ROOT_Z),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": -0.6,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={X2_ULTRA_FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
  priority={X2_ULTRA_FOOT_COLLISION_EXPR: 1, ".*": 0},
  friction={X2_ULTRA_FOOT_COLLISION_EXPR: (0.6,)},
)


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X2_ULTRA_XML))
  spec.delete(spec.key("init_state"))
  for joint in spec.joints:
    for expressions, inertia, gear, _, _ in X2_ULTRA_MOTOR_TABLE:
      if any(re.fullmatch(expr, joint.name) for expr in expressions):
        joint.damping[0] = 0.0
        joint.armature = inertia * gear**2
        joint.frictionloss = 0.3
  # IMU sensors on the pelvis, defined here so the entity works standalone.
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_0",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_0",
  )
  return spec


def _actuators() -> tuple[DcMotorActuatorCfg | BuiltinPositionActuatorCfg, ...]:
  actuators: list[DcMotorActuatorCfg | BuiltinPositionActuatorCfg] = []
  omega = 2 * math.pi * 10
  for expressions, inertia, gear, effort, rpm in X2_ULTRA_MOTOR_TABLE:
    armature = inertia * gear**2
    for expr in expressions:
      lower = any(part in expr for part in ("hip_", "knee_", "ankle_"))
      waist = expr.startswith("waist_")
      actuators.append(
        DcMotorActuatorCfg(
          target_names_expr=(expr,),
          stiffness=armature * omega**2 if lower else 20.0,
          damping=4 * armature * omega if lower else (4.0 if waist else 2.0),
          effort_limit=effort,
          saturation_effort=effort,
          velocity_limit=rpm * 2 * math.pi / 60 / gear,  # Joint-side no-load speed.
          armature=armature,
          frictionloss=0.3 if lower or waist else 0.0,
        )
      )
  for expressions, effort in X2_ULTRA_FALLBACK_TABLE:
    actuators.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=expressions,
        stiffness=20.0,
        damping=2.0,
        effort_limit=effort,
        armature=0.03,
        frictionloss=0.0,
      )
    )
  return tuple(actuators)


X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ARTICULATION = EntityArticulationInfoCfg(
  actuators=_actuators(),
  soft_joint_pos_limit_factor=0.9,
)
X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ACTION_SCALE = {
  expr: 0.25 * a.effort_limit / a.stiffness
  for a in X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ARTICULATION.actuators
  for expr in a.target_names_expr
}


def get_x2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    articulation=X2_ULTRA_BEYONDMIMIC_KPKD_10HZ_ARTICULATION,
  )
