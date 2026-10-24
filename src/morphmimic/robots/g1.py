"""Unitree G1 spec: the canonical robot, identity translator."""

from __future__ import annotations

from functools import cache

from morphmimic.robots.spec import RobotSpec, Translator, leg_length


@cache
def g1_leg_length() -> float:
  """G1 left thigh + shank at qpos0 (0.341 + 0.300 m): the height_scale reference."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

  return leg_length(get_g1_robot_cfg, "left_knee_link")


def g1_spec() -> RobotSpec:
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
  )

  from morphmimic.constants import (
    JOINT_NAMES,
    SEED_CSV_JOINT_COLUMNS,
    TRACKED_BODY_NAMES,
  )

  return RobotSpec(
    name="g1",
    entity_cfg_fn=get_g1_robot_cfg,
    action_scale=dict(G1_ACTION_SCALE),
    joint_names=JOINT_NAMES,
    root_body="pelvis",
    anchor_body="torso_link",
    imu_body="pelvis",
    tracked_bodies=TRACKED_BODY_NAMES,
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
    translator=Translator.identity(len(JOINT_NAMES)),
    csv_joint_columns=SEED_CSV_JOINT_COLUMNS,
    height_scale=1.0,
    contact_budget=(35, 250),
  )
