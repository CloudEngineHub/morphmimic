"""Robot, observation and world-model layout constants."""

from __future__ import annotations

from typing import Literal

# Tracked bodies, same order as mjlab's G1 tracking task.
# Index order matters everywhere.
TRACKED_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
ROOT_IDX = 0  # pelvis
ANCHOR_IDX = 7  # torso_link
ANCHOR_BODY_NAME = "torso_link"

# mjlab G1 joint order: left leg 6, right leg 6, waist 3, left arm 7, right arm 7.
_JOINT_STEMS: tuple[str, ...] = (
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow",
  "left_wrist_roll",
  "left_wrist_pitch",
  "left_wrist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow",
  "right_wrist_roll",
  "right_wrist_pitch",
  "right_wrist_yaw",
)
JOINT_NAMES: tuple[str, ...] = tuple(f"{j}_joint" for j in _JOINT_STEMS)
# BONES-SEED G1 CSV joint columns (degrees), in the same order as JOINT_NAMES.
SEED_CSV_JOINT_COLUMNS: tuple[str, ...] = tuple(f"{j}_joint_dof" for j in _JOINT_STEMS)

NUM_JOINTS = 29
NUM_BODIES = 14

# World-model state s (world frame):
# [q 29 | qd 29 | pos 14x3 | quat 14x4 | lin 14x3 | ang 14x3].
STATE_DIM = 240
# Canonical features (heading or root frame, `morphmimic.model.state`):
# [q 29 | qd 29 | pos 42 | rot6d 84 | lin 42 | ang 42].
LOCAL_DIM = 268

# Policy observation (single frame):
# [q_rel 29 | qd 29 | imu ang vel 3 | gravity 3 | a_prev 29].
PROPRIO_DIM = 93
ACTION_DIM = 29
HIST_LEN = 10

# Goal: K future reference frames of
# [ref q 29 | ref qd 29 | anchor pos 3 (reference heading frame) | anchor rot6d 6
# (relative to the robot anchor)]. Position-free: no robot position.
GOAL_K = 10
GOAL_FRAME_DIM = 67
GOAL_DIM = 670
# Version of the goal inputs (motion and keypoint), saved in checkpoints as
# "goal_version": 1 = robot-relative positions (checkpoints without the entry),
# 2 = position-free. A policy only works with the version it was trained on.
GOAL_VERSION = 2
# Frame of the motion goal's anchor position (`rewards_torch.goal_features`): "reference"
# (default), the reference's own heading frame (position-free); "robot", the current
# robot anchor frame, as goal version 1 (needs the robot's pose at deploy, from a state
# estimator). Saved in the checkpoint cfg as `goal_anchor`.
GoalAnchor = Literal["reference", "robot"]
GOAL_ANCHORS: tuple[GoalAnchor, ...] = ("reference", "robot")
# Version of the five-point goal's orientations, saved in checkpoints as
# "kp_goal_version": 1 = relative to the robot pelvis heading (yaw only; checkpoints
# without the entry), 2 = relative to the robot pelvis's full orientation (the IMU's).
# Only the five-point interface depends on it.
KP_GOAL_VERSION = 2
# Privileged critic input (critics only, never the policy): the reference
# anchor position in the robot anchor frame for each of the K goal frames, divided by
# CRITIC_ANCHOR_POS_SCALE (m, the reward's position kernel std). The reward's global
# anchor-position term depends on this offset, which the position-free goal lacks.
CRITIC_PRIV_DIM = GOAL_K * 3
CRITIC_ANCHOR_POS_SCALE = 0.3

# Five-point keypoint goal: SE(3) targets of five tracked
# bodies, in this order, per future frame: [pos 3 | rot6d 6] each.
KP_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_ankle",
  "right_ankle",
  "left_wrist",
  "right_wrist",
)
# Indices into TRACKED_BODY_NAMES.
KP_BODY_IDX: tuple[int, ...] = (0, 3, 6, 10, 13)
NUM_KP = 5
KP_POINT_DIM = 9
KP_FRAME_DIM = 45
KP_FEAT_DIM = 450  # GOAL_K * KP_FRAME_DIM
# Mask bits appended to the features: [point mask 5 | frame mask GOAL_K].
KP_MASK_DIM = 15
KP_GOAL_DIM = 465
# The "upper" subset: pelvis and both wrists.
KP_UPPER: tuple[bool, ...] = (True, False, False, True, True)
# Target-jitter half-widths (SONIC Table S4): position in m, orientation in rad.
KP_JITTER = {"pos_xy": 0.05, "pos_z": 0.01, "rot_rp": 0.1, "rot_yaw": 0.2}

# Context xi: [foot friction 1 | encoder bias 29 | torso CoM offset 3].
CTX_DIM = 33

DT = 0.02
FPS = 50
# Max per-step reward: (sum of positive reward weights = 5) * DT.
R_MAX = 0.1

# Uniform observation noise half-widths (Table A.1).
OBS_NOISE = {"joint_pos": 0.01, "joint_vel": 0.5, "ang_vel": 0.2, "gravity": 0.05}
