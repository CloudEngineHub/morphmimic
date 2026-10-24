"""Five-point keypoint goal: features, target jitter and masks.

Sparse SE(3) targets of five tracked bodies (pelvis, both ankles, both wrists) taken
from the K future reference frames: positions in the reference's own pelvis heading
frame (position-free), orientations relative to the robot pelvis's
full orientation, as its IMU measures it (five-point goal version 2). All functions take arbitrary leading batch dims except where noted. Quaternions
are wxyz.
"""

from __future__ import annotations

from typing import Literal

import torch
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  yaw_quat,
)
from torch import Tensor, nn

from morphmimic.constants import (
  GOAL_K,
  KP_BODY_IDX,
  KP_FEAT_DIM,
  KP_POINT_DIM,
  KP_UPPER,
  NUM_KP,
  ROOT_IDX,
)
from morphmimic.model.state import (
  rot6d_from_quat,
  split_state,
)

KpMaskMode = Literal["mixed", "full", "upper"]
KpPoints = Literal["full", "upper"]

# Index list (a tuple inside a subscript would be read as several dims).
_KP_IDX = list(KP_BODY_IDX)

# `mixed` masks. Points: all five, else the upper subset, else
# independent Bernoulli per point; horizon: full preview, else none, else uniform.
_P_ALL = 0.5
_P_UPPER = 0.3
_P_KEEP = 0.6  # per-point keep probability of the Bernoulli branch
_P_FULL_PREVIEW = 0.5
_P_NO_PREVIEW = 0.2


def keypoint_features(
  state: Tensor, ref_future: Tensor, jitter: Tensor | None = None
) -> Tensor:
  """Keypoint goal from K future frames `ref_future` [..., K, 240] -> [..., K * 45].

  Per frame and point (KP_BODY_IDX order): [position | rot6d orientation] of the
  reference body. Position-free: the positions are in the
  reference's own pelvis heading frame of the first goal frame (origin [pelvis_xy, 0],
  rotated by the inverse pelvis yaw, so z stays absolute), so they need no robot
  position; on hardware they come from the operator's trackers alone. The
  orientations are relative to the robot pelvis's full orientation (`KP_GOAL_VERSION`
  2; from the IMU and, on the T2, the waist encoders), so the policy sees its tilt
  relative to the targets as the motion goal shows it. Frames are concatenated in
  time order. `targets_features`
  computes the same from target poses.

  `jitter` [..., K, 5, 6] is target noise: (dx, dy, dz) is added to the position and
  (droll, dpitch, dyaw) is composed onto the orientation as a body-local rotation.
  """
  ref = split_state(ref_future)
  return targets_features(
    state, ref.body_pos[..., _KP_IDX, :], ref.body_quat[..., _KP_IDX, :], jitter
  )


def targets_features(
  state: Tensor, pos_w: Tensor, quat_w: Tensor, jitter: Tensor | None = None
) -> Tensor:
  """`keypoint_features` from world-frame target poses of the five points, `pos_w`
  [..., K, 5, 3] and `quat_w` [..., K, 5, 4] wxyz (KP_BODY_IDX order, point 0 the
  pelvis) -> [..., K * 45]. The position frame is the pelvis target's heading frame
  at frame 0; the robot `state` [..., 240] only gives the orientation frame (its pelvis)."""
  pelvis_quat = quat_w[..., 0, 0, :].contiguous()
  ref_yaw = yaw_quat(pelvis_quat)[..., None, None, :].expand(*pos_w.shape[:-1], 4)
  origin = nn.functional.pad(pos_w[..., 0, 0, :2], (0, 1))[..., None, None, :]
  pos = quat_apply_inverse(ref_yaw, pos_w - origin)
  robot_quat = nn.functional.normalize(split_state(state).body_quat[..., ROOT_IDX, :], dim=-1)
  robot_quat = robot_quat[..., None, None, :].expand(*pos_w.shape[:-1], 4)
  quat = quat_mul(quat_inv(robot_quat), quat_w)
  if jitter is not None:
    pos = pos + jitter[..., :3]
    quat = quat_mul(
      quat, quat_from_euler_xyz(jitter[..., 3], jitter[..., 4], jitter[..., 5])
    )
  return torch.cat([pos, rot6d_from_quat(quat)], dim=-1).flatten(-3)


def keypoint_jitter(
  shape: tuple[int, ...], half: dict[str, float], device: torch.device
) -> Tensor:
  """Uniform target jitter [*shape, 5, 6] in [-h, h] per dim, `half` as `KP_JITTER`:
  h = pos_xy for dims 0, 1; pos_z for 2; rot_rp for 3, 4; rot_yaw for 5."""
  h = torch.tensor(
    [
      half["pos_xy"],
      half["pos_xy"],
      half["pos_z"],
      half["rot_rp"],
      half["rot_rp"],
      half["rot_yaw"],
    ],
    device=device,
  )
  return (2.0 * torch.rand(*shape, NUM_KP, 6, device=device) - 1.0) * h


def _prefix_frame_mask(horizon: Tensor) -> Tensor:
  """Visible preview lengths [B] -> prefix frame masks [B, K]: frame k is visible iff
  k < horizon."""
  return torch.arange(GOAL_K, device=horizon.device) < horizon[:, None]


def _fixed_points(batch: int, points: KpPoints, device: torch.device) -> Tensor:
  """Deterministic point masks [B, 5] for the dedicated `full` / `upper` encoders."""
  if points == "full":
    return torch.ones(batch, NUM_KP, dtype=torch.bool, device=device)
  if points == "upper":
    return torch.tensor(KP_UPPER, device=device).expand(batch, NUM_KP).clone()
  raise ValueError(f"Expected points 'full' or 'upper', got {points!r}.")


def sample_kp_masks(
  batch: int, mode: KpMaskMode, device: torch.device
) -> tuple[Tensor, Tensor]:
  """Training masks: point mask [B, 5] and prefix frame mask [B, K], both bool.

  `full` / `upper`: that point subset and h = K. `mixed`: points are all five with
  probability 0.5, the upper subset with 0.3, else independent Bernoulli(0.6) per point
  (an empty row gets the pelvis); the visible horizon h is K with probability 0.5, 1
  with 0.2, else uniform in {1, ..., K}.
  """
  if mode in ("full", "upper"):
    points = _fixed_points(batch, mode, device)
    return points, _prefix_frame_mask(torch.full((batch,), GOAL_K, device=device))
  if mode != "mixed":
    raise ValueError(f"Expected mode 'mixed', 'full' or 'upper', got {mode!r}.")
  u = torch.rand(batch, 1, device=device)
  bernoulli = torch.rand(batch, NUM_KP, device=device) < _P_KEEP
  points = torch.where(
    u < _P_ALL,
    _fixed_points(batch, "full", device),
    torch.where(
      u < _P_ALL + _P_UPPER, _fixed_points(batch, "upper", device), bernoulli
    ),
  )
  points[:, 0] |= ~points.any(dim=-1)
  v = torch.rand(batch, device=device)
  uniform = torch.randint(1, GOAL_K + 1, (batch,), device=device)
  horizon = torch.where(
    v < _P_FULL_PREVIEW,
    torch.full_like(uniform, GOAL_K),
    torch.where(v < _P_FULL_PREVIEW + _P_NO_PREVIEW, torch.ones_like(uniform), uniform),
  )
  return points, _prefix_frame_mask(horizon)


def fixed_kp_masks(
  batch: int, points: KpPoints, horizon: int, device: torch.device
) -> tuple[Tensor, Tensor]:
  """Evaluation masks: the `points` subset and a visible horizon of `horizon` frames
  (1 <= horizon <= K), repeated over the batch."""
  if not 1 <= horizon <= GOAL_K:
    raise ValueError(f"Expected 1 <= horizon <= {GOAL_K}, got {horizon}.")
  frames = _prefix_frame_mask(torch.full((batch,), horizon, device=device))
  return _fixed_points(batch, points, device), frames


def mask_keypoint_goal(
  feats_norm: Tensor, point_mask: Tensor, frame_mask: Tensor
) -> Tensor:
  """Normalised features [B, 450] -> masked goal [B, 465].

  Every (frame k, point p) block is zeroed when `point_mask[p]` or `frame_mask[k]` is
  False; the mask bits [point_mask (5), frame_mask (K)] are appended as floats.
  """
  if feats_norm.shape[-1] != KP_FEAT_DIM:
    raise ValueError(
      f"Expected features last dim {KP_FEAT_DIM}, got {tuple(feats_norm.shape)}."
    )
  blocks = feats_norm.unflatten(-1, (GOAL_K, NUM_KP, KP_POINT_DIM))
  visible = (frame_mask[..., :, None] & point_mask[..., None, :])[..., None]
  masked = torch.where(visible, blocks, torch.zeros_like(blocks))
  return torch.cat(
    [
      masked.flatten(-3),
      point_mask.to(feats_norm.dtype),
      frame_mask.to(feats_norm.dtype),
    ],
    dim=-1,
  )
