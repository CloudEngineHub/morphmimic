"""Synthetic PICO samples GMR reads as a given G1 motion (GMR streamer tests, sessions).

The operator is the G1 itself, in GMR's conventions: G1 poses are put on GMR's G1
model (its mocap variant, which has the toe bodies), inverted through GMR's PICO-to-G1
config (`fit_gmr_configs.synth_frames`: its rotation offsets and scales; PICO's `Foot`
joint from the toe body, as the config targets it) and written as raw PICO samples
(`gmr_streamer.gmr_to_pico`). GMR on these samples gives back the G1 motion, so the G1
is the control and the X2 and T2 are retargeted from the same "human".
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.vr_teleop.fit_gmr_configs import (
  JOINTS,
  SIDES,
  _load_g1_config,
  synth_frames,
)
from deploy.vr_teleop.gmr_streamer import gmr_to_pico


@functools.cache
def _g1() -> tuple[mujoco.MjModel, dict]:
  from deploy.vr_teleop.gmr_retarget import register

  register()
  from general_motion_retargeting import params

  return mujoco.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"])), _load_g1_config()


def g1_operator(root_pos: np.ndarray, root_quat: np.ndarray,
                joint_pos: Mapping[str, np.ndarray]) -> list[np.ndarray]:
  """PICO samples `[24, 7]` (Unity frame, xyz + xyzw) of the G1 at root `root_pos`
  [T, 3] / `root_quat` [T, 4] wxyz with `joint_pos` (joint name -> [T] angles)."""
  model, cfg = _g1()
  data = mujoco.MjData(model)
  bodies = [JOINTS[j][0] for j in JOINTS]
  ids = [model.body(b).id for b in bodies]
  toes = {s: model.body(f"{s.lower()}_toe_link").id for s in SIDES}
  scale = cfg["human_scale_table"]
  out = []
  for t in range(len(root_pos)):
    data.qpos[:] = model.qpos0
    data.qpos[:3], data.qpos[3:7] = root_pos[t], root_quat[t]
    for name, angles in joint_pos.items():
      data.qpos[model.joint(name).qposadr[0]] = angles[t]
    mujoco.mj_kinematics(model, data)
    frame = synth_frames(cfg, data.xpos[ids][None], data.xquat[ids][None])[0]
    pelvis = data.xpos[model.body("pelvis").id]
    for side, toe in toes.items():
      offset = cfg["ik_match_table1"][f"{side.lower()}_toe_link"][4]
      rot = R.from_quat(data.xquat[toe], scalar_first=True) * R.from_quat(
        offset, scalar_first=True).inv()
      pos = pelvis / scale["Pelvis"] + (data.xpos[toe] - pelvis) / scale[f"{side}_Foot"]
      frame[f"{side}_Foot"] = (pos, rot.as_quat(scalar_first=True))
    out.append(gmr_to_pico(frame))
  return out


def g1_clip_operator(clip_path) -> tuple[list[np.ndarray], float]:
  """`g1_operator` of every frame of a G1 dataset clip (.npz), and its fps."""
  d = np.load(clip_path)
  names = [str(n) for n in d["body_names"]]
  root = names.index("pelvis")
  joints = {str(n): d["joint_pos"][:, k] for k, n in enumerate(d["joint_names"])}
  return (g1_operator(d["body_pos_w"][:, root], d["body_quat_w"][:, root], joints),
          float(d["fps"]))
