"""An exported clip played N times in a row, as one clip (`export_assets.py`'s
`<clip>.npz`: canonical reference states, start state, ghost qpos).

`--trim-s T` first cuts the idle standing at both ends of the clip down to T seconds:
from T before the hands first move to T after they last move (the elbows and wrists
against the pelvis, faster than `--still-speed`). Each repeat is then placed where the
one before ended (`policy_node.placed`: its frame-0 pelvis at the last frame's pelvis
position and heading; the goals are position-free but a jump would read as a step) and
joined to it by `--blend-s` of frames from the last frame into its first
(`teleop.blend_states`). Meant for clips that start and end in nearly the same pose,
standing still; it prints how far apart those are.

  uv run python -m deploy.sim2sim.loop_clip <assets>/clips/g1/<clip>.npz --loops 2 \\
    [--trim-s 1] [--blend-s 0.3] [--out <assets>/clips/g1/<clip>_x2.npz]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from deploy.sim2sim.obs import Robot
from deploy.sim2sim.policy_node import placed
from deploy.sim2sim.teleop import blend_states, heading
from deploy.vr_teleop.constants import FPS

HAND_BODIES = ("left_elbow_link", "left_wrist_yaw_link", "right_elbow_link",
               "right_wrist_yaw_link")


def hand_speed(states: np.ndarray, robot: Robot) -> np.ndarray:
  """Per frame [T], the fastest elbow or wrist against the pelvis (m/s), smoothed over
  0.2 s."""
  pos = states[:, 58:100].reshape(len(states), -1, 3)
  idx = [robot.tracked_bodies.index(b) for b in HAND_BODIES]
  rel = pos[:, idx] - pos[:, :1]
  speed = np.linalg.norm(np.diff(rel, axis=0), axis=2).max(1) * FPS
  speed = np.append(speed, speed[-1])
  k = max(1, round(0.2 * FPS))
  return np.convolve(speed, np.ones(k) / k, mode="same")


def active_span(states: np.ndarray, robot: Robot, still_speed: float = 0.15) -> tuple[int, int]:
  """First and last frame the hands move in."""
  moving = np.flatnonzero(hand_speed(states, robot) > still_speed)
  if not len(moving):
    raise ValueError("the hands never move")
  return int(moving[0]), int(moving[-1])


def _placed_ghost(ghost: np.ndarray, yaw: float, origin: np.ndarray,
                  at: np.ndarray) -> np.ndarray:
  """Ghost qpos [T, nq] (free body first) turned by `yaw` about `origin` (x, y) and moved
  so that `origin` goes to `at`, as `placed` moves the states."""
  c, s = math.cos(yaw), math.sin(yaw)
  out = ghost.copy()
  xy = ghost[:, :2] - origin
  out[:, 0], out[:, 1] = c * xy[:, 0] - s * xy[:, 1] + at[0], s * xy[:, 0] + c * xy[:, 1] + at[1]
  w, x, y, z = ghost[:, 3:7].T
  hw, hz = math.cos(yaw / 2), math.sin(yaw / 2)  # (hw, 0, 0, hz) * q
  out[:, 3:7] = np.stack((hw * w - hz * z, hw * x - hz * y, hw * y + hz * x, hw * z + hz * w), 1)
  return out


def trim(clip: dict, robot: Robot, keep_s: float, still_speed: float = 0.15) -> dict:
  """`clip` from `keep_s` before the hands first move to `keep_s` after they last do;
  the start state is the reference's at the new first frame, at rest."""
  first, last = active_span(np.asarray(clip["states"], np.float64), robot, still_speed)
  keep = round(keep_s * FPS)
  a, b = max(0, first - keep), min(len(clip["states"]), last + 1 + keep)
  ghost = np.asarray(clip["ghost_qpos"], np.float64)
  return {**clip, "states": clip["states"][a:b], "ghost_qpos": ghost[a:b],
          "init_qpos": ghost[a].copy(), "init_qvel": np.zeros_like(clip["init_qvel"])}


def loop(clip: dict, loops: int, blend_s: float = 0.3) -> dict:
  """`clip` (the npz's arrays) repeated `loops` times; the start state is the first's."""
  states, ghost = np.asarray(clip["states"], np.float64), np.asarray(clip["ghost_qpos"], np.float64)
  n_blend = round(blend_s * FPS)
  out_s, out_g = [states], [ghost]
  for _ in range(loops - 1):
    end_s, end_g = out_s[-1][-1], out_g[-1][-1]
    yaw = heading(end_s[100:104])
    nxt = placed(states, yaw, end_s[58:60])
    turn = yaw - heading(states[0, 100:104])
    nxt_g = _placed_ghost(ghost, turn, ghost[0, :2], end_g[:2] + (nxt[0, 58:60] - end_s[58:60]))
    alpha = [(i + 1) / (n_blend + 1) for i in range(n_blend)]
    out_s.append(np.reshape([blend_states(end_s, nxt[0], a) for a in alpha], (-1, states.shape[1])))
    q = np.array([(1 - a) * end_g + a * nxt_g[0] for a in alpha]).reshape(-1, ghost.shape[1])
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    out_g.append(q)
    out_s.append(nxt)
    out_g.append(nxt_g)
  return {**clip, "states": np.concatenate(out_s).astype(np.float32),
          "ghost_qpos": np.concatenate(out_g), "clip": f"{clip['clip']}_x{loops}"}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("clip", type=Path)
  parser.add_argument("--loops", type=int, default=2)
  parser.add_argument("--trim-s", type=float, default=None,
                      help="idle standing kept before and after the hands move (s)")
  parser.add_argument("--still-speed", type=float, default=0.15,
                      help="hand speed below which the clip counts as still (m/s)")
  parser.add_argument("--blend-s", type=float, default=0.3)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()
  clip = {k: v for k, v in np.load(args.clip).items()}
  robot = Robot.load(str(clip["robot"]))
  if args.trim_s is not None:
    first, last = active_span(np.asarray(clip["states"], np.float64), robot, args.still_speed)
    clip = trim(clip, robot, args.trim_s, args.still_speed)
    print(f"[loop_clip] the hands move {first / FPS:.2f}-{last / FPS:.2f} s of "
          f"{len(np.load(args.clip)['states']) / FPS:.1f} s; "
          f"kept {len(clip['states']) / FPS:.1f} s",
          flush=True)
  s = clip["states"]
  print(f"[loop_clip] {clip['clip']}: last frame vs first: joints max "
        f"{np.abs(s[-1, :29] - s[0, :29]).max():.3f} rad, joint speed "
        f"{np.abs(s[-1, 29:58]).max():.2f} / {np.abs(s[0, 29:58]).max():.2f} rad/s", flush=True)
  out = loop(clip, args.loops, args.blend_s)
  path = args.out or args.clip.with_name(f"{args.clip.stem}_x{args.loops}.npz")
  np.savez_compressed(path, **out)
  print(f"[loop_clip] {len(out['states'])} frames ({len(out['states']) / FPS:.1f} s) -> {path}")


if __name__ == "__main__":
  main()
