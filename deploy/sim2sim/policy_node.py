"""The robot-side policy process: numpy and onnxruntime only (no torch, no mjlab).

Each `protocol` state message (joint encoders and the IMU) is answered with joint
targets. Per message it builds the policy inputs from the measurements and the
reference motion (`obs.py`), runs the deployment folder's `policy.onnx` with the
robot's inputs from `<robot>.json`, and turns the chosen interface's action into native
joint targets. A `step_id` lower than the previous one restarts the reference and
clears the history.

The reference is a recorded clip (`--source clip`: `export_assets.py`'s `<clip>.npz`, its
canonical states, played from frame 0 at each reset; with `--from-stand S`, the robot's
case on hardware, the robot stands until a cue (the remote's Y), then the clip, turned
to the robot's heading, is led into from the stand over S seconds, `CuedClipReference`)
or the operator, live (`--source teleop`: the GMR streamer's reference states, through
the teleop modes, `teleop.TeleopReference`), or the neutral stand held still
(`--source stand`, `StandReference`, the default without `--clip`: a first run on
hardware). In teleop, stand and a clip from the stand the window the policy tracks is
also pushed to
`protocol.DISPLAY_PORT`, where the sim draws it.

  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot g1 \\
    --clip <assets>/clips/g1/<clip>.npz [--interface motion|keypoint]
  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot x2 \\
    --source teleop [--streamer tcp://HOST:5556]
  uv run --group export python -m deploy.sim2sim.policy_node <deploy_dir> --robot g1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

from deploy.sim2sim import protocol
from deploy.sim2sim.obs import (
  GOAL_K,
  PROPRIO_DIM,
  History,
  Robot,
  goal_keypoint,
  goal_motion,
  proprio_frame,
  reference_window,
)
from deploy.vr_teleop.constants import FPS, PORT, STAND_BLEND_S

NUM_KP = 5
# The reference frame the goal window starts at, relative to the control step. In the
# training env the robot starts in reference frame 0, but the motion command has
# already advanced to frame 1 when the first observation is made (mjlab updates the
# command once during the reset), and it advances once per step after that: at step t
# the policy has always seen frames t + 1 .. t + K.
REFERENCE_LEAD = 1


class ClipReference:
  """A recorded clip's windows: at step t, frames t + REFERENCE_LEAD onwards."""

  def __init__(self, states: np.ndarray) -> None:
    self.states = np.asarray(states, np.float64)

  def __len__(self) -> int:
    return len(self.states)

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    return reference_window(self.states, step_id + REFERENCE_LEAD)

  def reset(self) -> None:
    pass


def placed(states: np.ndarray, yaw: float, at: np.ndarray) -> np.ndarray:
  """Reference states [T, 240] turned and moved so that frame 0's pelvis faces `yaw`
  (rad) at `at` (x, y)."""
  from deploy.sim2sim.teleop import heading, turn_states

  out = turn_states(states, yaw - heading(states[0, 100:104]), states[0, 58:60])
  pos = out[:, 58:100].reshape(len(out), -1, 3)
  pos[..., :2] += at
  out[:, 58:100] = pos.reshape(len(out), -1)
  return out


class CuedClipReference:
  """A clip on cue, from the neutral stand: the robot's case on hardware, where it
  starts in the stand facing wherever the IMU's yaw says (the remote's A runs the
  policy, Y plays the clip).

  Until a cue, and again after the clip, the window is the neutral stand (`stand.json`)
  facing the robot. A cue (`cue`: the state message's count of them went up) while
  standing turns the clip so its frame-0 pelvis heading is the robot's (as teleop's POSE
  entry) and plays `lead_in_s` from the stand into frame 0, the clip, then `return_s`
  from its last frame back into the stand. The stand is placed where the clip starts
  and ends: the goals are relative to each window's frame 0, so a jump in position
  would read as a step. A cue during the motion is ignored."""

  def __init__(self, states: np.ndarray, robot: Robot, lead_in_s: float = 2.0,
               return_s: float = STAND_BLEND_S) -> None:
    import deploy.sim2sim.teleop  # noqa: F401  Loaded now, not in the first control step.
    from deploy.vr_teleop.gmr_retarget import STAND_FILE

    self.states = np.asarray(states, np.float64)
    self.robot = robot
    self.lead_in, self.ret = round(lead_in_s * FPS), round(return_s * FPS)
    self.stand = np.asarray(json.loads(STAND_FILE.read_text())[robot.name]["state"])
    self.reset()

  def __len__(self) -> int:
    """Frames one cue plays: the lead-in, the clip and the return."""
    return self.lead_in + len(self.states) + self.ret

  def reset(self) -> None:
    self._cues, self._pending = 0, False
    self._idle: np.ndarray | None = None  # The stand window, between motions.
    self._played: np.ndarray | None = None
    self._start = 0

  def cue(self, step_id: int, cues: int) -> None:
    """`cues`: cues since the policy started (the robot side's counter)."""
    if cues <= self._cues:
      return
    self._cues = cues
    if self._played is not None or self._pending:
      print("[policy_node] cue ignored: the motion is still playing", flush=True)
    else:
      self._pending = True

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    from deploy.sim2sim.teleop import heading

    if self._idle is None:
      yaw = heading(self.robot.slot0_quat(imu_quat, q))
      self._idle = np.repeat(placed(self.stand[None], yaw, np.zeros(2)), GOAL_K, 0)
    if self._pending:
      self._pending = False
      self._played, self._start = self._motion(heading(self.robot.slot0_quat(imu_quat, q))), step_id
      print(f"[policy_node] cue: playing the clip ({len(self)} frames)", flush=True)
    if self._played is not None:
      frame = step_id - self._start + REFERENCE_LEAD
      if frame < len(self._played):
        return reference_window(self._played, frame)
      self._idle = np.repeat(self._played[-1:], GOAL_K, 0)
      self._played = None
      print("[policy_node] clip done: standing", flush=True)
    return self._idle

  def _motion(self, yaw: float) -> np.ndarray:
    from deploy.sim2sim.teleop import blend_states, heading

    start = self._idle[0]
    clip = placed(self.states, yaw, start[58:60])
    end = clip[-1]
    stand = placed(self.stand[None], heading(end[100:104]), end[58:60])[0]
    lead = [blend_states(start, clip[0], (i + 1) / self.lead_in) for i in range(self.lead_in)]
    back = [blend_states(end, stand, (i + 1) / self.ret) for i in range(self.ret)]
    return np.concatenate([np.reshape(x, (-1, clip.shape[1])) for x in (lead, clip, back)])


class StandReference:
  """The neutral stand (`stand.json`, as in teleop's IDLE) in every window frame, turned
  to the robot's heading at the first step after each reset: the goals compare the
  reference's orientations with the robot's full one, and the IMU's yaw is arbitrary."""

  def __init__(self, robot: Robot) -> None:
    import deploy.sim2sim.teleop  # noqa: F401  Loaded now, not in the first control step.
    from deploy.vr_teleop.gmr_retarget import STAND_FILE

    self.robot = robot
    self.stand = np.asarray(json.loads(STAND_FILE.read_text())[robot.name]["state"])
    self._window: np.ndarray | None = None

  def window(self, step_id: int, imu_quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    if self._window is None:
      from deploy.sim2sim.teleop import heading, turn_states

      yaw = heading(self.robot.slot0_quat(imu_quat, q))
      self._window = np.repeat(turn_states(self.stand, yaw, np.zeros(2))[None], GOAL_K, 0)
    return self._window

  def reset(self) -> None:
    self._window = None


class PolicyRunner:
  """One robot's policy: measurements and a reference window in, joint targets out.
  `reference` gives the window (`ClipReference`, `teleop.TeleopReference`)."""

  def __init__(self, deploy_dir: Path, robot: str, reference,
               interface: str = "keypoint", kp_horizon: int = GOAL_K) -> None:
    import onnxruntime as ort

    if interface not in ("motion", "keypoint"):
      raise ValueError(f"interface must be motion or keypoint, got {interface!r}")
    if not 1 <= kp_horizon <= GOAL_K:
      raise ValueError(f"kp_horizon must be in 1..{GOAL_K}")
    doc = json.loads((Path(deploy_dir) / f"{robot}.json").read_text())
    self.robot = Robot.from_dict(doc["robot"])
    self.inputs = {k: np.asarray(v, np.float32) for k, v in doc["inputs"].items()}
    self.kp_version = int(doc.get("kp_goal_version", 1))
    self.session = ort.InferenceSession(str(Path(deploy_dir) / "policy.onnx"),
                                        providers=["CPUExecutionProvider"])
    width = {i.name: i.shape[-1] for i in self.session.get_inputs()}["obs_history"]
    if width % PROPRIO_DIM:
      raise ValueError(f"obs_history width {width} is not a multiple of {PROPRIO_DIM}")
    self.history = History(width // PROPRIO_DIM)
    self.reference = reference
    self.interface = interface
    self.kp_masks = {
      "kp_point_mask": np.ones((1, NUM_KP), bool),
      "kp_frame_mask": (np.arange(GOAL_K) < kp_horizon)[None],
    }
    self.last_action = np.zeros(29)
    self.prev_step = -1
    self._warm_up()

  def _warm_up(self) -> None:
    """One inference on zeros now, so the first control step does not pay for
    onnxruntime's first run."""
    types = {"tensor(float)": np.float32, "tensor(bool)": bool}
    feeds = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], types[i.type])
             for i in self.session.get_inputs()}
    feeds.update(self.inputs)
    self.session.run(None, feeds)

  def reset(self) -> None:
    self.history.reset()
    self.last_action = np.zeros(29)

  def step(self, step_id: int, quat: np.ndarray, gyro: np.ndarray, q: np.ndarray,
           dq: np.ndarray, cues: int = 0) -> np.ndarray:
    """Native joint targets [J] for the measurements of control step `step_id` (steps
    since the last reset, when the robot stood in reference frame 0); `cues` (the
    remote's Y presses since then) go to a reference that takes them."""
    if step_id < self.prev_step:
      self.reset()
      self.reference.reset()
    self.prev_step = step_id
    if hasattr(self.reference, "cue"):
      self.reference.cue(step_id, cues)
    quat = quat / np.linalg.norm(quat)  # the IMU's
    frame = proprio_frame(self.robot, q, dq, quat, gyro, self.last_action)
    obs = self.history.push(frame)
    ref = self.last_ref = self.reference.window(step_id, quat, q)
    feeds = {
      "obs_history": obs[None].astype(np.float32),
      "goal_motion": goal_motion(ref, self.robot.anchor_quat(quat, q))[None],
      "goal_keypoint": goal_keypoint(ref, self.robot.slot0_quat(quat, q),
                                     self.kp_version)[None],
      **self.kp_masks,
      **self.inputs,
    }
    name = "action_motion" if self.interface == "motion" else "action_keypoint"
    action = self.session.run([name], feeds)[0][0].astype(np.float64)
    self.last_action = action
    return self.robot.targets(action)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("deploy_dir", type=Path,
                      help="export_onnx.py output: policy.onnx and <robot>.json")
  parser.add_argument("--robot", required=True)
  parser.add_argument("--source", choices=["clip", "teleop", "stand"], default=None,
                      help="default: clip with --clip, else stand")
  parser.add_argument("--clip", type=Path, help="reference clip .npz (--source clip)")
  parser.add_argument("--from-stand", type=float, default=None, metavar="S",
                      help="clip: the robot stands until a cue (hardware: Y), then the"
                      " clip, turned to its heading, led in over S s")
  parser.add_argument("--streamer", default=f"tcp://127.0.0.1:{PORT}",
                      help="the GMR streamer's endpoint (--source teleop)")
  parser.add_argument("--bind-host", default=protocol.HOST,
                      help="address the state port listens on (0.0.0.0 for a remote robot)")
  parser.add_argument("--robot-host", default=protocol.HOST,
                      help="where the robot side runs (sim_node, or deploy.hardware.g1_node)")
  parser.add_argument("--interface", choices=["motion", "keypoint"], default="keypoint")
  parser.add_argument("--kp-horizon", type=int, default=GOAL_K)
  parser.add_argument("--idle-exit-s", type=float, default=0.0,
                      help="exit after this long without a state message (0: never)")
  args = parser.parse_args()

  if args.source is None:
    args.source = "clip" if args.clip is not None else "stand"
  robot = Robot.from_dict(json.loads((args.deploy_dir / f"{args.robot}.json").read_text())["robot"])
  display = None
  if args.source == "clip":
    if args.clip is None:
      parser.error("--source clip needs --clip")
    states = np.load(args.clip)["states"]
    if args.from_stand is None:
      reference = ClipReference(states)
    else:
      reference = CuedClipReference(states, robot, args.from_stand)
      display = protocol.sender(protocol.DISPLAY_PORT, args.robot_host)
    what = f"{len(reference)} reference frames"
  elif args.source == "stand":
    reference = StandReference(robot)
    display = protocol.sender(protocol.DISPLAY_PORT, args.robot_host)
    what = "the neutral stand"
  else:
    from deploy.sim2sim.teleop import TeleopReference

    reference = TeleopReference(robot, args.streamer)
    display = protocol.sender(protocol.DISPLAY_PORT, args.robot_host)
    what = f"live reference from {args.streamer}"
  runner = PolicyRunner(args.deploy_dir, args.robot, reference, args.interface,
                        args.kp_horizon)
  j = runner.robot.num_native
  state_in = protocol.receiver(protocol.STATE_PORT, args.bind_host, timeout=0.5)
  action_out = protocol.sender(protocol.ACTION_PORT, args.robot_host, timeout=0.1)
  print(f"[policy_node] {args.robot}: {args.interface} interface, {what}; state <- "
        f"{protocol.STATE_PORT}, action -> {protocol.ACTION_PORT}", flush=True)
  last, n, t_log, t_cost = time.monotonic(), 0, time.monotonic(), 0.0
  try:
    while True:
      try:
        data = protocol.recv_latest(state_in)
      except zmq.Again:
        if args.idle_exit_s and time.monotonic() - last > args.idle_exit_s:
          break
        continue
      last = time.monotonic()
      step_id, quat, gyro, q, dq = protocol.unpack_state(data, j)
      targets = runner.step(step_id, quat, gyro, q, dq, protocol.state_cues(data, j))
      try:
        action_out.send(protocol.pack_action(step_id, targets))
      except zmq.Again:  # The robot side is gone; its next state message resumes us.
        pass
      if display is not None:  # Drawing only: never wait for it (no sim on hardware).
        try:
          display.send(protocol.pack_window(runner.last_ref), zmq.NOBLOCK)
        except zmq.Again:
          pass
      t_cost += time.monotonic() - last
      n += 1
      if time.monotonic() - t_log > 5.0:
        print(f"[policy_node] step {step_id}, {1e3 * t_cost / n:.2f} ms per step", flush=True)
        t_log, n, t_cost = time.monotonic(), 0, 0.0
  except KeyboardInterrupt:
    pass
  finally:
    state_in.close()
    action_out.close()
    if display is not None:
      display.close()
    if hasattr(reference, "close"):
      reference.close()


if __name__ == "__main__":
  main()
