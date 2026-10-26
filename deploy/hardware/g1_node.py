"""The G1 on hardware: the robot side of the sim2sim protocol
(`deploy/sim2sim/protocol.py`).

It owns the Unitree SDK (`unitree_interface`, built from `third_party/unitree_sdk2_wrapper`
by `deploy/install_unitree_sdk.sh`): it reads the IMU (the pelvis IMU, the G1's
`Robot.imu_body`: quaternion wxyz and body-frame gyro), the 29 joint encoders and the
wireless remote, and writes 29 PD targets at 50 Hz. The policy node (`policy_node.py`,
clip or teleop) does everything else, unchanged from the sim: per step this node sends
the state message and applies the action of the same `step_id`. It has no pose source
yet (`pose_source.py`): its state messages carry a NaN pose, and it refuses an export
of the robot goal anchor (`--goal-anchor robot`).

The gains are the ones the policy was trained with, from the deployment folder's
`<robot>.json` (the robot constants' `pd`: mjlab's G1 actuators). Other G1 projects use
different gains (e.g. on the hip pitch and the wrist pitch/yaw), so never copy gains
across projects. `--kp-scale` softens the stiffness for a cautious first run.

The physical remote has the last word:

  START   from damping, interpolate over 2 s to the stand (`--stand`: the neutral stand
          of `stand.json`, default, or the training default pose) and hold it
  A       start the policy loop (the step count restarts at 0: the policy node clears
          its history; in teleop it then holds OFF until a recalibration; with a clip,
          or without one, the policy holds the neutral stand, aimed at the robot's heading)
  Y       during the policy loop: play the clip (from the stand, turned to the robot's
          heading, then back to the stand; again on the next Y)
  B       graceful stop: damp (kp 0, kd only) at the current pose; START stands again
  SELECT  emergency stop: damp and exit

Two additions: the loop waits up to `--action-wait-ms` for this step's action (else it
holds the last target), and if no new action arrives for `--action-timeout-s` (the
policy node stalled or died) it restarts the step count, so the policy node resets onto
the stand (a clip is dropped), and blends from the last target to the PD stand until the
policy answers; its first answer takes over again, on the stand (B damps). With
`--max-steps N` the policy runs N steps (50 per second), then the node blends over 2 s
back to the PD stand and holds it: A runs the policy again, B damps.

  uv run --group export python -m deploy.hardware.g1_node --deploy <deploy_dir> \\
    --net <iface> [--kp-scale 1.0] [--max-steps 1500] [--policy-host 127.0.0.1] \\
    [--record ticks.npz]

Its logic runs against any object with the SDK's methods (`G1Node(sdk, ...)`), so it
can be tested with a fake SDK, without a robot.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

from deploy.hardware.pose_source import NO_POSE_SOURCE
from deploy.sim2sim import protocol
from deploy.sim2sim.obs import Robot

STAND_FILE = Path(__file__).resolve().parents[1] / "vr_teleop/gmr_configs/stand.json"
STAND_S = 2.0  # START: interpolation time to the stand.
DAMP_TICKS = 50  # Damping commands sent on a stop (1 s).
# The remote's `keys` bitmask (the wrapper's WirelessController; its .pyi A/B bools are
# stale).
KEYS = {"R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008, "R2": 0x0010,
        "L2": 0x0020, "F1": 0x0040, "F2": 0x0080, "A": 0x0100, "B": 0x0200, "X": 0x0400,
        "Y": 0x0800, "up": 0x1000, "right": 0x2000, "down": 0x4000, "left": 0x8000}
STAND, GO, DAMP, KILL, CUE = "start", "A", "B", "select", "Y"


def pressed(ctrl, name: str) -> bool:
  return bool(int(getattr(ctrl, "keys", 0)) & KEYS[name])


class Exit(Exception):
  """SELECT: damp and leave."""


@dataclass
class Gains:
  kp: np.ndarray
  kd: np.ndarray

  @classmethod
  def from_deploy(cls, robot_doc: dict, kp_scale: float = 1.0) -> Gains:
    pd = robot_doc["pd"]
    return cls(np.asarray(pd["kp"], float) * kp_scale, np.asarray(pd["kd"], float))


@dataclass
class Recording:
  rows: dict = field(default_factory=lambda: {k: [] for k in (
    "t", "step_id", "quat", "gyro", "q", "dq", "target", "fresh")})

  def add(self, **row) -> None:
    for k, v in row.items():
      self.rows[k].append(np.asarray(v))

  def save(self, path: Path) -> int:
    np.savez_compressed(path, **{k: np.array(v) for k, v in self.rows.items()})
    return len(self.rows["t"])


class G1Node:
  """The remote's state machine and the 50 Hz policy loop over an SDK object."""

  def __init__(self, sdk, robot: Robot, gains: Gains, stand_q: np.ndarray,
               policy_host: str = protocol.HOST, bind_host: str = protocol.HOST,
               action_wait_s: float = 0.010, action_timeout_s: float = 0.25,
               max_steps: int | None = None, clock=time.perf_counter, sleep=time.sleep,
               record: Recording | None = None,
               state_port: int = protocol.STATE_PORT, action_port: int = protocol.ACTION_PORT):
    self.sdk, self.robot, self.gains = sdk, robot, gains
    self.stand_q = np.asarray(stand_q, float)
    self.n = robot.num_native
    self.dt = robot.control_dt
    self.action_wait_s, self.action_timeout_s = action_wait_s, action_timeout_s
    self.max_steps = max_steps
    self.target = self.stand_q.copy()  # The last policy target applied.
    self.clock, self.sleep = clock, sleep
    self.record = record
    self.action_in = protocol.receiver(action_port, bind_host, timeout=0.0)
    self.state_out = protocol.sender(state_port, policy_host, timeout=0.002)

  def close(self) -> None:
    self.action_in.close()
    self.state_out.close()

  # SDK access.

  def _q(self) -> np.ndarray:
    return np.asarray(self.sdk.read_low_state().motor.q[: self.n], float)

  def _send(self, q_target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    cmd = self.sdk.create_zero_command()
    cmd.q_target = [float(x) for x in q_target]
    cmd.dq_target = [0.0] * self.n
    cmd.kp = [float(x) for x in kp]
    cmd.kd = [float(x) for x in kd]
    cmd.tau_ff = [0.0] * self.n
    self.sdk.write_low_command(cmd)

  def _buttons(self, *names: str) -> str | None:
    ctrl = self.sdk.read_wireless_controller()
    return next((n for n in names if pressed(ctrl, n)), None)

  # States.

  def damp(self, ticks: int = DAMP_TICKS) -> None:
    """Zero stiffness, damping only, at the current pose."""
    for _ in range(ticks):
      self._send(self._q(), np.zeros(self.n), self.gains.kd)
      self.sleep(self.dt)

  def wait_for(self, *buttons: str, hold) -> str:
    """Hold (call `hold` every tick) until one of `buttons`, which it returns; SELECT
    exits."""
    while True:
      hit = self._buttons(*buttons, KILL)
      if hit == KILL:
        raise Exit
      if hit is not None:
        return hit
      hold()
      self.sleep(self.dt)

  def to_stand(self, start: np.ndarray) -> None:
    """PD targets from `start` to the stand over `STAND_S`."""
    steps = max(1, round(STAND_S / self.dt))
    for i in range(steps):
      if self._buttons(KILL):
        raise Exit
      alpha = (i + 1) / steps
      self._send((1 - alpha) * start + alpha * self.stand_q, self.gains.kp, self.gains.kd)
      self.sleep(self.dt)

  def hold_stand(self, *buttons: str) -> str:
    return self.wait_for(*buttons, hold=lambda: self._send(self.stand_q, self.gains.kp,
                                                           self.gains.kd))

  def stand_up(self) -> None:
    print(f"[g1_node] press {STAND} on the remote to stand ({STAND_S:.0f} s to the stand)",
          flush=True)
    self.wait_for(STAND, hold=lambda: self._send(self._q(), np.zeros(self.n), self.gains.kd))
    self.to_stand(self._q())
    print(f"[g1_node] standing; press {GO} to start the policy, {DAMP} to damp", flush=True)
    self.hold_stand(GO)

  def policy_loop(self) -> str:
    """The 50 Hz loop until B ('damp'), `max_steps` ('done') or SELECT (Exit).

    No action for `action_timeout_s` (a stalled or dead policy node): the step count and
    the cues restart at 0, which the policy node takes as a reset (it clears the clip and
    tracks the stand, facing the robot), and until it answers again the targets blend
    from the last one to the PD stand over `STAND_S` and hold it. Its first answer takes
    over again, on the stand."""
    print("[g1_node] policy running"
          + (f" for {self.max_steps} steps" if self.max_steps is not None else ""), flush=True)
    step, target, last_fresh = 0, self.stand_q.copy(), self.clock()
    misses, cues, cue_held = 0, 0, False
    stalled: list | None = None  # [the target it stalled at, ticks since]
    blend_ticks = max(1, round(STAND_S / self.dt))
    next_tick = self.clock()
    while self.max_steps is None or step < self.max_steps:
      delay = next_tick - self.clock()
      if delay > 0:
        self.sleep(delay)
      next_tick += self.dt
      hit = self._buttons(KILL, DAMP, CUE)
      if hit == KILL:
        raise Exit
      if hit == DAMP:
        return "damp"
      if hit == CUE and not cue_held and stalled is None:  # A press, not a hold.
        cues += 1
        print(f"[g1_node] {CUE}: cue {cues}", flush=True)
      cue_held = hit == CUE
      state = self.sdk.read_low_state()
      quat = np.asarray(state.imu.quat, float)  # wxyz
      gyro = np.asarray(state.imu.omega, float)  # body frame
      q = np.asarray(state.motor.q[: self.n], float)
      dq = np.asarray(state.motor.dq[: self.n], float)
      sent = step
      try:  # At most 2 ms on a missing policy node (then: a late action).
        # No pose source yet: the pose goes as NaN (the SE(3) LIO plugs in here).
        self.state_out.send(protocol.pack_state(step, quat, gyro, q, dq, cues))
        fresh = self._await_action(step)
      except zmq.Again:
        fresh = None
      if fresh is not None:
        if stalled is not None:
          print(f"[g1_node] the policy answers again: on the stand (step {step})", flush=True)
          stalled = None
        target, last_fresh = fresh, self.clock()
      elif stalled is not None:
        stalled[1] += 1
        alpha = min(1.0, stalled[1] / blend_ticks)
        target = (1 - alpha) * stalled[0] + alpha * self.stand_q
      else:
        misses += 1
        if self.clock() - last_fresh > self.action_timeout_s:
          print(f"[g1_node] no action for {self.action_timeout_s:.2f} s (policy node down?):"
                f" to the PD stand until it answers, then the policy on the stand; {DAMP}"
                " damps", flush=True)
          stalled, step, cues = [target.copy(), 0], -1, 0  # -1: the next step is 0.
      self._send(target, self.gains.kp, self.gains.kd)
      self.target = target
      if self.record is not None:
        self.record.add(t=self.clock(), step_id=sent, quat=quat, gyro=gyro, q=q, dq=dq,
                        target=target, fresh=fresh is not None)
      step += 1
      if step % 500 == 0 and misses:
        print(f"[g1_node] step {step}: {misses} late actions so far (held the last)", flush=True)
    print(f"[g1_node] {step} policy steps done ({misses} late): back to the stand", flush=True)
    return "done"

  def _await_action(self, step: int) -> np.ndarray | None:
    """The action for `step`, waiting up to `action_wait_s`; None if it is late."""
    deadline = self.clock() + self.action_wait_s
    while True:
      remaining = deadline - self.clock()
      self.action_in.setsockopt(zmq.RCVTIMEO, max(0, round(remaining * 1e3)))
      try:
        got, targets = protocol.unpack_action(self.action_in.recv(), self.n)
      except zmq.Again:
        return None
      if got == step:
        return targets
      if remaining <= 0:
        return None

  def run(self) -> None:
    """Stand, run, damp, repeat until SELECT or Ctrl-C; always ends damped. The policy
    loop's end ('done', 'timeout') or B: back to the PD stand, or damp."""
    try:
      while True:
        self.stand_up()
        while self.policy_loop() != "damp":
          self.to_stand(self.target)
          print(f"[g1_node] holding the stand; {GO} runs the policy again, {DAMP} damps",
                flush=True)
          if self.hold_stand(GO, DAMP) == DAMP:
            break
        self.damp()
        print(f"[g1_node] damped; {STAND} stands again, {KILL} or Ctrl-C exits", flush=True)
    except Exit:
      print(f"[g1_node] {KILL}: emergency stop", flush=True)
    except KeyboardInterrupt:
      pass
    finally:
      self.damp()


def stand_pose(robot: Robot, which: str) -> np.ndarray:
  """The pose START stands the robot in, in native joint order."""
  if which == "default":
    return robot.default_joint_pos.copy()
  stand = json.loads(STAND_FILE.read_text())[robot.name]["joint_pos"]
  return np.array([stand[n] for n in robot.native_joint_names])


def make_node(deploy_dir: Path, net: str, stand: str = "neutral", kp_scale: float = 1.0,
              **kwargs) -> G1Node:
  """The node on the real G1 (needs `unitree_interface`). Raises RuntimeError for an
  export of the robot goal anchor, which needs a pose source the G1 does not have yet."""
  full = json.loads((Path(deploy_dir) / "g1.json").read_text())
  if full.get("goal_anchor", "reference") == "robot":
    raise RuntimeError(NO_POSE_SOURCE)
  import unitree_interface

  doc = full["robot"]
  robot = Robot.from_dict(doc)
  gains = Gains.from_deploy(doc, kp_scale)
  print(f"[g1_node] gains from {deploy_dir}/g1.json (x{kp_scale} kp): kp "
        f"{gains.kp.min():.1f}..{gains.kp.max():.1f}, kd {gains.kd.min():.2f}..{gains.kd.max():.2f}",
        flush=True)
  sdk = unitree_interface.UnitreeInterface.create_g1(net)
  if not sdk.release_motion_control():
    print("[g1_node] WARNING: could not release high-level motion control; LowCmd may be"
          " ignored", flush=True)
  sdk.set_control_mode(unitree_interface.ControlMode.PR)
  return G1Node(sdk, robot, gains, stand_pose(robot, stand), **kwargs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--deploy", type=Path, required=True, help="export_onnx.py folder")
  parser.add_argument("--net", required=True, help="network interface to the robot (DDS)")
  parser.add_argument("--stand", choices=["neutral", "default"], default="neutral")
  parser.add_argument("--kp-scale", type=float, default=1.0)
  parser.add_argument("--policy-host", default=protocol.HOST, help="where policy_node runs")
  parser.add_argument("--bind-host", default=protocol.HOST,
                      help="address the action port listens on (0.0.0.0 for a remote policy)")
  parser.add_argument("--action-wait-ms", type=float, default=10.0)
  parser.add_argument("--action-timeout-s", type=float, default=0.25)
  parser.add_argument("--max-steps", type=int, default=None,
                      help="policy steps per A (50 per second), then hold the stand")
  parser.add_argument("--record", type=Path, default=None, help="save every policy tick")
  args = parser.parse_args()
  stopping = [False]

  def on_term(*_):
    if not stopping[0]:  # The first signal stops; repeats cannot interrupt the damping.
      stopping[0] = True
      raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, on_term)
  rec = Recording() if args.record else None
  node = make_node(args.deploy, args.net, args.stand, args.kp_scale,
                   policy_host=args.policy_host, bind_host=args.bind_host,
                   action_wait_s=args.action_wait_ms / 1e3,
                   action_timeout_s=args.action_timeout_s, max_steps=args.max_steps,
                   record=rec)
  try:
    node.run()
  finally:
    node.close()
    if rec is not None and rec.rows["t"]:
      print(f"[g1_node] saved {rec.save(args.record)} ticks to {args.record}", flush=True)


if __name__ == "__main__":
  main()
