"""ZMQ messages between the robot (or `sim_node.py`) and `policy_node.py`.

The same two messages a hardware node exchanges with the policy, one little-endian
float32 frame each:

  State  (robot -> policy, every control step), to `STATE_PORT`:
    [step_id, imu_quat_wxyz (4), imu_gyro (3), q (J), dq (J), cues]
    Joints in the robot's native order; the IMU is the orientation of the body it sits
    on (`Robot.imu_body`) and that body's angular velocity in its own frame. A
    `step_id` lower than the previous one is a reset: the policy clears its history and
    restarts the reference at frame 0. `cues` counts the operator's cues since that
    reset (the G1 remote's Y: play the clip); a count, so a dropped message loses none.
    A message without it reads as 0.

  Action (policy -> robot), to `ACTION_PORT`:
    [step_id, q_target (J)]
    Native joint position targets for the PD controllers, answering the state of the
    same `step_id`.

Each receiver binds a PULL socket and reads with `recv_latest` (it drains the queue and
keeps the newest message, as a controller should); each sender connects a PUSH socket,
which queues from `connect` on, so the first state of a run is not lost while the
policy starts (PUB/SUB would drop it). Queues are short (`QUEUE`), so a sender facing a
dead receiver fills up at once and its timed sends fail instead of piling up a backlog. (Not ZMQ's `CONFLATE`: with a short send queue it stops returning the
sender's credit, and the sender stalls after `QUEUE` messages.) The policy binds the state port, the robot the action port, like the
teleop streamer (`deploy/vr_teleop`), which also speaks ZMQ.

numpy and pyzmq only.
"""

from __future__ import annotations

import numpy as np
import zmq

HOST = "127.0.0.1"
STATE_PORT = 9881
ACTION_PORT = 9880
DISPLAY_PORT = 9882  # Policy node -> sim, teleop only: the window being tracked.
_STATE_HEAD = 1 + 4 + 3


def pack_state(step_id: int, quat: np.ndarray, gyro: np.ndarray, q: np.ndarray,
               dq: np.ndarray, cues: int = 0) -> bytes:
  return np.concatenate(([step_id], quat, gyro, q, dq, [cues])).astype("<f4").tobytes()


def _state_values(data: bytes, num_joints: int) -> np.ndarray:
  buf = np.frombuffer(data, dtype="<f4").astype(np.float64)
  n = _STATE_HEAD + 2 * num_joints
  if len(buf) not in (n, n + 1):
    raise ValueError(f"state message has {len(buf)} values, expected {n} or {n + 1} for "
                     f"{num_joints} joints")
  return buf


def unpack_state(data: bytes, num_joints: int):
  """-> (step_id, quat [4], gyro [3], q [J], dq [J])."""
  buf, j = _state_values(data, num_joints), num_joints
  return int(buf[0]), buf[1:5], buf[5:8], buf[8:8 + j], buf[8 + j:8 + 2 * j]


def state_cues(data: bytes, num_joints: int) -> int:
  """The state message's cue count (0 if it has none)."""
  buf = _state_values(data, num_joints)
  return int(buf[-1]) if len(buf) > _STATE_HEAD + 2 * num_joints else 0


def pack_action(step_id: int, targets: np.ndarray) -> bytes:
  return np.concatenate(([step_id], targets)).astype("<f4").tobytes()


def unpack_action(data: bytes, num_joints: int):
  """-> (step_id, targets [J])."""
  buf = np.frombuffer(data, dtype="<f4").astype(np.float64)
  if len(buf) != 1 + num_joints:
    raise ValueError(f"action message has {len(buf)} values, expected {1 + num_joints}")
  return int(buf[0]), buf[1:]


def pack_window(window: np.ndarray) -> bytes:
  """The reference window [K, 240] the policy tracks (for drawing only)."""
  return np.ascontiguousarray(window, "<f4").tobytes()


def unpack_window(data: bytes) -> np.ndarray:
  return np.frombuffer(data, dtype="<f4").reshape(-1, 240).astype(np.float64)


QUEUE = 16  # Messages a socket buffers (each direction).


def receiver(port: int, host: str = HOST, timeout: float | None = None) -> zmq.Socket:
  """A bound PULL socket; `recv` (and `recv_latest`) raise `zmq.Again` after `timeout`
  seconds without a message (None: wait forever)."""
  sock = zmq.Context.instance().socket(zmq.PULL)
  sock.setsockopt(zmq.RCVHWM, QUEUE)
  sock.setsockopt(zmq.LINGER, 0)
  sock.setsockopt(zmq.RCVTIMEO, -1 if timeout is None else round(timeout * 1e3))
  sock.bind(f"tcp://{host}:{port}")
  return sock


def sender(port: int, host: str = HOST, timeout: float | None = None) -> zmq.Socket:
  """A PUSH socket connected to the receiver on `port`; `send` raises `zmq.Again` after
  `timeout` seconds with the queue full (None: wait forever). ZMQ returns the queue's
  credit asynchronously, so a send can be refused for a few ms even with a live
  receiver: give a timeout rather than `zmq.NOBLOCK` where a message must not be lost."""
  sock = zmq.Context.instance().socket(zmq.PUSH)
  sock.setsockopt(zmq.LINGER, 0)
  sock.setsockopt(zmq.SNDHWM, QUEUE)
  sock.setsockopt(zmq.SNDTIMEO, -1 if timeout is None else round(timeout * 1e3))
  sock.connect(f"tcp://{host}:{port}")
  return sock


def recv_latest(sock: zmq.Socket) -> bytes:
  """The newest message queued on `sock`: waits for one (the socket's timeout), then
  drains whatever else has arrived."""
  latest = sock.recv()
  while True:
    try:
      latest = sock.recv(zmq.NOBLOCK)
    except zmq.Again:
      return latest
