"""Resample irregular PICO body samples onto a fixed 50 Hz grid of device time.

The PICO delivers `[24, 7]` body poses (xyz + quaternion **xyzw**, PICO's layout) at a
jittery device rate. The sim consumes one frame per policy step, so the streamer feeds
each sample to a `Resampler` and gets back the frames on a uniform grid anchored at the
first sample's device timestamp (positions lerped, quaternions nlerped). Grid time is
device time, so network jitter never changes the frame spacing.

The resampler works on the raw PICO layout, so its quaternions are xyzw; the `Frame`
quaternions of `protocol` (the retargeted five points) are wxyz.
"""

from __future__ import annotations

import numpy as np

from deploy.vr_teleop.constants import FPS

_NS = 1_000_000_000


def _nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
  """Normalized lerp of unit quaternions `[..., 4]` along the shorter arc.

  q and -q are the same rotation, so q1 is negated where it lies in the opposite
  hemisphere of q0; otherwise the lerp would pass near zero and spin the long way.
  """
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0.0, -q1, q1)
  q = (1.0 - alpha) * q0 + alpha * q1
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


class Resampler:
  """Turns timestamped `[24, 7]` samples into frames on a `fps` grid.

  Grid point k sits at `t_first + k / fps`. Each `push` returns the grid points in
  (previous t, t], so every grid point is emitted exactly once. A gap in the stream
  yields one interpolated frame per grid point inside it, unless it is longer than
  `max_gap_s`: then the stream paused (PICO stall, network drop), and replaying the
  pause as a glide between the stale and the new pose would invent seconds of motion.
  Instead the grid is re-anchored at the new sample, which is returned alone, exactly
  like the first push.
  """

  def __init__(self, fps: float = FPS, max_gap_s: float = 0.2) -> None:
    self.fps = fps
    self.max_gap_ns = round(max_gap_s * _NS)
    self._t0 = 0  # Device time (ns) of grid point 0, the first sample.
    self._prev_t: int | None = None
    self._prev: np.ndarray | None = None

  def push(self, t_ns: int, body: np.ndarray) -> list[np.ndarray]:
    """Add the sample at device time `t_ns`; return the new grid frames, each `[24, 7]`.

    The first sample, and any sample more than `max_gap_s` after the previous one, is
    returned as is and re-anchors the grid. A sample whose time is not after the
    previous one is dropped (and does not change the interpolation state).
    """
    body = np.asarray(body, dtype=np.float64)
    if self._prev_t is None or t_ns - self._prev_t > self.max_gap_ns:
      self._t0, self._prev_t, self._prev = t_ns, t_ns, body.copy()
      return [body.copy()]
    if t_ns <= self._prev_t:
      return []

    # Grid points k with prev_t < t0 + k / fps <= t_ns.
    k_lo = int(np.floor((self._prev_t - self._t0) * self.fps / _NS)) + 1
    k_hi = int(np.floor((t_ns - self._t0) * self.fps / _NS))
    span = t_ns - self._prev_t
    prev = self._prev
    # Keep the stored sample in the hemisphere of its predecessor, so the output
    # quaternion signs stay continuous across samples, not just within a segment.
    body = body.copy()
    dot = np.sum(prev[:, 3:] * body[:, 3:], axis=-1, keepdims=True)
    body[:, 3:] = np.where(dot < 0.0, -body[:, 3:], body[:, 3:])
    frames = []
    for k in range(k_lo, k_hi + 1):
      alpha = (self._t0 + k * _NS / self.fps - self._prev_t) / span
      frame = np.empty_like(body)
      frame[:, :3] = (1.0 - alpha) * prev[:, :3] + alpha * body[:, :3]
      frame[:, 3:] = _nlerp(prev[:, 3:], body[:, 3:], alpha)
      frames.append(frame)
    self._prev_t, self._prev = t_ns, body
    return frames
