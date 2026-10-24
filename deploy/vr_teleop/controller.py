"""The teleop mode logic on the receiving side of a PICO stream (no env, no torch).

`SimController` is what the sim2sim policy node (`deploy/sim2sim/teleop.py`) does with
the streamer's messages: follow the mode they carry, latch OFF after a fall until a
recalibration, go to IDLE when the stream goes stale, and play the frames through a
`Timeline` with its jitter buffer. It only reads the messages' `mode`, `calib_id` and
frame `index`.
"""

from __future__ import annotations

from deploy.vr_teleop.constants import BLEND_S, DELAY, FPS, TIMEOUT_S
from deploy.vr_teleop.controls import IDLE, OFF, POSE, POSE_PAUSE
from deploy.vr_teleop.timeline import Timeline


class SimController:
  """Mode, fall latch and timeout of the sim, and the timeline it plays (no env).

  `tick(now, messages)` once per control step, with the messages received since the
  last one (oldest first) and wall time `now` in seconds, returns the mode to run.
  After a tick, `entered_pose` says whether POSE (re)started on it (entered from
  another mode, or the timeline jumped to catch up while in POSE), `calib_changed`
  whether a message carried a new calib_id, and `blend_alpha`
  (0 on entry, 1 after `blend_steps` steps) how far the targets have blended from the
  previous ones to the operator's. `stale` is set while no message has arrived for
  `timeout_s` and the stream was not OFF (the mode is then IDLE), and `stale_steps`
  counts the ticks since it began.
  """

  def __init__(
    self,
    delay: int = DELAY,
    timeout_s: float = TIMEOUT_S,
    blend_steps: int = round(BLEND_S * FPS),
  ) -> None:
    self.timeline = Timeline(delay)
    self.timeout_s = timeout_s
    self.blend_steps = max(1, blend_steps)
    self.mode = OFF
    self.calib_id: int | None = None  # Of the latest message.
    self.latched = False
    self._latch_calib: int | None = None
    self._stream_mode = OFF
    self._last_msg_t: float | None = None
    self._pose_steps = 0
    self.entered_pose = False
    self.calib_changed = False
    self.blend_alpha = 1.0
    self.stale = False
    self.stale_steps = 0

  @property
  def stream_mode(self) -> str:
    """The mode of the latest message (OFF before any)."""
    return self._stream_mode

  def latch_off(self) -> None:
    """Hold OFF until the streamer's calib_id changes (after a fall)."""
    self.latched = True
    self._latch_calib = self.calib_id

  def tick(self, now: float, messages: list) -> str:
    self.calib_changed = False
    caught_up = False
    for m in messages:
      if self.calib_id is not None and m.calib_id != self.calib_id:
        self.calib_changed = True
        self.timeline.clear()  # Frames of another calibration; the next add re-anchors.
      self.calib_id = m.calib_id
      if self.latched:
        if self._latch_calib is None:
          self._latch_calib = m.calib_id  # Latched before any message.
        elif m.calib_id != self._latch_calib:
          self.latched = False
      caught_up |= self.timeline.add(m.frames)
      self._stream_mode = m.mode
      self._last_msg_t = now

    stale = False
    if self.latched or self._last_msg_t is None:
      mode = OFF
    elif now - self._last_msg_t > self.timeout_s:
      stale = self._stream_mode != OFF
      mode = IDLE if stale else OFF
    else:
      mode = self._stream_mode
    self.stale_steps = self.stale_steps + 1 if stale and self.stale else 0
    self.stale = stale

    # A catch-up jump in POSE skips the playback ahead: handled like an entry.
    self.entered_pose = mode == POSE and (self.mode != POSE or caught_up)
    if self.entered_pose:
      self.timeline.restart()
      self._pose_steps = 0
    elif mode == POSE:
      self.timeline.advance_buffered()  # Jitter buffer: waits while lag <= delay.
      self._pose_steps += 1
    elif mode != POSE_PAUSE:
      self.timeline.restart()  # Unused in OFF / IDLE: kept live, `delay` behind.
    self.blend_alpha = min(1.0, self._pose_steps / self.blend_steps)
    self.mode = mode
    return mode


def recovery_steps(stream_mode: str) -> str:
  """What the operator does to release the OFF hold after a fall, which only a new
  calib_id releases. The streamer recalibrates only from its
  own OFF, so if it is not in OFF the first A+B+X+Y is its emergency stop."""
  recalibrate = (
    "stand in the calibration pose and press A+B+X+Y (recalibrates and releases the "
    "hold)"
  )
  if stream_mode == OFF:
    return f"to resume, {recalibrate}"
  return (
    f"to resume, press A+B+X+Y twice: first anywhere (the streamer is in "
    f"{stream_mode}, so this is its emergency stop to OFF), then {recalibrate}"
  )
