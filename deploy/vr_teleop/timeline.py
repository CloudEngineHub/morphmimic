"""Receiver-side timeline of streamed frames: chunk merging and the jitter buffer.

The streamer sends overlapping 10-frame chunks. `Timeline` merges them by frame index,
plays them back one frame per 50 Hz control step trailing the newest frame by `DELAY`
frames (a jitter buffer that also makes the K-frame preview fully real: the cursor only
advances while it trails by more than `DELAY`, so it waits out stream stalls) and
returns the K frames the policy needs (`frames`). What is done with them (heading
alignment, blending, the stance) is the policy node's (`deploy/sim2sim/teleop.py`).
"""

from __future__ import annotations

from bisect import bisect_right

from deploy.vr_teleop.constants import CATCH_UP, DELAY, GOAL_K


class Timeline:
  """Frames keyed by the streamer's monotonic index, and the playback cursor.

  Only frames the cursor can still need are kept: the newest frame at or before the
  cursor (the hold frame if indices are missing) and everything after it.
  """

  def __init__(self, delay: int = DELAY, k: int = GOAL_K, catch_up: int = CATCH_UP):
    self.delay = delay
    self.k = k
    self.catch_up = catch_up
    self._frames: dict = {}  # index -> `protocol.RefFrame`
    self._cursor = 0
    self._newest = 0

  def clear(self) -> None:
    self._frames.clear()

  @property
  def lag(self) -> int:
    """Frames the cursor trails the newest frame (0 when empty)."""
    return self._newest - self._cursor if self._frames else 0

  def add(self, frames: list) -> bool:
    """Merge a chunk, dropping indices already held. A chunk whose newest index is
    below the cursor is a restarted stream (indices start at 0 again): the old
    timeline is dropped first.

    Returns True if the cursor jumped to catch up (the newest frame got more than
    `catch_up` frames ahead of it): the playback skipped ahead, so a caller playing
    the window should treat it like a new start.
    """
    if not frames:
      return False
    if self._frames and max(f.index for f in frames) < self._cursor:
      self.clear()
    was_empty = not self._frames
    for f in frames:
      if f.index not in self._frames and (was_empty or f.index >= self._cursor):
        self._frames[f.index] = f
    self._newest = max(self._frames)
    caught_up = not was_empty and self._newest - self._cursor > self.catch_up
    if was_empty or caught_up:
      self.restart()
    else:
      self._prune()
    return caught_up

  def restart(self) -> None:
    """Put the cursor `delay` frames behind the newest frame (not before the oldest)."""
    if not self._frames:
      return
    self._cursor = max(self._newest - self.delay, min(self._frames))
    self._prune()

  def advance(self) -> None:
    """Step the cursor one frame, never past the newest."""
    if self._frames:
      self._cursor = min(self._cursor + 1, self._newest)
      self._prune()

  def advance_buffered(self) -> None:
    """The jitter buffer's step: `advance` only while the cursor trails the newest
    frame by more than `delay`. After a stall the cursor waits until the lag has
    refilled to `delay`, so the preview stays real; in steady state the lag stays at
    `delay` and recovers by itself after uneven arrivals."""
    if self.lag > self.delay:
      self.advance()

  def frames(self) -> list | None:
    """Frames cursor ... cursor+k-1, each clamped to the newest (and a missing index
    holding the nearest earlier frame); None if empty."""
    if not self._frames:
      return None
    keys = sorted(self._frames)
    return [self._frames[keys[max(bisect_right(keys, min(i, self._newest)) - 1, 0)]]
            for i in range(self._cursor, self._cursor + self.k)]

  def _prune(self) -> None:
    """Drop frames older than the newest frame at or before the cursor."""
    keys = sorted(self._frames)
    hold = keys[max(bisect_right(keys, self._cursor) - 1, 0)]
    for i in keys:
      if i < hold:
        del self._frames[i]
