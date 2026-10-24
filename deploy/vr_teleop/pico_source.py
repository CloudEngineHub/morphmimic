"""Where PICO body samples come from: the headset, or a recording of it.

`PicoSource` reads the XRoboToolkit SDK on a background thread: solved body joints, the
device timestamp and the A/B/X/Y and left-menu buttons. `Recorder` saves every sample to
an `.npz`, and `ReplaySource` plays one back
with the same interface, so the whole streamer runs (and is tested) without hardware.

Every source exposes `start()`, `latest()`, `stop()`, `fps` (the measured incoming
sample rate, an EMA over device-time deltas) and an optional `on_sample` callback
invoked once per sample as it arrives, which is how a `Recorder` is attached without
losing samples between polls. The SDK is imported only in `PicoSource.start`.
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deploy.vr_teleop.controls import Buttons

_NS = 1_000_000_000
_BUTTON_FIELDS = ("a", "b", "x", "y", "menu_left")


@dataclass
class Sample:
  t_ns: int  # Device timestamp (ns).
  body: np.ndarray  # [24, 7]: Unity xyz + quaternion xyzw (PICO's layout).
  buttons: Buttons


class _FpsEma:
  """Rate estimate: EMA (0.1) of 1 / device-time delta between samples."""

  def __init__(self) -> None:
    self.fps = 0.0
    self._last_t: int | None = None

  def update(self, t_ns: int) -> None:
    if self._last_t is not None and t_ns > self._last_t:
      inst = _NS / (t_ns - self._last_t)
      self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
    self._last_t = t_ns


class PicoSource:
  """Background reader on the XRoboToolkit SDK; `latest()` is the newest sample."""

  def __init__(self) -> None:
    self.on_sample: Callable[[Sample], None] | None = None
    self._rate = _FpsEma()
    self._latest: Sample | None = None
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._xrt = None

  @property
  def fps(self) -> float:
    return self._rate.fps

  def start(self) -> None:
    import xrobotoolkit_sdk as xrt  # Only on the machine with the headset.

    xrt.init()
    self._xrt = xrt
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)

  def latest(self) -> Sample | None:
    with self._lock:
      return self._latest

  def _run(self) -> None:
    """The reader loop. SDK errors are logged (once per run of consecutive errors) and
    retried; anything else that ends the thread is logged once, since the
    streamer would otherwise just see no new samples."""
    try:
      self._read_loop()
    except Exception as e:  # noqa: BLE001 - report it; the thread ends either way.
      print(f"[PicoSource] reader thread stopped: {e!r}; no new samples", flush=True)

  def _read_loop(self) -> None:
    xrt = self._xrt
    last_t: int | None = None
    failing = False  # In a run of consecutive SDK errors (logged at its start).
    while not self._stop.is_set():
      try:
        if not xrt.is_body_data_available():
          time.sleep(0.001)
          continue
        t_ns = int(xrt.get_time_stamp_ns())
        if t_ns == last_t:  # No new device sample yet.
          time.sleep(0.000001)
          continue
        sample = Sample(
          t_ns,
          np.array(xrt.get_body_joints_pose(), dtype=np.float64).reshape(24, 7),
          Buttons(
            bool(xrt.get_A_button()),
            bool(xrt.get_B_button()),
            bool(xrt.get_X_button()),
            bool(xrt.get_Y_button()),
            bool(xrt.get_left_menu_button()),
          ),
        )
      except Exception as e:  # noqa: BLE001 - keep the reader alive.
        if not failing:
          print(f"[PicoSource] read error (retrying): {e!r}", flush=True)
        failing = True
        time.sleep(0.001)
        continue
      failing = False
      last_t = t_ns
      self._rate.update(t_ns)
      with self._lock:
        self._latest = sample
      if self.on_sample is not None:
        self.on_sample(sample)


class Recorder:
  """Collects samples and writes them to `path` (`.npz`: t_ns, body, buttons) on close.

  `add` is thread-safe (the PICO thread calls it). Samples stay in memory until `close`
  (~0.4 GB per hour at 90 Hz), so the path is checked up front: the parent directory is
  created and must be writable, and the path must not be a directory.
  """

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    if self.path.is_dir():
      raise IsADirectoryError(f"--record path {self.path} is a directory")
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=self.path.parent):
      pass  # Raises PermissionError now rather than at close, after the session.
    self._t_ns: list[int] = []
    self._body: list[np.ndarray] = []
    self._buttons: list[list[bool]] = []
    self._lock = threading.Lock()

  def __len__(self) -> int:
    with self._lock:
      return len(self._t_ns)

  def add(self, sample: Sample) -> None:
    with self._lock:
      self._t_ns.append(sample.t_ns)
      self._body.append(np.array(sample.body, dtype=np.float64))
      self._buttons.append([bool(getattr(sample.buttons, k)) for k in _BUTTON_FIELDS])

  def close(self) -> None:
    # A file object: np.savez would append ".npz" to a path without it.
    with self._lock, open(self.path, "wb") as f:
      np.savez(
        f,
        t_ns=np.array(self._t_ns, dtype=np.int64),
        body=np.array(self._body, dtype=np.float64).reshape(-1, 24, 7),
        buttons=np.array(self._buttons, dtype=bool).reshape(-1, len(_BUTTON_FIELDS)),
      )


class ReplaySource:
  """Plays a `Recorder` file back through the `PicoSource` interface.

  With `realtime`, `latest()` returns the newest sample whose recorded time offset has
  elapsed since `start()`, like a live headset. Without it, each `latest()` call
  advances by one sample, so tests are deterministic. Either way, once the recording
  ends `latest()` keeps returning the last sample (a stalled headset).
  """

  def __init__(self, path: str | Path, realtime: bool = True) -> None:
    with np.load(path) as d:
      self._t_ns = d["t_ns"].astype(np.int64)
      self._body = d["body"]
      self._buttons = d["buttons"]
    self.realtime = realtime
    self.on_sample: Callable[[Sample], None] | None = None
    self._rate = _FpsEma()
    self._next = 0  # Index of the next sample to deliver.
    self._latest: Sample | None = None
    self._start_s = 0.0

  @property
  def fps(self) -> float:
    return self._rate.fps

  @property
  def finished(self) -> bool:
    return self._next >= len(self._t_ns)

  def start(self) -> None:
    self._start_s = time.monotonic()

  def stop(self) -> None:
    pass

  def latest(self) -> Sample | None:
    if self.realtime:
      elapsed_ns = (time.monotonic() - self._start_s) * _NS
      while not self.finished and self._t_ns[self._next] - self._t_ns[0] <= elapsed_ns:
        self._deliver()
    elif not self.finished:
      self._deliver()
    return self._latest

  def _deliver(self) -> None:
    i = self._next
    self._next += 1
    t_ns = int(self._t_ns[i])
    self._latest = Sample(
      t_ns, self._body[i].copy(), Buttons(*map(bool, self._buttons[i]))
    )
    self._rate.update(t_ns)
    if self.on_sample is not None:
      self.on_sample(self._latest)
