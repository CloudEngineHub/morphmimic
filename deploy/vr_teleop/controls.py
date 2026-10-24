"""The PICO button state machine (OFF / IDLE / POSE / POSE_PAUSE).

The streamer polls the controller buttons at ~100 Hz and calls `Controls.update`; the
mode it holds goes into every message to the sim, and a "calibrate" event tells the
streamer to run calibration. The sim and the real deployment share these semantics.

Pressing all four of A, B, X, Y passes through the A+X subset for a few ms, so a combo
only fires once its *exact* pressed set has been held for `COMBO_HOLD_S`. Each combo
fires at most once per press and re-arms only when A, B, X, Y are all released.
A+B+X+Y (the emergency stop) can still fire after A+X fired in the same press; A+X
cannot fire once any combo has fired in it.
"""

from __future__ import annotations

from dataclasses import dataclass

from deploy.vr_teleop.constants import COMBO_HOLD_S

OFF = "OFF"
IDLE = "IDLE"
POSE = "POSE"
POSE_PAUSE = "POSE_PAUSE"

_ABXY = frozenset("abxy")
_AX = frozenset("ax")


@dataclass
class Buttons:
  """Raw controller button states (no SDK dependency, so it is easy to fake)."""

  a: bool
  b: bool
  x: bool
  y: bool
  menu_left: bool


class Controls:
  """The streamer's control mode and calibration id, driven by the buttons.

  Call `update(buttons, t)` once per poll, with wall time `t` in seconds. OFF at
  start-up; A+B+X+Y from OFF (in the calibration pose) goes to IDLE, raises `calib_id`
  and returns "calibrate"; A+X toggles IDLE <-> POSE; holding left Menu in POSE is
  POSE_PAUSE until released; A+B+X+Y from any other mode is the emergency stop (OFF,
  returns "estop"). A combo fires once its exact button set has been held
  `COMBO_HOLD_S`, at most once per press (module docstring), so a staggered A+B+X+Y
  press whose A+X subset fires first still stops.
  """

  def __init__(self) -> None:
    self.mode = OFF
    self.calib_id = 0
    self._pressed: frozenset[str] = frozenset()  # Face buttons pressed last update.
    self._since = 0.0  # Time `_pressed` last changed.
    # Combos fired in the current press; cleared when A, B, X, Y are all released.
    self._fired: set[frozenset[str]] = set()

  def update(self, buttons: Buttons, t: float) -> str | None:
    """Advance the state machine; return "calibrate", "estop" or None."""
    pressed = frozenset(k for k in "abxy" if getattr(buttons, k))
    if pressed != self._pressed:
      self._pressed = pressed
      self._since = t
    if not pressed:
      self._fired.clear()

    # A+B+X+Y fires once per press, even after A+X; A+X only while nothing has fired.
    armed = (pressed == _ABXY and _ABXY not in self._fired) or (
      pressed == _AX and not self._fired
    )
    event = None
    if armed and t - self._since >= COMBO_HOLD_S:
      self._fired.add(pressed)
      event = self._fire(pressed)

    if self.mode == POSE and buttons.menu_left:
      self.mode = POSE_PAUSE
    elif self.mode == POSE_PAUSE and not buttons.menu_left:
      self.mode = POSE
    return event

  def _fire(self, combo: frozenset[str]) -> str | None:
    if combo == _ABXY:
      if self.mode == OFF:
        self.mode = IDLE
        self.calib_id += 1
        return "calibrate"
      self.mode = OFF
      return "estop"
    # A+X toggles IDLE <-> POSE; it does nothing in OFF or POSE_PAUSE.
    if self.mode == IDLE:
      self.mode = POSE
    elif self.mode == POSE:
      self.mode = IDLE
    return None
