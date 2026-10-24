"""VR teleop timing, network and safety constants (spec 2026-10-22-vr-teleop-sim).

Shared by the PICO streamer and the sim2sim policy node so both ends agree on the
stream layout.
"""

from __future__ import annotations

FPS = 50  # Stream and policy rate (Hz).
GOAL_K = 10  # Policy preview frames (`morphmimic.constants.GOAL_K`; a test checks they agree).
CHUNK = 10  # Frames per streamer message.
# Preview delay in frames. Frame 0 of the preview is the current frame f, since
# `MotionLibrary.future(f, K)` returns f ... f+K-1, so D = K - 1 gives a fully real
# preview.
DELAY = 9
CATCH_UP = 200  # Frames the timeline may fall behind before it resets (catch-up).
PORT = 5556  # ZMQ PUB/SUB port.
# Seconds without data before the stream counts as stale: the policy node goes to IDLE
# and the reference blends over STAND_BLEND_S to the neutral stand (`stand.json`).
TIMEOUT_S = 3.0
STAND_BLEND_S = 1.0
BLEND_S = 0.5  # Entering POSE blends the tracked window -> the operator's over this (s).
FALL_HEIGHT = 0.3  # Pelvis below this height (m) counts as a fall.
FALL_TILT_DEG = 60  # Pelvis tilt over this (degrees) counts as a fall.
COMBO_HOLD_S = 0.1  # Seconds an exact button combo must be held before it fires.
