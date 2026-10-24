"""Write a synthetic PICO session that GMR reads as a SEED clip (no headset needed).

  uv run --group teleop python -m deploy.tools.make_gmr_session OUT.npz \\
    --clip $DATA/seed_g1_npz50/230713/wave_R_001__A430.npz

The operator is the G1 of the clip in GMR's conventions (`gmr_operator.py`), sampled at
100 Hz, with this button script: the clip's first frame 0-4 s (A+B+X+Y at 1.0 s:
calibrate -> IDLE), A+X at 4 s (-> POSE) and the clip from then on, its last frame for
1 s, then A+X back to IDLE and 1 s more. The recording plays back
through `streamer --robot ROBOT --replay OUT.npz` (or `deploy.run --teleop --replay`)
for any robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deploy.tools.gmr_operator import g1_clip_operator
from deploy.vr_teleop.controls import Buttons
from deploy.vr_teleop.pico_source import Recorder, Sample

HZ = 100
T0_NS = 5_000_000_000_000
CALIBRATE_S = (1.0, 1.2)
PLAY_S = 4.0


def gmr_clip_session(rec: Recorder, clip: Path) -> float:
  """Record the session of a G1 dataset clip (.npz); returns its length in seconds."""
  bodies, fps = g1_clip_operator(clip)
  clip_s = len(bodies) / fps
  end_ax = PLAY_S + clip_s + 1.0
  length = end_ax + 1.0
  for k in range(int(length * HZ)):
    t = k / HZ
    f = min(len(bodies) - 1, max(0, int((t - PLAY_S) * fps)))
    abxy = CALIBRATE_S[0] <= t < CALIBRATE_S[1]
    ax = abxy or any(s <= t < s + 0.2 for s in (PLAY_S, end_ax))
    rec.add(Sample(T0_NS + k * 1_000_000_000 // HZ, bodies[f], Buttons(ax, abxy, ax, abxy, False)))
  return length


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("out", type=Path)
  parser.add_argument("--clip", type=Path, required=True, help="a G1 dataset clip (.npz)")
  args = parser.parse_args()
  rec = Recorder(args.out)
  length = gmr_clip_session(rec, args.clip.expanduser())
  rec.close()
  print(f"wrote {len(rec)} samples ({length:.1f} s) to {args.out}: POSE from {PLAY_S:.0f} s")


if __name__ == "__main__":
  main()
