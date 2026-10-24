"""Learning-rate schedule (paper Table A.5): cosine decay to a fraction, then constant."""

from __future__ import annotations

import math


def cosine_to(step: int, total: int, final_frac: float) -> float:
  """LR multiplier at `step`: cosine from 1 at step 0 down to `final_frac` at `total`,
  then `final_frac`."""
  if step >= total:
    return final_frac
  return final_frac + (1.0 - final_frac) * 0.5 * (
    1.0 + math.cos(math.pi * step / total)
  )
