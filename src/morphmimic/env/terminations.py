"""Termination terms of the MorphMimic env."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

from morphmimic.env.commands import MultiMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> Tensor:
  """True where the command shows its clip's last frame [B].

  Evaluated before the command advances, so the transition that ends here is still
  rewarded against that last frame. Configure it as a time-out (truncation).
  """
  cmd = cast(MultiMotionCommand, env.command_manager.get_term(command_name))
  return cmd.library.remaining(cmd.global_frame) == 0
