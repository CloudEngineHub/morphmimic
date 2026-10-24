"""Play one clip from its first frame in a `MorphMimicEnv` (`export_assets.py` uses it
to read a clip's references and its initial state)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from torch import Tensor

from morphmimic.constants import GOAL_DIM, KP_FEAT_DIM, PROPRIO_DIM, R_MAX
from morphmimic.data.manifest import load_subset
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.env_cfg import COMMAND_NAME
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.train.collector import GOAL_GROUPS
from morphmimic.train.obs_history import ObsHistory

def resolve_clip(subset_yaml: str, clip: str) -> int:
  """Index of `clip` (a clip name or an index) in the subset."""
  names = [Path(p).stem for p in load_subset(subset_yaml)]
  if clip in names:
    return names.index(clip)
  if clip.lstrip("-").isdigit() and -len(names) <= int(clip) < len(names):
    return int(clip) % len(names)
  raise SystemExit(f"clip {clip!r} is neither a name nor an index of {subset_yaml}")


class ClipPlayback:
  """mjlab viewer env (`mjlab.viewer.EnvProtocol`) in which every env of `env` plays
  clip `clip_id` from its first frame and restarts it when it ends or fails.

  `MorphMimicEnv` runs with `auto_reset=False` and the policy's obs history lives
  outside the env, so this wrapper resets finished envs, keeps the `ObsHistory` of the
  `policy` obs and the goals (`GOAL_GROUPS`), and serves both as the viewer's
  observation: `get_observations()` -> (obs history flat, goals). Each finished
  playthrough prints whether it failed and its normalised return (benchmark definition).
  """

  def __init__(self, env: MorphMimicEnv, clip_id: int, hist_len: int) -> None:
    self._env = env
    command = env.command_manager.get_term(COMMAND_NAME)
    assert isinstance(command, MultiMotionCommand)
    self.command = command
    self.clip_id = clip_id
    self.clip_name = command.library.clip_names[clip_id]
    self.clip_len = int(command.library.clip_len[clip_id])
    self.num_envs = env.num_envs
    self.history = ObsHistory(env.num_envs, hist_len, PROPRIO_DIM, env.device)
    self.goals = {
      "goal": torch.zeros(env.num_envs, GOAL_DIM, device=env.device),
      "goal_kp": torch.zeros(env.num_envs, KP_FEAT_DIM, device=env.device),
    }
    self.ep_return = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    self.playthroughs: list[dict[str, Any]] = []
    self.reset()

  @property
  def unwrapped(self) -> MorphMimicEnv:
    return self._env

  @property
  def device(self) -> str:
    return self._env.device

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self._env.cfg

  def get_observations(self) -> tuple[Tensor, dict[str, Tensor]]:
    return self.history.flat(), self.goals

  def reset(self) -> tuple[tuple[Tensor, dict[str, Tensor]], dict]:
    self._restart(torch.arange(self.num_envs, device=self.device))
    return self.get_observations(), {}

  def step(self, action: Tensor):
    frame = self.command.global_frame.clone()
    obs, reward, terminated, truncated, extras = self._env.step(action)
    self.ep_return += reward.double()
    self.history.push(obs["policy"])
    self.goals = {g: obs[g].clone() for g in GOAL_GROUPS}
    finished = (terminated | truncated).nonzero().squeeze(-1)
    if finished.numel() > 0:
      start = int(self.command.library.clip_start[self.clip_id])
      for i in finished.tolist():
        failed = bool(terminated[i])
        result = {
          "env": i,
          "failed": failed,
          "frame": int(frame[i]) - start,
          "norm_return": float(self.ep_return[i]) / (R_MAX * (self.clip_len - 1)),
        }
        self.playthroughs.append(result)
        what = f"failed at frame {result['frame']}" if failed else "completed"
        print(
          f"[morphmimic] {self.clip_name}: {what} of {self.clip_len}, normalised "
          f"return {result['norm_return']:.3f}",
          flush=True,
        )
      self._restart(finished)
    return obs, reward, terminated, truncated, extras

  def close(self) -> None:
    self._env.close()

  def _restart(self, env_ids: Tensor) -> None:
    self.command.assign_clips(env_ids, torch.full_like(env_ids, self.clip_id))
    obs, _ = self._env.reset(env_ids=env_ids)
    self.history.reset(env_ids, obs["policy"][env_ids])
    for g in GOAL_GROUPS:
      self.goals[g][env_ids] = obs[g][env_ids]
    self.ep_return[env_ids] = 0.0
