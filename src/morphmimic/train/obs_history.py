"""Stacked policy-observation history.

One implementation for the simulator (the collector stacks the noisy `policy` obs) and
for imagination (rollouts resume the histories sampled from the replay), so both build
the policy input the same way: after a reset every slot holds the episode's first obs,
then each step drops the oldest obs and appends the newest.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ObsHistory:
  """Last `hist_len` single-frame observations of each env, oldest first.

  `reset` fills all slots of the given envs with their obs (the episode's first obs),
  `push` shifts in one new obs per env, `flat` is the policy input
  [num_envs, hist_len * obs_dim]. Updates are out of place, so gradients flow from
  `flat()` to the pushed observations (imagination differentiates through them).
  """

  def __init__(
    self, num_envs: int, hist_len: int, obs_dim: int, device: str | torch.device
  ):
    if hist_len < 1:
      raise ValueError(f"hist_len must be >= 1, got {hist_len}")
    self.buf = torch.zeros(num_envs, hist_len, obs_dim, device=device)

  @classmethod
  def from_batch(cls, obs_hist: Tensor) -> ObsHistory:
    """History holding `obs_hist` [B, H, D] (oldest first), e.g. sampled starts."""
    _, hist_len, obs_dim = obs_hist.shape
    history = cls(0, hist_len, obs_dim, obs_hist.device)
    history.buf = obs_hist
    return history

  @property
  def num_envs(self) -> int:
    return self.buf.shape[0]

  @property
  def hist_len(self) -> int:
    return self.buf.shape[1]

  @property
  def obs_dim(self) -> int:
    return self.buf.shape[2]

  def reset(self, env_ids: Tensor | None, obs: Tensor) -> None:
    """Fill every slot of `env_ids` (all envs if None) with `obs` [len(env_ids), D]."""
    if env_ids is None:
      self._check(obs, self.num_envs)
      self.buf = obs.unsqueeze(1).expand(-1, self.hist_len, -1).to(self.buf).clone()
      return
    self._check(obs, env_ids.numel())
    buf = self.buf.clone()
    buf[env_ids] = obs.unsqueeze(1).expand(-1, self.hist_len, -1).to(buf)
    self.buf = buf

  def push(self, obs: Tensor) -> None:
    """Append `obs` [num_envs, D] as the newest frame, dropping the oldest."""
    self._check(obs, self.num_envs)
    self.buf = self.shifted(self.buf, obs)

  @staticmethod
  def shifted(buf: Tensor, obs: Tensor) -> Tensor:
    """`push` as a function: the history `buf` [N, H, D] with `obs` [N, D] appended
    as the newest frame and the oldest dropped (a new tensor)."""
    return torch.cat([buf[:, 1:], obs.unsqueeze(1).to(buf)], dim=1)

  def flat(self) -> Tensor:
    """[num_envs, hist_len * obs_dim], oldest frame first."""
    return self.buf.reshape(self.num_envs, -1)

  def _check(self, obs: Tensor, num: int) -> None:
    if tuple(obs.shape) != (num, self.obs_dim):
      raise ValueError(f"expected obs of shape {(num, self.obs_dim)}, got {obs.shape}")
