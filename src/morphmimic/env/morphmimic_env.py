"""`ManagerBasedRlEnv` whose rewards and terminations see the post-step state."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv, types

from morphmimic.env.commands import MultiMotionCommand


class MorphMimicEnv(ManagerBasedRlEnv):
  """mjlab 1.6.0's `ManagerBasedRlEnv` with two changes in `step()`: right after the
  decimation loop it runs `sim.forward()` and refreshes the motion command's relative
  body poses, before terminations and rewards; the later `sim.forward()` only runs when
  envs were auto-reset.

  mjlab computes them on kinematics one physics substep stale and on relative body
  poses built from the previous step's robot anchor. With the refresh, the reward of
  the transition (s_t, a_t) -> s_{t+1} equals `rewards_torch.tracking_reward(s_{t+1},
  ref[f_t], a_t, a_{t-1})` on the logged states, where f_t is the frame the command
  showed at time t; the command then advances to f_t + 1.
  """

  motion_command_name = "motion"

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    # Body copied from mjlab 1.6.0 ManagerBasedRlEnv.step; only the blocks marked
    # "MorphMimic" differ.
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = {}
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()

    # MorphMimic: post-step kinematics and relative body poses for terminations and
    # rewards.
    self.sim.forward()
    motion = self.command_manager.get_term(self.motion_command_name)
    assert isinstance(motion, MultiMotionCommand)
    motion.update_relative_body_poses()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Events fire before auto-reset, on the terminal state, matching the
    # explicit flow.
    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # MorphMimic: mjlab's second forward() only matters for freshly auto-reset envs;
    # the forward() above already refreshed every other env. (The cfg has no step or
    # interval events that would write state after it.)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.sim.forward()

    # Pass dt=0 for freshly reset envs so their command timers start full,
    # matching reset().
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      command_dt = self._command_dt
      command_dt.fill_(self.step_dt)
      command_dt[reset_env_ids] = 0.0
      self.command_manager.compute(dt=command_dt)
    else:
      self.command_manager.compute(dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
