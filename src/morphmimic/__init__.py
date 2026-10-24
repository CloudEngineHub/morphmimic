"""MorphMimic on mjlab. Importing the package registers the tracking tasks, one per
robot of the registry (`morphmimic.robots`)."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from morphmimic.env.env_cfg import morphmimic_env_cfg

_TASKS = {
  "g1": "MorphMimic-Tracking-Flat-Unitree-G1",
  "x2": "MorphMimic-Tracking-Flat-AgiBot-X2",
  "t2": "MorphMimic-Tracking-Flat-Booster-T2",
}

for _name, _task_id in _TASKS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=morphmimic_env_cfg(_name),
    play_env_cfg=morphmimic_env_cfg(_name, play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  )
