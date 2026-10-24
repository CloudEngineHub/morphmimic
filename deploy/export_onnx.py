"""One ONNX policy for every robot of a shared-policy run (or for one robot's
checkpoint), with each robot's inputs in a json file and a manifest that shows the
policy is shared.

Given one checkpoint (a single-robot run, or one robot of any run), it exports that
robot's policy with the same files.

A multi-robot run writes one checkpoint per robot (`<run>/<robot>/ckpt_<it>.pt`), even
when the policy is shared (`--share all` or `--share policy`). The export first checks
that the shared networks are bit-identical across those checkpoints, then writes:

- `policy.onnx`: the shared policy, both goal interfaces. The per-robot parts are
  inputs, not constants: the robot's input-normaliser statistics and its morphology
  vector. One file serves every robot.
- `<robot>.json`: those per-robot inputs, read from the robot's checkpoint, and the
  robot's constants (`robot`: joint map, default pose, action scale, kinematic tree,
  timing; baked in `deploy/sim2sim/robots/<robot>.json`), so the folder holds all the
  robot side needs.
- `MANIFEST.txt`: the shared-network check (a fingerprint per network and robot), the
  per-robot entries, the ONNX inputs and outputs, a parity check against the PyTorch
  policy, and the SHA-256 of every file.
- `SHA256SUMS`: the same digests, for `sha256sum -c SHA256SUMS`.

A fingerprint is the first 8 hex digits of the SHA-256 of the network's weights saved
with `torch.save` (per-robot entries left out, tensors in name order). For the policy
those are exactly the bytes written by `--export-weights`, so `sha256sum` on the
written files shows the same digits without trusting this script.

  uv run --group export python -m deploy.export_onnx <run_dir> <out_dir> [--iteration N]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from morphmimic.constants import (
  ACTION_DIM,
  GOAL_DIM,
  GOAL_K,
  GOAL_VERSION,
  JOINT_NAMES,
  KP_FEAT_DIM,
  NUM_KP,
  PROPRIO_DIM,
)
from morphmimic.model.keypoints import mask_keypoint_goal
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.robots import check_imu_body
from morphmimic.train.imagination import act, act_kp
from morphmimic.train.normalizers import RunningNormalizer, normalize

FINGERPRINT_LEN = 8
OPSET = 18
PARITY_BATCH = 64
PARITY_ATOL = 1e-4
_CKPT = re.compile(r"ckpt_(\d+)\.pt$")

# Entries of a network's state dict that belong to one robot (they differ between the
# checkpoints by design), with the reason. Everything else in a shared network must be
# bit-identical.
PER_ROBOT = {
  "policy": {"decoder.morph_actor": "the robot's morphology vector (json `morph`)"},
  "critic": {"morph": "the robot's morphology vector"},
  "critic_kp": {"morph": "the robot's morphology vector"},
  "target_critic": {"morph": "the robot's morphology vector"},
  "target_critic_kp": {"morph": "the robot's morphology vector"},
  "wm": {
    "morph": "the robot's morphology vector",
    "normalizer.mean": "state normaliser, fit on the robot's motions",
    "normalizer.std": "state normaliser, fit on the robot's motions",
    "height_mean": "body-height statistics of the robot",
    "height_std": "body-height statistics of the robot",
    "residual_scale": "residual scale, estimated from the robot's warm-up",
    "joint_limits": "the robot's joint limits",
    "default_joint_pos": "the robot's default pose",
  },
}
NETWORK_LABEL = {
  "policy": "policy",
  "critic": "critic",
  "critic_kp": "critic (five-point)",
  "target_critic": "target critic",
  "target_critic_kp": "target critic (five-point)",
  "wm": "world model",
}
# Which networks a `--share` mode shares.
SHARED_BY = {
  "all": ("policy", "critic", "critic_kp", "target_critic", "target_critic_kp", "wm"),
  "policy": ("policy", "critic", "critic_kp", "target_critic", "target_critic_kp"),
  "wm": ("wm",),
}
# Per-robot inputs of the ONNX model: json key -> (checkpoint entry, statistic, width).
ROBOT_INPUTS = {
  "obs_mean": ("obs_norm", "mean", PROPRIO_DIM),
  "obs_var": ("obs_norm", "var", PROPRIO_DIM),
  "goal_mean": ("goal_norm", "mean", GOAL_DIM),
  "goal_var": ("goal_norm", "var", GOAL_DIM),
  "kp_mean": ("kp_norm", "mean", KP_FEAT_DIM),
  "kp_var": ("kp_norm", "var", KP_FEAT_DIM),
}


@dataclass
class NetworkCheck:
  """One network across the checkpoints."""

  network: str
  shared: bool
  """Shared by the run's `--share` mode, so it must be identical."""
  values: int
  """Number of compared values (per-robot entries left out)."""
  fingerprints: dict[str, str]
  differing: list[str] = field(default_factory=list)

  @property
  def identical(self) -> bool:
    return not self.differing


@dataclass
class Report:
  run: str
  share: str
  iteration: int
  robots: list[str]
  checks: list[NetworkCheck]
  files: dict[str, str]
  """File name -> SHA-256."""
  parity: dict[str, tuple[float, float]]
  """Robot -> max |ONNX - PyTorch| of the motion and keypoint actions."""

  @property
  def policy_fingerprint(self) -> str:
    return next(c for c in self.checks if c.network == "policy").fingerprints[self.robots[0]]


def resolve_paths(paths: list[Path], iteration: int | None) -> list[Path]:
  """Checkpoint files from `paths`: files are kept; a run directory gives each robot
  subdirectory's `ckpt_<iteration>.pt`, by default the latest iteration that every
  robot has."""
  files: list[Path] = []
  for path in map(Path, paths):
    if path.is_file():
      files.append(path)
      continue
    robots = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob("ckpt_*.pt")))
    if not robots:
      raise ValueError(f"{path}: no <robot>/ckpt_<it>.pt checkpoints")
    its = [{int(m.group(1)) for f in d.iterdir() if (m := _CKPT.search(f.name))}
           for d in robots]
    common = set.intersection(*its)
    it = iteration if iteration is not None else max(common, default=None)
    if it is None or it not in common:
      raise ValueError(f"{path}: no iteration {iteration} in every robot dir {robots}")
    files += [d / f"ckpt_{it}.pt" for d in robots]
  return files


def _weights(state: dict[str, Tensor], network: str) -> dict[str, Tensor]:
  """The network's tensors without its per-robot entries, in name order, contiguous."""
  own = PER_ROBOT.get(network, {})
  return {k: v.detach().contiguous() for k, v in sorted(state.items()) if k not in own}


def _saved(tensors: dict[str, Tensor]) -> bytes:
  buf = io.BytesIO()
  torch.save(tensors, buf)
  return buf.getvalue()


def _fingerprint(tensors: dict[str, Tensor]) -> str:
  return hashlib.sha256(_saved(tensors)).hexdigest()[:FINGERPRINT_LEN]


def _load(paths: list[Path]) -> dict[str, dict]:
  ckpts: dict[str, dict] = {}
  for path in paths:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    robot = ckpt["cfg"]["robot"]
    if robot in ckpts:
      raise ValueError(f"two checkpoints of robot {robot}: {path}")
    ckpts[robot] = ckpt
    ckpt["_path"] = str(path)
  first = next(iter(ckpts.values()))
  for key in ("iteration", "goal_version"):
    values = {r: c.get(key) for r, c in ckpts.items()}
    if len(set(values.values())) != 1:
      raise ValueError(f"checkpoints differ in {key}: {values}")
  for robot, ckpt in ckpts.items():
    check_imu_body(ckpt, ckpt["_path"])
  if int(first.get("goal_version", 1)) != GOAL_VERSION:
    raise ValueError(
      f"goal version {first.get('goal_version', 1)}; this code builds {GOAL_VERSION}"
    )
  share = {r: c["cfg"].get("share") for r, c in ckpts.items()}
  if len(set(share.values())) != 1:
    raise ValueError(f"checkpoints differ in share: {share}")
  return ckpts


def check_shared(ckpts: dict[str, dict]) -> list[NetworkCheck]:
  """Compares every network present in all checkpoints."""
  share = next(iter(ckpts.values()))["cfg"]["share"]
  checks = []
  for network in NETWORK_LABEL:
    if not all(network in c for c in ckpts.values()):
      continue
    weights = {r: _weights(c[network], network) for r, c in ckpts.items()}
    first = next(iter(weights.values()))
    differing = sorted(
      name for name in first
      if not all(name in w and torch.equal(w[name], first[name]) for w in weights.values())
    )
    checks.append(NetworkCheck(
      network=network,
      shared=network in SHARED_BY.get(share, ()),
      values=sum(t.numel() for t in first.values()),
      fingerprints={r: _fingerprint(w) for r, w in weights.items()},
      differing=differing,
    ))
  return checks


class DeployPolicy(nn.Module):
  """The shared policy with every per-robot part as an input; both interfaces.

  Same computation as `imagination.act` / `act_kp`: inputs normalised with the given
  statistics, the five-point goal masked (`mask_keypoint_goal`), the decoder fed the
  given morphology vector."""

  def __init__(self, policy: Policy, eps: float) -> None:
    super().__init__()
    self.policy = policy
    self.eps = eps

  def forward(
    self,
    obs_history: Tensor,
    goal_motion: Tensor,
    goal_keypoint: Tensor,
    kp_point_mask: Tensor,
    kp_frame_mask: Tensor,
    obs_mean: Tensor,
    obs_var: Tensor,
    goal_mean: Tensor,
    goal_var: Tensor,
    kp_mean: Tensor,
    kp_var: Tensor,
    morph: Tensor | None = None,
  ) -> tuple[Tensor, Tensor]:
    b = obs_history.shape[0]
    obs = normalize(obs_history.reshape(b, -1, PROPRIO_DIM), obs_mean, obs_var, self.eps)
    obs = obs.reshape(b, -1)
    goal = normalize(goal_motion, goal_mean, goal_var, self.eps)
    kp = mask_keypoint_goal(
      normalize(goal_keypoint, kp_mean, kp_var, self.eps), kp_point_mask, kp_frame_mask
    )
    return self.policy(obs, goal, morph), self.policy.forward_kp(obs, kp, morph)


def _build_policy(ckpt: dict) -> Policy:
  """The checkpoint's policy, as `load_policy` builds it (no morphology input for a run
  without one, e.g. a single-robot run)."""
  cfg, morph = ckpt["cfg"], ckpt.get("morph")
  policy = Policy(PolicyCfg(
    enc_hidden=tuple(cfg["policy_hidden"]),
    dec_hidden=tuple(cfg["policy_hidden"]),
    act=cfg["act"],
    hist_len=int(cfg["hist_len"]),
    morph_dim=0 if morph is None else int(morph["actor"].numel()),
    film=False if morph is None else bool(morph["film"]),
    keypoint=True,
  ))
  policy.load_state_dict(ckpt["policy"])
  return policy.eval()


def _robot_inputs(ckpt: dict) -> dict[str, np.ndarray]:
  out = {key: ckpt[entry][stat].numpy().astype(np.float32)
         for key, (entry, stat, _) in ROBOT_INPUTS.items()}
  morph = ckpt["policy"].get("decoder.morph_actor")
  if morph is not None and morph.numel():  # a run without the morphology input has none
    out["morph"] = morph.numpy().astype(np.float32)
  return out


def _parity_batch(hist_len: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(0)
  b = PARITY_BATCH
  return {
    "obs_history": rng.normal(size=(b, hist_len * PROPRIO_DIM)).astype(np.float32),
    "goal_motion": rng.normal(size=(b, GOAL_DIM)).astype(np.float32),
    "goal_keypoint": rng.normal(size=(b, KP_FEAT_DIM)).astype(np.float32),
    "kp_point_mask": rng.random((b, NUM_KP)) < 0.7,
    "kp_frame_mask": np.arange(GOAL_K)[None] < rng.integers(1, GOAL_K + 1, (b, 1)),
  }


def _pytorch_actions(ckpt: dict, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
  """One robot's actions through the training input path (its own checkpoint)."""
  policy = _build_policy(ckpt)
  norms = {}
  for entry, dim in (("obs_norm", PROPRIO_DIM), ("goal_norm", GOAL_DIM),
                     ("kp_norm", KP_FEAT_DIM)):
    norms[entry] = RunningNormalizer(dim).eval()
    norms[entry].load_state_dict(ckpt[entry])
  t = {k: torch.from_numpy(v) for k, v in batch.items()}
  with torch.no_grad():
    motion = act(policy, norms["obs_norm"], norms["goal_norm"], t["obs_history"],
                 t["goal_motion"])
    keypoint = act_kp(policy, norms["obs_norm"], norms["kp_norm"], t["obs_history"],
                      t["goal_keypoint"], t["kp_point_mask"], t["kp_frame_mask"])
  return motion.numpy(), keypoint.numpy()


def _export_onnx(policy: Policy, eps: float, inputs: dict[str, np.ndarray],
                 batch: dict[str, np.ndarray], path: Path) -> None:
  model = DeployPolicy(policy, eps).eval()
  names = list(batch) + list(inputs)
  args = tuple(torch.from_numpy(batch[k][:2]) for k in batch) + tuple(
    torch.from_numpy(inputs[k]) for k in inputs
  )
  batch_dim = torch.export.Dim("batch")
  dynamic = {k: ({0: batch_dim} if k in batch else None) for k in names}
  program = torch.onnx.export(
    model, args, input_names=names, output_names=["action_motion", "action_keypoint"],
    dynamic_shapes=dynamic, opset_version=OPSET, dynamo=True, external_data=False,
  )
  program.save(str(path), external_data=False)
  _strip_node_metadata(path)


def _strip_node_metadata(path: Path) -> None:
  """Drops the exporter's per-node debug metadata (source stack traces, module names),
  which records absolute paths of the machine that exported. The graph is unchanged."""
  import onnx

  model = onnx.load(str(path))
  for graph in [model.graph, *model.functions]:
    for node in graph.node:
      del node.metadata_props[:]
  onnx.save(model, str(path))


def _git_commit() -> str:
  try:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent, check=True)
    return out.stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _manifest(report: Report, ckpts: dict[str, dict], hist_len: int) -> str:
  robots = report.robots
  lines = [
    "MorphMimic deployment export",
    f"  run          {report.run} (--share {report.share}), iteration {report.iteration},"
    f" goal version {GOAL_VERSION}",
    f"  written      {datetime.now():%Y-%m-%d %H:%M} by deploy/export_onnx.py at {_git_commit()}",
    "",
  ]
  if len(robots) == 1:
    lines += [
      f"SINGLE-ROBOT POLICY: the {robots[0]} policy of this checkpoint; policy.onnx holds it.",
      "",
      "Network fingerprints (first 8 hex digits of the SHA-256 of the network's weights,",
      "per-robot entries left out; for the policy, the SHA-256 of the file written by",
      "--export-weights)",
      "",
    ]
  else:
    lines += [
      f"ONE POLICY: its weights are bit-identical in the {len(robots)} checkpoints"
      f" ({', '.join(robots)}); policy.onnx holds them once.",
      "",
      "Shared-network check (fingerprint = first 8 hex digits of the SHA-256 of the",
      "network's weights, per-robot entries left out; for the policy, the SHA-256 of the",
      "files written by --export-weights, so `sha256sum <robot>.policy.pt` shows it too)",
      "",
    ]
  single = len(robots) == 1
  header = f"  {'network':28s}" + ("" if single else f"{'shared':8s}") + f"{'values':>12s}  "
  lines.append(header + "".join(f"{r:10s}" for r in robots) + ("" if single else "result"))
  for c in report.checks:
    if single:
      result = ""
    elif c.identical:
      result = "identical"
    elif c.shared:
      result = f"DIFFERS: {', '.join(c.differing[:3])}"
    else:
      result = "separate (a network per robot)"
    shared = "" if single else f"{'yes' if c.shared else 'no':8s}"
    lines.append(
      f"  {NETWORK_LABEL[c.network]:28s}{shared}{c.values:>12,d}  "
      + "".join(f"{c.fingerprints[r]:10s}" for r in robots) + result
    )
  lines += ["", "Per-robot entries (differ by design; in <robot>.json):", ""]
  has_morph = "morph" in ckpts[robots[0]]
  if has_morph:
    width = ckpts[robots[0]]["morph"]["actor"].numel()
    lines.append(f"  {'decoder morphology vector':28s}{width:>5d}  json `morph`:"
                 " which body the decoder drives")
  for entry, what in (("obs_norm", "proprioception"), ("goal_norm", "motion goal"),
                      ("kp_norm", "five-point goal")):
    keys = [k for k, (e, _, _) in ROBOT_INPUTS.items() if e == entry]
    width = ROBOT_INPUTS[keys[0]][2]
    lines.append(f"  {entry + ' mean, var':28s}{width:>5d}  json `{keys[0]}`, `{keys[1]}`:"
                 f" {what} normaliser, fit on the robot's own data")
  lines += [
    "  (in the training checkpoints also: the world model's state normaliser, height",
    "  statistics, residual scale, joint limits and default pose; the critics' copies",
    "  of the morphology vector; the reset sampler and optimiser state)",
    "",
    "ONNX model policy.onnx (B = batch)",
    "",
    f"  obs_history     [B, {hist_len * PROPRIO_DIM}]  float  {hist_len} proprioception frames, oldest first,"
    f" {PROPRIO_DIM} each",
    f"  goal_motion     [B, {GOAL_DIM}]  float  motion goal (zeros if unused)",
    f"  goal_keypoint   [B, {KP_FEAT_DIM}]  float  five-point goal features (zeros if unused)",
    f"  kp_point_mask   [B, {NUM_KP}]    bool   visible points: pelvis, l ankle, r ankle,"
    " l wrist, r wrist",
    f"  kp_frame_mask   [B, {GOAL_K}]   bool   visible preview frames (a prefix)",
    "  obs_mean, obs_var, goal_mean, goal_var, kp_mean, kp_var"
    + (", morph" if has_morph else " (no morphology input in this run)"),
    "                  [width]  float  the robot's values, from <robot>.json",
    f"  -> action_motion   [B, {ACTION_DIM}]  through the motion goal",
    f"  -> action_keypoint [B, {ACTION_DIM}]  through the five-point goal",
    f"  Actions and joint observations use the canonical {len(JOINT_NAMES)}-joint order"
    " (the G1's;",
    "  `canonical_joint_order` in the json); other robots map it with their translator.",
    "",
    f"Parity with PyTorch (each robot's own checkpoint, {PARITY_BATCH} random inputs,"
    " random masks):",
    "",
  ]
  for r, (dm, dk) in report.parity.items():
    lines.append(f"  {r:6s} max |ONNX - PyTorch|  motion {dm:.1e}  keypoint {dk:.1e}")
  lines += ["", "Files (check with: sha256sum -c SHA256SUMS)", ""]
  lines += [f"  {digest}  {name}" for name, digest in report.files.items()]
  lines += ["", "Source checkpoints", ""]
  lines += [f"  {r:6s} {ckpts[r]['_path']}" for r in robots]
  return "\n".join(lines) + "\n"


def export(paths: list[Path], out_dir: Path, weights_dir: Path | None = None) -> Report:
  """Checks the shared networks of `paths` (one checkpoint per robot), then writes
  `policy.onnx`, `<robot>.json` and `MANIFEST.txt` to `out_dir`. One checkpoint exports
  that robot's policy. Raises ValueError, before writing anything, when several
  checkpoints do not share their policy or a shared network differs. With `weights_dir`, also writes each robot's policy weights (per-robot
  entries left out) to `<robot>.policy.pt`: byte-identical files."""
  ckpts = _load(paths)
  robots = list(ckpts)
  first = ckpts[robots[0]]
  share = first["cfg"]["share"]
  if len(robots) > 1 and "policy" not in SHARED_BY.get(share, ()):
    raise ValueError(
      f"several checkpoints need a run with a shared policy (--share all or policy);"
      f" got {len(robots)} checkpoints, share {share!r}. Export one robot's checkpoint"
      " to get its own policy."
    )
  checks = check_shared(ckpts)
  bad = [c for c in checks if c.shared and not c.identical]
  if bad:
    detail = "; ".join(f"{c.network}: {', '.join(c.differing[:5])}" for c in bad)
    raise ValueError(f"shared networks differ across the checkpoints: {detail}")

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  hist_len = int(first["cfg"]["hist_len"])
  eps = RunningNormalizer(1).eps
  batch = _parity_batch(hist_len)
  inputs = {r: _robot_inputs(c) for r, c in ckpts.items()}
  _export_onnx(_build_policy(first), eps, inputs[robots[0]], batch, out_dir / "policy.onnx")

  import onnxruntime as ort

  session = ort.InferenceSession(str(out_dir / "policy.onnx"))
  parity = {}
  for r, ckpt in ckpts.items():
    motion, keypoint = session.run(None, {**batch, **inputs[r]})
    ref_motion, ref_keypoint = _pytorch_actions(ckpt, batch)
    parity[r] = (float(np.abs(motion - ref_motion).max()),
                 float(np.abs(keypoint - ref_keypoint).max()))
  if max(max(p) for p in parity.values()) > PARITY_ATOL:
    raise ValueError(f"ONNX does not match PyTorch: {parity}")

  robots_dir = Path(__file__).parent / "sim2sim" / "robots"
  for r, ckpt in ckpts.items():
    constants = robots_dir / f"{r}.json"
    if not constants.exists():
      raise ValueError(f"no robot constants {constants}: run deploy/sim2sim/export_assets.py --bake")
    doc = {
      "about": "Per-robot inputs of policy.onnx (see MANIFEST.txt).",
      "run": first["cfg"].get("run_name"),
      "iteration": int(first["iteration"]),
      "goal_version": GOAL_VERSION,
      "kp_goal_version": int(ckpt.get("kp_goal_version", 1)),
      "source_checkpoint": ckpt["_path"],
      "canonical_joint_order": list(JOINT_NAMES),
      "inputs": {k: v.tolist() for k, v in inputs[r].items()},
      "robot": json.loads(constants.read_text()),
    }
    (out_dir / f"{r}.json").write_text(json.dumps(doc, indent=1) + "\n")
  if weights_dir is not None:
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    for r, ckpt in ckpts.items():
      (weights_dir / f"{r}.policy.pt").write_bytes(_saved(_weights(ckpt["policy"], "policy")))

  names = ["policy.onnx", *(f"{r}.json" for r in robots)]
  report = Report(
    run=str(first["cfg"].get("run_name")),
    share=share,
    iteration=int(first["iteration"]),
    robots=robots,
    checks=checks,
    files={n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest() for n in names},
    parity=parity,
  )
  (out_dir / "MANIFEST.txt").write_text(_manifest(report, ckpts, hist_len))
  (out_dir / "SHA256SUMS").write_text(
    "".join(f"{digest}  {name}\n" for name, digest in report.files.items())
  )
  return report


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("paths", nargs="+", type=Path,
                      help="a run directory (<robot>/ckpt_<it>.pt) or checkpoint files")
  parser.add_argument("out_dir", type=Path)
  parser.add_argument("--iteration", type=int, default=None,
                      help="checkpoint iteration (default: the latest every robot has)")
  parser.add_argument("--export-weights", type=Path, default=None,
                      help="also write <robot>.policy.pt weight files here")
  args = parser.parse_args()
  export(resolve_paths(args.paths, args.iteration), args.out_dir, args.export_weights)
  print((args.out_dir / "MANIFEST.txt").read_text(), end="")


if __name__ == "__main__":
  main()
