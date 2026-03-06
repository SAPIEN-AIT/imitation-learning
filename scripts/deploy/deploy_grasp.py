# scripts/deploy.py
"""
Deploy a trained BC or ACT policy in the MuJoCo sim (open-loop evaluation).

The policy reads the current qpos/qvel from the sim, predicts the next finger
control, and writes it directly to data.ctrl — no camera, no MediaPipe.

Usage
-----
    mjpython scripts/deploy.py                         # loads checkpoints/best.pt
    mjpython scripts/deploy.py --ckpt checkpoints/last.pt
    mjpython scripts/deploy.py --ckpt checkpoints/best.pt --fps 15

Controls (MuJoCo viewer)
------------------------
  R  — reset sim to open-hand start pose  (begin a new roll-out)
  Q  — quit

ACT temporal ensembling
------------------------
When the loaded policy is ACT, predicted action chunks overlap in time.
We keep a rolling queue of active chunks and compute a weighted average of
all predictions that cover the current step (weight decays with age), which
produces smoother motion than naively taking the first action each time.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer

from src.policy.dataset import QPOS_OBJ_SLICE, QPOS_FINGER_SLICE, QVEL_FINGER_SLICE, OBS_DIM, ACT_DIM
from src.policy.bc import BCPolicy
from src.policy.act import ACTPolicy


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy a trained IL policy in MuJoCo")
    p.add_argument("--ckpt", type=str,
                   default=str(_ROOT / "checkpoints" / "best.pt"),
                   help="Path to checkpoint .pt file")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Target control frequency (Hz)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Auto-reset after N steps (0 = never)")
    p.add_argument("--ensemble-k", type=int, default=0,
                   help="ACT: requery policy every K steps (0 = every step)")
    p.add_argument("--smooth", type=float, default=0.15,
                   help="EMA smoothing factor for output actions (0=off, 0.1=heavy, 1=raw). "
                        "Lower = smoother but more lag. Default 0.15.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    """Load model, norm_stats, and config from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    norm = {k: v.float() for k, v in ckpt["norm_stats"].items()}

    policy_kind = cfg["policy"]
    if policy_kind == "bc":
        bc_cfg = cfg.get("bc", {})
        model = BCPolicy(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=bc_cfg.get("hidden", [256, 256]),
            dropout=0.0,
        )
    elif policy_kind == "act":
        act_cfg = cfg.get("act", {})
        model = ACTPolicy(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            chunk_size=act_cfg.get("chunk_size", 10),
            d_model=act_cfg.get("d_model", 256),
            n_heads=act_cfg.get("n_heads", 4),
            enc_layers=act_cfg.get("enc_layers", 2),
            dec_layers=act_cfg.get("dec_layers", 4),
            latent_dim=act_cfg.get("latent_dim", 32),
        )
    else:
        raise ValueError(f"Unknown policy kind: {policy_kind}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_loss = ckpt.get("val_loss", float("nan"))
    epoch    = ckpt.get("epoch", "?")
    print(f"[Deploy] Loaded {policy_kind.upper()} policy  "
          f"(epoch {epoch}, val_loss={val_loss:.6f})")
    return model, norm, cfg, policy_kind


def get_obs(data: mujoco.MjData) -> torch.Tensor:
    """Build a (1, OBS_DIM) observation tensor from the sim state."""
    obj_pos = data.qpos[QPOS_OBJ_SLICE].astype(np.float32)     # ( 3,)  bottle xyz
    qpos_f  = data.qpos[QPOS_FINGER_SLICE].astype(np.float32)  # (16,)
    qvel_f  = data.qvel[QVEL_FINGER_SLICE].astype(np.float32)  # (16,)
    obs = np.concatenate([obj_pos, qpos_f, qvel_f])             # (35,)
    return torch.from_numpy(obs).unsqueeze(0)                   # (1, 35)


# ─────────────────────────────────────────────────────────────────────────────
# ACT temporal ensembling
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEnsembler:
    """
    Maintains a rolling queue of ACT action chunks.
    At each step, all currently-active chunks contribute to a weighted average,
    with weight = exp(-k * age/chunk_size) so newer predictions dominate.
    """

    def __init__(self, chunk_size: int, k: float = 0.01):
        self.chunk_size = chunk_size
        self.k = k
        # Each entry: (chunk_np (chunk_size, act_dim), start_step)
        self._chunks: deque = deque()
        self._step = 0

    def push(self, chunk: np.ndarray) -> None:
        """Add a new predicted chunk (chunk_size, act_dim)."""
        self._chunks.append((chunk, self._step))
        # Evict chunks that have fully elapsed
        while self._chunks and \
              self._step - self._chunks[0][1] >= self.chunk_size:
            self._chunks.popleft()

    def get_action(self) -> np.ndarray | None:
        """Return the weighted-average action for the current step."""
        if not self._chunks:
            return None
        weighted_sum = np.zeros(self._chunks[0][0].shape[1])
        weight_total = 0.0
        for chunk, start in self._chunks:
            idx = self._step - start
            if 0 <= idx < len(chunk):
                age = idx / max(self.chunk_size, 1)
                w = np.exp(-self.k * age)
                weighted_sum += w * chunk[idx]
                weight_total += w
        self._step += 1
        return weighted_sum / max(weight_total, 1e-8)

    def reset(self) -> None:
        self._chunks.clear()
        self._step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load policy ──────────────────────────────────────────────────────────
    model, norm, cfg, policy_kind = load_checkpoint(args.ckpt)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    norm  = {k: v.to(device) for k, v in norm.items()}
    print(f"[Deploy] device = {device}")

    # ACT chunk size & ensembler
    chunk_size = 1
    ensembler  = None
    if policy_kind == "act":
        chunk_size = cfg.get("act", {}).get("chunk_size", 10)
        ensembler  = TemporalEnsembler(chunk_size)
        requery_every = args.ensemble_k if args.ensemble_k > 0 else max(chunk_size // 4, 1)
        print(f"[Deploy] ACT chunk_size={chunk_size}  requery_every={requery_every}")

    # ── Load MuJoCo ──────────────────────────────────────────────────────────
    teleop_cfg  = yaml.safe_load(open(_ROOT / "configs" / "teleop_config.yaml"))
    phys_cfg    = teleop_cfg["physics"]
    ws_cfg      = teleop_cfg["workspace"]
    n_substeps  = phys_cfg["n_substeps"]

    scene_xml = str(_ROOT / phys_cfg["scene_xml"])
    mj_model  = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data   = mujoco.MjData(mj_model)

    # Find mocap body for wrist
    mid = mj_model.body("hand_proxy").mocapid[0]

    # ── Start / reset pose ───────────────────────────────────────────────────
    START_Y = ws_cfg["start_y"]
    START_Z = ws_cfg["start_z"]
    BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # identity

    def reset_sim(*, succeeded: bool | None = None) -> None:
        nonlocal n_rollouts, n_success
        if succeeded is not None:
            n_rollouts += 1
            if succeeded:
                n_success += 1
            rate = n_success / n_rollouts * 100
            status = "SUCCESS" if succeeded else "fail"
            print(f"[Deploy] Roll-out {n_rollouts}: {status}  "
                  f"(success rate {n_success}/{n_rollouts} = {rate:.0f}%)")
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.mocap_pos[mid]  = np.array([0.0, START_Y, START_Z])
        mj_data.mocap_quat[mid] = BASE_QUAT.copy()
        mj_data.ctrl[:]         = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        if ensembler is not None:
            ensembler.reset()
        print("[Deploy] Sim reset — starting new roll-out")

    reset_sim()

    # ── Success tracking ─────────────────────────────────────────────────────
    BOTTLE_START_Z  = 0.28   # z position from scene.xml
    LIFT_THRESHOLD  = 0.05   # metres above start = success
    SUCCESS_Z       = BOTTLE_START_Z + LIFT_THRESHOLD

    # Find bottle body id once
    bottle_id = mj_model.body("bottle").id

    n_rollouts = 0
    n_success  = 0

    # ── State ────────────────────────────────────────────────────────────────
    reset_flag   = False
    step_count   = 0
    requery_ctr  = 0
    dt           = 1.0 / args.fps
    _smooth_alpha = max(0.0, min(1.0, args.smooth))   # EMA alpha in [0, 1]
    _ema_action   = np.zeros(ACT_DIM, dtype=np.float32)   # running average
    print(f"[Deploy] action EMA alpha={_smooth_alpha:.2f}  "
          f"({'off' if _smooth_alpha >= 1.0 else f'smoothing ~{1/_smooth_alpha:.0f}x'})")

    # ── Key callback ────────────────────────────────────────────────────────
    def key_callback(keycode: int) -> None:
        nonlocal reset_flag
        if keycode == ord("R") or keycode == ord("r"):
            reset_flag = True

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"  Policy: {policy_kind.upper()}  |  {args.fps:.0f} Hz  |  "
          f"max_steps={args.max_steps or '∞'}")
    print("  R = reset hand to open pose")
    print("=" * 55)

    # ── Main loop ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as v:

        t_next = time.monotonic()

        while v.is_running():

            # ---- Check success ----
            bottle_z = mj_data.xpos[bottle_id, 2]
            succeeded = bottle_z >= SUCCESS_Z

            # ---- Reset ----
            if reset_flag:
                reset_flag  = False
                reset_sim(succeeded=succeeded if step_count > 0 else None)
                step_count  = 0
                requery_ctr = 0
                _ema_action[:] = 0.0

            # ---- Auto-reset after max_steps ----
            elif args.max_steps > 0 and step_count >= args.max_steps:
                reset_sim(succeeded=succeeded)
                step_count  = 0
                requery_ctr = 0

            # ---- Early success reset ----
            elif succeeded:
                reset_sim(succeeded=True)
                step_count  = 0
                requery_ctr = 0

            # ---- Build observation ----
            obs = get_obs(mj_data).to(device)   # (1, 32)

            # ---- Policy inference ----
            with torch.no_grad():
                if policy_kind == "bc":
                    # BC: normalise → forward → un-normalise
                    obs_n = (obs - norm["obs_mean"]) / norm["obs_std"]
                    act_n = model(obs_n)
                    action = (act_n * norm["act_std"] + norm["act_mean"]).squeeze(0).cpu().numpy()

                else:  # ACT
                    # Requery every requery_every steps
                    if requery_ctr % requery_every == 0:
                        obs_n = (obs - norm["obs_mean"]) / norm["obs_std"]
                        z = torch.randn(1, model.latent_dim, device=device)
                        chunk_n = model._decode(obs_n, z)           # (1, chunk, 16)
                        chunk_act = (
                            chunk_n * norm["act_std"] + norm["act_mean"]
                        ).squeeze(0).cpu().numpy()                   # (chunk, 16)
                        ensembler.push(chunk_act)
                    requery_ctr += 1

                    action = ensembler.get_action()
                    if action is None:
                        action = np.zeros(ACT_DIM, dtype=np.float32)

            # ---- EMA smoothing ----
            if _smooth_alpha < 1.0:
                _ema_action = _smooth_alpha * action + (1.0 - _smooth_alpha) * _ema_action
                action = _ema_action

            # ---- Write control ----
            mj_data.ctrl[:] = action

            # ---- Step physics ----
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)
            v.sync()

            step_count += 1

            # ---- Rate limiting ----
            t_next += dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                t_next = time.monotonic()   # catch up if behind


if __name__ == "__main__":
    main()
