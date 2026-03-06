# scripts/deploy_fist.py
"""
Deploy script for the FIST-MAKING task (Phase 0).

Obs: [finger_qpos (16), finger_qvel (16)] = 32-dim  (no object in scene)
Act: finger ctrl (16-dim)

Frozen — do not modify. Use deploy.py for new tasks.

Usage
-----
    mjpython scripts/deploy_fist.py
    mjpython scripts/deploy_fist.py --ckpt checkpoints/fist/best.pt --fps 15

Controls
--------
  R  — reset sim
  Q  — quit
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

from src.policy.bc import BCPolicy
from src.policy.act import ACTPolicy

# Fixed dims for fist task
_OBS_DIM = 32
_ACT_DIM = 16
_QPOS_FINGER = slice(7, 23)
_QVEL_FINGER = slice(6, 22)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=str(_ROOT / "checkpoints" / "fist" / "best.pt"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--ensemble-k", type=int, default=0)
    p.add_argument("--smooth", type=float, default=0.15)
    return p.parse_args()


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    norm = {k: v.float() for k, v in ckpt["norm_stats"].items()}

    policy_kind = cfg["policy"]
    if policy_kind == "bc":
        bc_cfg = cfg.get("bc", {})
        model = BCPolicy(
            obs_dim=_OBS_DIM, act_dim=_ACT_DIM,
            hidden=bc_cfg.get("hidden", [256, 256]),
            dropout=0.0,
        )
    elif policy_kind == "act":
        act_cfg = cfg.get("act", {})
        model = ACTPolicy(
            obs_dim=_OBS_DIM, act_dim=_ACT_DIM,
            chunk_size=act_cfg.get("chunk_size", 10),
            d_model=act_cfg.get("d_model", 256),
            n_heads=act_cfg.get("n_heads", 4),
            enc_layers=act_cfg.get("enc_layers", 2),
            dec_layers=act_cfg.get("dec_layers", 4),
            latent_dim=act_cfg.get("latent_dim", 32),
        )
    else:
        raise ValueError(f"Unknown policy: {policy_kind}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_loss = ckpt.get("val_loss", float("nan"))
    epoch    = ckpt.get("epoch", "?")
    print(f"[Fist] Loaded {policy_kind.upper()}  (epoch {epoch}, val_loss={val_loss:.6f})")
    return model, norm, cfg, policy_kind


def get_obs(data: mujoco.MjData) -> torch.Tensor:
    qpos_f = data.qpos[_QPOS_FINGER].astype(np.float32)  # (16,)
    qvel_f = data.qvel[_QVEL_FINGER].astype(np.float32)  # (16,)
    obs = np.concatenate([qpos_f, qvel_f])                # (32,)
    return torch.from_numpy(obs).unsqueeze(0)             # (1, 32)


class TemporalEnsembler:
    def __init__(self, chunk_size: int, k: float = 0.01):
        self.chunk_size = chunk_size
        self.k = k
        self._chunks: deque = deque()
        self._step = 0

    def push(self, chunk: np.ndarray) -> None:
        self._chunks.append((chunk, self._step))
        while self._chunks and self._step - self._chunks[0][1] >= self.chunk_size:
            self._chunks.popleft()

    def get_action(self) -> np.ndarray | None:
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


def main() -> None:
    args = parse_args()
    model, norm, cfg, policy_kind = load_checkpoint(args.ckpt)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    norm  = {k: v.to(device) for k, v in norm.items()}
    print(f"[Fist] device = {device}")

    chunk_size = 1
    ensembler  = None
    if policy_kind == "act":
        chunk_size = cfg.get("act", {}).get("chunk_size", 10)
        ensembler  = TemporalEnsembler(chunk_size)
        requery_every = args.ensemble_k if args.ensemble_k > 0 else max(chunk_size // 4, 1)

    teleop_cfg = yaml.safe_load(open(_ROOT / "configs" / "teleop_config.yaml"))
    phys_cfg   = teleop_cfg["physics"]
    ws_cfg     = teleop_cfg["workspace"]
    n_substeps = phys_cfg["n_substeps"]

    scene_xml = str(_ROOT / phys_cfg["scene_xml"])
    mj_model  = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data   = mujoco.MjData(mj_model)
    mid = mj_model.body("hand_proxy").mocapid[0]

    START_Y   = ws_cfg["start_y"]
    START_Z   = ws_cfg["start_z"]
    BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

    def reset_sim() -> None:
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.mocap_pos[mid]  = np.array([0.0, START_Y, START_Z])
        mj_data.mocap_quat[mid] = BASE_QUAT.copy()
        mj_data.ctrl[:]         = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        if ensembler is not None:
            ensembler.reset()
        print("[Fist] Reset")

    reset_sim()

    reset_flag  = False
    step_count  = 0
    requery_ctr = 0
    dt          = 1.0 / args.fps
    alpha       = max(0.0, min(1.0, args.smooth))
    ema_action  = np.zeros(_ACT_DIM, dtype=np.float32)

    def key_callback(keycode: int) -> None:
        nonlocal reset_flag
        if keycode in (ord("R"), ord("r")):
            reset_flag = True

    print("=" * 45)
    print(f"  FIST policy: {policy_kind.upper()}  |  {args.fps:.0f} Hz")
    print(f"  Checkpoint:  checkpoints/fist/")
    print("  R = reset   Q = quit")
    print("=" * 45)

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as v:
        t_next = time.monotonic()

        while v.is_running():
            if reset_flag:
                reset_flag  = False
                reset_sim()
                step_count  = 0
                requery_ctr = 0
                ema_action[:] = 0.0

            if args.max_steps > 0 and step_count >= args.max_steps:
                reset_sim()
                step_count  = 0
                requery_ctr = 0

            obs = get_obs(mj_data).to(device)

            with torch.no_grad():
                if policy_kind == "bc":
                    obs_n  = (obs - norm["obs_mean"]) / norm["obs_std"]
                    act_n  = model(obs_n)
                    action = (act_n * norm["act_std"] + norm["act_mean"]).squeeze(0).cpu().numpy()
                else:
                    if requery_ctr % requery_every == 0:
                        obs_n   = (obs - norm["obs_mean"]) / norm["obs_std"]
                        z       = torch.randn(1, model.latent_dim, device=device)
                        chunk_n = model._decode(obs_n, z)
                        chunk_act = (
                            chunk_n * norm["act_std"] + norm["act_mean"]
                        ).squeeze(0).cpu().numpy()
                        ensembler.push(chunk_act)
                    requery_ctr += 1
                    action = ensembler.get_action()
                    if action is None:
                        action = np.zeros(_ACT_DIM, dtype=np.float32)

            if alpha < 1.0:
                ema_action = alpha * action + (1.0 - alpha) * ema_action
                action = ema_action

            mj_data.ctrl[:] = action
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)
            v.sync()
            step_count += 1

            t_next += dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                t_next = time.monotonic()


if __name__ == "__main__":
    main()
