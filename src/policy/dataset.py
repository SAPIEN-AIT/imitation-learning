# src/policy/dataset.py
"""
Dataset utilities for loading LEAP Hand HDF5 demos into PyTorch.

Each demo file has::

    observations/qpos   (T, 30)   – full MuJoCo qpos (free-joint + 16 fingers)
    observations/qvel   (T, 28)   – full MuJoCo qvel (free-joint + 16 fingers)
    actions/ctrl        (T, 16)   – finger control targets

We slice the finger-only portion:
    qpos_fingers = qpos[:, 7:23]   (16-dim)
    qvel_fingers = qvel[:, 6:22]   (16-dim)

An observation at time *t* is ``[qpos_fingers, qvel_fingers]`` (32-dim).
The action is ``ctrl`` at time *t* (16-dim).

For **action-chunking** (ACT / Diffusion Policy), the dataset can return a
window of *chunk_size* future actions instead of a single action.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


# ── Index slices ────────────────────────────────────────────────────────────
QPOS_OBJ_SLICE    = slice(0, 3)    # bottle xyz position in qpos (freejoint)
QPOS_FINGER_SLICE = slice(7, 23)   # 16 finger joints in qpos
QVEL_FINGER_SLICE = slice(6, 22)   # 16 finger joints in qvel
OBS_DIM = 35                       # 3 obj_pos + 16 finger qpos + 16 finger qvel
ACT_DIM = 16                       # ctrl


# ── Single-demo dataset ────────────────────────────────────────────────────

class _SingleDemo(Dataset):
    """One HDF5 demo → (obs, action_chunk) pairs."""

    def __init__(self, path: Path, chunk_size: int = 1) -> None:
        super().__init__()
        self.chunk_size = chunk_size

        with h5py.File(str(path), "r") as f:
            qpos = f["observations/qpos"][:]        # (T, 30)
            qvel = f["observations/qvel"][:]        # (T, 28)
            ctrl = f["actions/ctrl"][:]              # (T, 16)

        # Object position + finger slices
        obj_pos = qpos[:, QPOS_OBJ_SLICE]           # (T,  3)  bottle xyz
        qpos_f  = qpos[:, QPOS_FINGER_SLICE]         # (T, 16)
        qvel_f  = qvel[:, QVEL_FINGER_SLICE]          # (T, 16)

        self.obs  = np.concatenate([obj_pos, qpos_f, qvel_f], axis=1).astype(np.float32)   # (T, 35)
        self.ctrl = ctrl.astype(np.float32)                                         # (T, 16)

        # Valid start indices: enough room for a full chunk of future actions
        self.length = max(len(self.obs) - chunk_size + 1, 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.from_numpy(self.obs[idx])                                       # (32,)
        act = torch.from_numpy(self.ctrl[idx : idx + self.chunk_size])              # (chunk, 16)
        if self.chunk_size == 1:
            act = act.squeeze(0)                                                     # (16,)
        return obs, act


# ── Public helpers ──────────────────────────────────────────────────────────

class DemoDataset(ConcatDataset):
    """
    Concatenation of every ``*.h5`` demo in *demo_dir*.

    Parameters
    ----------
    demo_dir : path to folder with HDF5 files
    chunk_size : number of future actions per sample (1 = BC, >1 = ACT / Diffusion)
    """

    def __init__(self, demo_dir: str | Path, chunk_size: int = 1) -> None:
        demo_dir = Path(demo_dir)
        files = sorted(demo_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .h5 demos found in {demo_dir}")
        datasets = [_SingleDemo(f, chunk_size=chunk_size) for f in files]
        super().__init__(datasets)

        # Store normalisation stats across the full dataset
        all_obs  = np.concatenate([d.obs  for d in datasets], axis=0)
        all_ctrl = np.concatenate([d.ctrl for d in datasets], axis=0)

        self.obs_mean  = all_obs.mean(axis=0)
        self.obs_std   = all_obs.std(axis=0).clip(min=1e-6)
        self.act_mean  = all_ctrl.mean(axis=0)
        self.act_std   = all_ctrl.std(axis=0).clip(min=1e-6)

        print(f"[Dataset] Loaded {len(files)} demos — "
              f"{len(self)} samples  (chunk_size={chunk_size})")


def load_demos(demo_dir: str | Path,
               chunk_size: int = 1,
               batch_size: int = 64,
               val_ratio: float = 0.1,
               seed: int = 42,
               num_workers: int = 0):
    """
    Convenience loader → returns ``(train_loader, val_loader, norm_stats)``.

    ``norm_stats`` is a dict with obs/act mean/std as torch tensors.
    """
    from torch.utils.data import DataLoader, random_split

    ds = DemoDataset(demo_dir, chunk_size=chunk_size)

    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    norm_stats = {
        "obs_mean": torch.from_numpy(ds.obs_mean),
        "obs_std":  torch.from_numpy(ds.obs_std),
        "act_mean": torch.from_numpy(ds.act_mean),
        "act_std":  torch.from_numpy(ds.act_std),
    }

    return train_loader, val_loader, norm_stats
