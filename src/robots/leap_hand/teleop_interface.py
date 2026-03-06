# src/robots/leap_hand/teleop_interface.py
"""
LeapTeleopInterface — stateful controller that converts MediaPipe landmarks
into MuJoCo mocap position/orientation + 16 LEAP finger actuators.

All geometric constants are read from ``configs/teleop_config.yaml`` so nothing
is hard-coded here.
"""

import numpy as np
import mujoco
from src.utils.smoother import OneEuroFilter
from src.robots.leap_hand.retargeter import IKRetargeter


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product (MuJoCo convention: w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


class LeapTeleopInterface:
    """
    Processes one frame of MediaPipe tracking and writes to MuJoCo ``data``:
      - ``data.mocap_pos``  (wrist XYZ)
      - ``data.mocap_quat`` (wrist orientation via incremental quaternions)
      - ``data.ctrl``       (16 finger actuators)

    Call ``calibrate()`` once the hand is visible to snapshot the current
    orientation as the zero-reference.  Before calibration the hand stays frozen
    at its start pose.
    """

    def __init__(self, model: mujoco.MjModel, config: dict):
        self.model = model
        self.config = config
        self.mid = model.body("hand_proxy").mocapid[0]
        self.ik = IKRetargeter(model)

        # ── Filters (one per signal family) ──────────────────────────────────
        filt_cfg = config["filters"]
        self.filters = {
            "pos":    OneEuroFilter(freq=filt_cfg["position"]["freq"],
                                    min_cutoff=filt_cfg["position"]["mc"],
                                    beta=filt_cfg["position"]["beta"]),
            "joint":  OneEuroFilter(freq=filt_cfg["joint"]["freq"],
                                    min_cutoff=filt_cfg["joint"]["mc"],
                                    beta=filt_cfg["joint"]["beta"]),
            "roll":   OneEuroFilter(freq=filt_cfg["wrist"]["freq"],
                                    min_cutoff=filt_cfg["wrist"]["mc"],
                                    beta=filt_cfg["wrist"]["beta"]),
            "pitch":  OneEuroFilter(freq=filt_cfg["wrist"]["freq"],
                                    min_cutoff=filt_cfg["wrist"]["mc"],
                                    beta=filt_cfg["wrist"]["beta"]),
            "yaw":    OneEuroFilter(freq=filt_cfg["wrist"]["freq"],
                                    min_cutoff=filt_cfg["wrist"]["mc"],
                                    beta=filt_cfg["wrist"]["beta"]),
        }

        # ── Calibration reference angles (None ⇒ uncalibrated) ──────────────
        self.ref_angles: dict = {"roll": None, "pitch": None, "yaw": None}

        # ── Constants from config ────────────────────────────────────────────
        wc = config["wrist_control"]
        self.base_quat     = np.array([0.0, 1.0, 0.0, 0.0])
        self.wrist_scale   = wc["scale"]
        self.dz_rx         = wc["deadzones"]["rx"]
        self.dz_ry         = wc["deadzones"]["ry"]
        self.dz_rz         = wc["deadzones"]["rz"]
        self.max_rad       = wc["max_rad"]
        self.max_step_m    = wc["max_step_m"]
        self.ry_pos_boost  = wc["yaw_boost"]["pos"]
        self.ry_neg_boost  = wc["yaw_boost"]["neg"]
        # Decouple factor: shrink roll toward zero when yaw is active
        self.rz_ry_decouple = config.get("wrist_control", {}).get("rz_ry_decouple", 0.6)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self, raw_angles: dict) -> None:
        """
        Snapshot the current hand orientation as the zero-reference.

        Parameters
        ----------
        raw_angles : dict
            Must contain keys ``"roll"``, ``"pitch"``, ``"yaw"`` with the
            current raw MediaPipe angles.
        """
        self.ref_angles = raw_angles.copy()
        for f in self.filters.values():
            f.reset()
        print("[CALIB] Reference orientation captured.")

    @property
    def is_calibrated(self) -> bool:
        return self.ref_angles["roll"] is not None

    def compute_raw_angles(self, lm) -> dict:
        """
        Compute the three raw wrist angles from MediaPipe landmarks.

        Parameters
        ----------
        lm : list
            MediaPipe ``landmark`` list (21 entries).

        Returns
        -------
        dict with keys ``"roll"``, ``"pitch"``, ``"yaw"``.
        """
        # Roll — angle between index-MCP and ring-MCP across the knuckle line
        idx_mcp = lm[5]
        ring_mcp = lm[13]
        raw_roll = np.arctan2(ring_mcp.y - idx_mcp.y,
                              ring_mcp.x - idx_mcp.x)

        # Pitch — wrist-to-middle-MCP tilt using MediaPipe's pseudo-depth z
        mid_mcp  = lm[9]
        wrist_lm = lm[0]
        dy_p = mid_mcp.y - wrist_lm.y
        dz_p = mid_mcp.z - wrist_lm.z
        raw_pitch = np.arctan2(dz_p, dy_p)

        # Yaw — lateral palm tilt (index vs. pinky z difference)
        idx_mcp_y = lm[5]
        pky_mcp_y = lm[17]
        dx_y = pky_mcp_y.x - idx_mcp_y.x
        dz_y = pky_mcp_y.z - idx_mcp_y.z
        raw_yaw = dz_y / max(abs(dx_y), 0.01)

        return {"roll": raw_roll, "pitch": raw_pitch, "yaw": raw_yaw}

    def update(self, data: mujoco.MjData, hand_landmarks, sim_pos: np.ndarray) -> None:
        """
        Process one frame of tracking data and update MuJoCo control signals.

        Parameters
        ----------
        data : mujoco.MjData
            Live simulation data (mocap_pos, mocap_quat, ctrl are written).
        hand_landmarks : list
            MediaPipe ``landmark`` list (21 entries) for one hand.
        sim_pos : np.ndarray, shape (3,)
            Target wrist position in MuJoCo world coordinates, computed by
            the calling script from stereo/monocular back-projection.
        """
        if not self.is_calibrated:
            return

        wc = self.config.get("wrist_control", {})
        wrist_enabled     = wc.get("enabled", True)
        lock_orientation  = wc.get("lock_orientation", False)

        # ── 1. Position update ───────────────────────────────────────────────
        if wrist_enabled:
            new_pos = self.filters["pos"](sim_pos)
            data.mocap_pos[self.mid] = self._apply_step_limit(
                data.mocap_pos[self.mid], new_pos
            )

        # ── 2. Orientation update ────────────────────────────────────────────
        raw = self.compute_raw_angles(hand_landmarks)

        # ---- Roll (Rz) ----
        # Wrap delta into [-π, π] and invert direction for sim convention
        delta_rz = (raw["roll"] - self.ref_angles["roll"] + np.pi) % (2 * np.pi) - np.pi
        delta_rz = -delta_rz
        delta_rz = float(np.clip(delta_rz, -0.9, 0.9))
        delta_rz = float(self.filters["roll"](np.array([delta_rz]))[0])
        # Apply dead-zone
        if abs(delta_rz) < self.dz_rz:
            delta_rz = 0.0
        else:
            delta_rz = np.sign(delta_rz) * (abs(delta_rz) - self.dz_rz)
        wrist_z = float(np.clip(delta_rz * self.wrist_scale * 0.9,
                                -self.max_rad, self.max_rad))

        # ---- Pitch (Rx) ----
        delta_rx = (raw["pitch"] - self.ref_angles["pitch"] + np.pi) % (2 * np.pi) - np.pi
        delta_rx = float(np.clip(delta_rx, -0.9, 0.9))
        delta_rx = float(self.filters["pitch"](np.array([delta_rx]))[0])
        if abs(delta_rx) < self.dz_rx:
            delta_rx = 0.0
        else:
            delta_rx = np.sign(delta_rx) * (abs(delta_rx) - self.dz_rx)
        wrist_x = float(np.clip(-delta_rx * self.wrist_scale * 5.0,
                                -self.max_rad, self.max_rad))

        # ---- Yaw (Ry) with asymmetric boost ----
        delta_ry = (raw["yaw"] - self.ref_angles["yaw"] + np.pi) % (2 * np.pi) - np.pi
        delta_ry = -delta_ry
        # Asymmetric boost: different sensitivity for positive/negative yaw
        if delta_ry > 0:
            delta_ry *= self.ry_pos_boost
        else:
            delta_ry *= self.ry_neg_boost
        delta_ry = float(np.clip(delta_ry, -0.9, 0.9))
        delta_ry = float(self.filters["yaw"](np.array([delta_ry]))[0])
        if abs(delta_ry) < self.dz_ry:
            delta_ry = 0.0
        else:
            delta_ry = np.sign(delta_ry) * (abs(delta_ry) - self.dz_ry)
        wrist_y = float(np.clip(-delta_ry * self.wrist_scale,
                                -self.max_rad, self.max_rad))

        # ---- RZ/RY decoupling ----
        # When yaw (Ry) is active, shrink roll (Rz) to prevent interference
        reduction = self.rz_ry_decouple * abs(wrist_y)
        if abs(wrist_z) > reduction:
            wrist_z = wrist_z - np.sign(wrist_z) * reduction
        else:
            wrist_z = 0.0

        # ---- Compose incremental rotation quaternion ----
        # Order: Rx (pitch) → Ry (roll mapped to sim Y) → Rz (yaw mapped to sim Z)
        half_x = wrist_x / 2.0
        dq_x = np.array([np.cos(half_x), np.sin(half_x), 0.0, 0.0])

        half_y = wrist_z / 2.0
        dq_y = np.array([np.cos(half_y), 0.0, np.sin(half_y), 0.0])

        half_z = wrist_y / 2.0
        dq_z = np.array([np.cos(half_z), 0.0, 0.0, np.sin(half_z)])

        q = _quat_mul(self.base_quat, dq_x)
        q = _quat_mul(q, dq_y)
        q = _quat_mul(q, dq_z)
        q = q / np.linalg.norm(q)
        if wrist_enabled and not lock_orientation:
            data.mocap_quat[self.mid] = q

        # ── 3. Finger retargeting ────────────────────────────────────────────
        q_raw = self.ik.retarget(None, hand_landmarks)
        q_smooth = self.filters["joint"](q_raw)
        # Clamp to actuator control range
        for i in range(data.model.nu):
            lo, hi = data.model.actuator_ctrlrange[i]
            q_smooth[i] = np.clip(q_smooth[i], lo, hi)
        data.ctrl[:] = q_smooth

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_step_limit(self, current: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
        """Clamp per-frame position delta to ``max_step_m``."""
        delta = target - current
        dist = np.linalg.norm(delta)
        if dist > self.max_step_m:
            return current + delta * (self.max_step_m / dist)
        return target