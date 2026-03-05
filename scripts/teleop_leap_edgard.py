"""
teleop_leap.py — Hand teleoperation with direct angle retargeting.

Pipeline (30 Hz vision loop):
  ZED left camera frame (monocular mode)
    → MediaPipe hand tracking (single camera)
    → direct angle retargeting  (MediaPipe 3-D joint angles → 16 LEAP joints)
    → One Euro filtered mocap position  (hand proxy follows wrist, Y fixed)
    → One Euro filtered joint targets
    → MuJoCo position actuators

  Stereo depth is intentionally disabled (STEREO_DEPTH = False).
  Once IK is fully tuned, flip that flag to re-enable epipolar + triangulation.

Run with mjpython (NOT plain python — cv2.imshow conflicts with Cocoa on macOS):
    mjpython teleop_leap.py

Tuning guide (constants block below):
    JOINT_MC / JOINT_BETA  — joint filter: lower MC = smoother, higher beta = less lag
    POS_MC   / POS_BETA    — position filter
    X/Z_SCALE              — workspace size in simulation metres
    DEPTH_SCALE            — how much stereo depth maps to sim-Y movement (only when STEREO_DEPTH=True)
"""

import os
import multiprocessing as _mp
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from vision.camera                       import ZEDCamera
from vision.detectors                    import StereoHandTracker
import vision.geometry                   as geo
from vision.smoother                     import OneEuroFilter
from robots.leap_hand.ik_retargeting     import IKRetargeter, palm_quat

# ── Tunable constants ─────────────────────────────────────────────────────────
CAMERA_ID    = 0       # 0 = webcam / seule caméra détectée. Mettre 1 quand la ZED est branchée.
N_SUBSTEPS   = 16       # lighter physics load for better FPS/thermals

# ── Mode toggle ───────────────────────────────────────────────────────────────
# False = monocular (left frame only, Y fixed) → tune IK first.
# True  = stereo    (epipolar + triangulated depth) → requires ZED camera.
STEREO_DEPTH = True

# One Euro Filter — joints (16-dim)
JOINT_FREQ   = 30.0    # Hz: expected vision loop rate
JOINT_MC     = 1.0     # min_cutoff: lower → smoother at rest
JOINT_BETA   = 0.03    # beta:       higher → less lag during fast motion

# One Euro Filter — wrist position (3-dim)
POS_FREQ     = 30.0
POS_MC       = 0.8
POS_BETA     = 0.005

# Workspace mapping
# In the new pinhole back-projection model the workspace scales automatically
# with depth (back_project returns real metres).  X_SCALE / Z_SCALE are kept
# here as legacy references but are no longer applied to the position output.
X_SCALE      = 0.3     # legacy — no longer used
Z_SCALE      = 0.3     # legacy — no longer used

# Depth (stereo Z) → sim Y — only used when STEREO_DEPTH = True
DEPTH_MIN_M  = 0.20
DEPTH_MAX_M  = 0.90
DEPTH_MID_M  = 0.45    # neutral depth → hand sits at START_Y
DEPTH_SCALE  = 2.0     # m of sim-Y movement per m of depth change
TRANS_SCALE  = 2.0     # global translation gain (higher = more movement)
START_Y      = 0.30    # initial sim Y (forward) of the hand proxy — also used as fixed Y
START_Z      = 0.45    # initial sim Z (height) of the hand proxy

# Epipolar constraint — only checked when STEREO_DEPTH = True
EPIPOLAR_TOL = 40      # px (relaxed — tighten once Y_OFFSET_PX is tuned for your unit)

# ── Wrist rotation (via mocap_quat) ──────────────────────────────────────────
# Rz = roll  from knuckle line (INDEX_MCP → RING_MCP)
# Rx = pitch from wrist→middle-MCP tilt (uses MediaPipe .z depth)
# Ry = yaw   from index↔pinky depth skew (uses MediaPipe .z depth)
WRIST_SCALE    = 2.0    # gain on detected angle delta (shared X/Y/Z)  [-20%]
WRIST_DZ_RX    = 0.03   # rad (~7°) — deadzone pitch
WRIST_DZ_RY    = 0.08   # rad (~7°) — deadzone yaw
WRIST_DZ_RZ    = 0.12   # rad (~6°) — deadzone roll
WRIST_MAX_RAD  = 2.0    # max clamp (~45°, reduced to avoid vibration at limits)
MOCAP_MAX_STEP = 0.010  # max position change per frame (m) — prevents teleportation
RY_POS_BOOST   = 1.8    # compensate MediaPipe .z asymmetry: positive yaw harder to reach
RY_NEG_BOOST   = 5    # boost negative yaw sensitivity to match positive side
RZ_RY_DECOUPLE = 0.6    # subtract this × Ry from Rz to cancel cross-talk

# One Euro Filters for wrist angles (1-dim each)
WRIST_FREQ     = 30.0
WRIST_MC       = 0.3
WRIST_BETA     = 0.01

# Hold last pose when hand disappears (avoids jerk on tracking loss)
HOLD_POSE_SEC  = 1.0   # seconds to hold last pose before resetting

# Rest orientation of the hand (unchanged from last push)
BASE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # Rx(180°): palm facing up (stable physics)

# ── Handedness filter ─────────────────────────────────────────────────────────────────────
# ZED is a non-mirrored camera: your RIGHT hand appears on the LEFT side of the
# image, so MediaPipe labels it "Left".  Flip to "Right" if using a mirrored cam.
TARGET_HAND   = "Left"   # tracks your physical right hand on a non-mirrored ZED

# cv2.imshow conflicts with mjpython's Cocoa event loop on macOS.
# Set to True only when running with plain `python` (not `mjpython`).
SHOW_CAMERA  = True

# ── Quaternion helpers ────────────────────────────────────────────────────────
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_ensure_hemi(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Negate q if it is in the opposite hemisphere from ref (avoids filter flips)."""
    return -q if np.dot(q, ref) < 0 else q


# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_DIR, "robots", "leap_hand", "scene.xml")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_hand(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Teleport the LEAP palm to the starting position before physics runs.

    Without this, the palm starts at its XML-defined origin and falls under
    gravity before the weld constraint can engage on the first frame.
    """
    mid  = model.body("hand_proxy").mocapid[0]
    pos  = np.array([0.0, START_Y, START_Z])

    data.mocap_pos[mid]  = pos
    data.mocap_quat[mid] = BASE_QUAT.copy()

    # Palm freejoint: initialize at the weld target pose (BASE_QUAT * relpose)
    # relpose = Rx(-90°) → palm at Rx(180°) * Rx(-90°) = Rx(+90°): fingers +Z, palm -Y
    RELPOSE_QUAT = np.array([0.5, -0.5, 0.5, 0.5])
    palm_quat_init = _quat_mul(BASE_QUAT, RELPOSE_QUAT)

    jid  = model.joint("palm_free").id
    addr = model.jnt_qposadr[jid]
    data.qpos[addr:addr+3] = pos
    data.qpos[addr+3:addr+7] = palm_quat_init

    mujoco.mj_forward(model, data)


def _update(data:     mujoco.MjData,
            zed:      ZEDCamera,
            tracker:  StereoHandTracker,
            ik:       IKRetargeter,
            pos_f:    OneEuroFilter,
            joint_f:  OneEuroFilter,
            orient_f: OneEuroFilter,
            pitch_f:  OneEuroFilter,
            yaw_f:    OneEuroFilter,
            mid:      int) -> None:
    """
    Single-frame update: capture → detect → retarget → actuate.

    Left camera drives finger retargeting (always).
    When STEREO_DEPTH=True, both cameras triangulate wrist depth for sim-Y.
    If stereo fails on a frame, the last good position is kept but fingers
    still update — no frame is ever fully dropped.
    """
    global _wrist_ref_angle, _wrist_calib_count, _pitch_ref_angle, _pitch_calib_count, _yaw_ref_angle, _yaw_calib_count, _last_hand_time, _calibrate_flag
    import time as _time
    frame_l, frame_r = zed.get_frames()
    if frame_l is None:
        return

    # ── Reset (touche R) ──────────────────────────────────────────────────
    if _reset_flag is not None and _reset_flag.value:
        _reset_flag.value = 0
        _wrist_ref_angle = None; _wrist_calib_count = 0
        _pitch_ref_angle = None; _pitch_calib_count = 0
        _yaw_ref_angle   = None; _yaw_calib_count  = 0
        orient_f.reset(); pitch_f.reset(); yaw_f.reset()
        joint_f.reset()
        data.ctrl[:] = 0.0
        data.qvel[:] = 0.0
        _init_hand(model, data)
        print("[RESET] Hand position, fingers & calibration reset (R key)")

    h, w, _ = frame_l.shape
    res_l, res_r = tracker.process(frame_l, frame_r)

    # ── Left camera must see a hand to do anything ────────────────────────
    if not res_l.multi_hand_landmarks:
        elapsed = _time.monotonic() - _last_hand_time
        holding = elapsed < HOLD_POSE_SEC and _last_hand_time > 0

        if not holding:
            orient_f.reset()
            pitch_f.reset()
            yaw_f.reset()
            data.mocap_quat[mid] = BASE_QUAT.copy()

        hold_label = f"HOLD {HOLD_POSE_SEC - elapsed:.1f}s" if holding else "NO HAND"
        hold_col   = (0, 200, 255) if holding else (0, 0, 255)
        if SHOW_CAMERA:
            cv2.putText(frame_l, f"L: {hold_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hold_col, 2)
            if frame_r is not None:
                r_det = "R: HAND" if res_r.multi_hand_landmarks else "R: NO HAND"
                r_col = (0, 220, 0) if res_r.multi_hand_landmarks else (0, 0, 255)
                cv2.putText(frame_r, r_det, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_col, 2)
                _show(np.hstack([frame_l, frame_r]))
            else:
                _show(frame_l)
        else:
            _show(frame_l)
        return

    _last_hand_time = _time.monotonic()

    lm_l = res_l.multi_hand_landmarks[0].landmark
    cam   = geo.ZED2I

    # ── Palm center position (avg of wrist + 4 MCP) ────────────────────
    _palm_ids = (0, 5, 9, 13, 17)  # wrist, index/middle/ring/pinky MCP
    u_w = sum(lm_l[i].x for i in _palm_ids) / len(_palm_ids) * w
    v_w = sum(lm_l[i].y for i in _palm_ids) / len(_palm_ids) * h
    sim_x = (u_w - cam.cx) / cam.fx * START_Y * TRANS_SCALE
    sim_z =  START_Z + (-(v_w - cam.cy) / cam.fy * START_Y) * TRANS_SCALE

    # ── Depth axis (sim Y): stereo triangulation or fixed ─────────────────
    sim_y      = START_Y
    depth_cm   = None          # None = no stereo depth available
    hud_mode   = "MONO [L]"   # displayed mode label
    hud_col    = (0, 165, 255) # orange = mono
    hud_detail = ""

    if STEREO_DEPTH and res_r.multi_hand_landmarks:
        lm_r = res_r.multi_hand_landmarks[0].landmark
        py_l = int(lm_l[0].y * h)
        py_r = int(lm_r[0].y * h)
        valid, epi_err = geo.check_epipolar_constraint(
            py_l, py_r, tolerance_px=EPIPOLAR_TOL)

        if valid:
            p3d = geo.stereo_hand_3d(lm_l, lm_r, w, h,
                                      depth_min_m=DEPTH_MIN_M,
                                      depth_max_m=DEPTH_MAX_M)
            if p3d is not None:
                x_m, y_m, z_m = p3d
                sim_x = -x_m * TRANS_SCALE
                sim_y = START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE * TRANS_SCALE
                sim_z = START_Z + (-y_m) * TRANS_SCALE
                depth_cm = z_m * 100
                hud_mode   = "STEREO [L+R]"
                hud_col    = (0, 220, 0)
                hud_detail = f"epi={epi_err:.0f}px"
            else:
                hud_detail = f"bad disparity epi={epi_err:.0f}px"
        else:
            hud_detail = f"epi REJECTED err={epi_err:.0f}px"

    # ── Compute raw wrist angles ────────────────────────────────────────
    idx_mcp  = lm_l[5]
    ring_mcp = lm_l[13]
    raw_angle = np.arctan2(ring_mcp.y - idx_mcp.y, ring_mcp.x - idx_mcp.x)

    mid_mcp  = lm_l[9]
    wrist_lm = lm_l[0]
    dy_p = mid_mcp.y - wrist_lm.y
    dz_p = mid_mcp.z - wrist_lm.z
    raw_pitch = np.arctan2(dz_p, dy_p)

    idx_mcp_y = lm_l[5]
    pky_mcp_y = lm_l[17]
    dx_y = pky_mcp_y.x - idx_mcp_y.x
    dz_y = pky_mcp_y.z - idx_mcp_y.z
    raw_yaw = dz_y / max(abs(dx_y), 0.01)

    # ── Press A → snapshot current orientation as reference ───────────
    if _calibrate_flag:
        _wrist_ref_angle = raw_angle
        _pitch_ref_angle = raw_pitch
        _yaw_ref_angle   = raw_yaw
        orient_f.reset()
        pitch_f.reset()
        yaw_f.reset()
        pos_f.reset()
        joint_f.reset()
        _calibrate_flag = False
        print("[CALIB] Orientation de référence capturée.")

    # ── Before calibration: hand frozen at start pose ─────────────────
    if _wrist_ref_angle is None:
        wrist_x = wrist_y = wrist_z = wrist_z_raw = 0.0
    else:
        # ── Mocap position (only after calibration) ──────────────────
        raw_pos = np.array([sim_x, sim_y, sim_z])
        new_pos = pos_f(raw_pos)
        delta_pos = new_pos - data.mocap_pos[mid]
        dist = np.linalg.norm(delta_pos)
        if dist > MOCAP_MAX_STEP:
            new_pos = data.mocap_pos[mid] + delta_pos * (MOCAP_MAX_STEP / dist)
        data.mocap_pos[mid] = new_pos

        # Roll (Rz)
        delta = (raw_angle - _wrist_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta = -delta
        delta = float(np.clip(delta, -0.9, 0.9))
        delta = float(orient_f(np.array([delta]))[0])
        if abs(delta) < WRIST_DZ_RZ:
            delta = 0.0
        else:
            delta = np.sign(delta) * (abs(delta) - WRIST_DZ_RZ)
        wrist_z = float(np.clip(delta * WRIST_SCALE * 0.9, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Pitch (Rx)
        delta_p = (raw_pitch - _pitch_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta_p = delta_p
        delta_p = float(np.clip(delta_p, -0.9, 0.9))
        delta_p = float(pitch_f(np.array([delta_p]))[0])
        if abs(delta_p) < WRIST_DZ_RX:
            delta_p = 0.0
        else:
            delta_p = np.sign(delta_p) * (abs(delta_p) - WRIST_DZ_RX)
        wrist_x = float(np.clip(-delta_p * WRIST_SCALE * 5.0, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Yaw (Ry) — boost positive side to compensate MediaPipe .z asymmetry
        delta_y = (raw_yaw - _yaw_ref_angle + np.pi) % (2 * np.pi) - np.pi
        delta_y = -delta_y
        if delta_y > 0:
            delta_y *= RY_POS_BOOST
        else:
            delta_y *= RY_NEG_BOOST
        delta_y = float(np.clip(delta_y, -0.9, 0.9))
        delta_y = float(yaw_f(np.array([delta_y]))[0])
        if abs(delta_y) < WRIST_DZ_RY:
            delta_y = 0.0
        else:
            delta_y = np.sign(delta_y) * (abs(delta_y) - WRIST_DZ_RY)
        wrist_y = float(np.clip(-delta_y * WRIST_SCALE, -WRIST_MAX_RAD, WRIST_MAX_RAD))

        # Decouple: shrink Rz toward zero when Ry is active (kills cross-talk)
        wrist_z_raw = wrist_z
        reduction = RZ_RY_DECOUPLE * abs(wrist_y)
        if abs(wrist_z) > reduction:
            wrist_z = wrist_z - np.sign(wrist_z) * reduction
        else:
            wrist_z = 0.0

        # Incremental rotation: small per-axis quats applied to BASE_QUAT
        # in the body-local frame (avoids gimbal lock)
        half_x = wrist_x / 2.0
        dq_x = np.array([np.cos(half_x), np.sin(half_x), 0.0, 0.0])
        half_y = wrist_z / 2.0
        dq_y = np.array([np.cos(half_y), 0.0, np.sin(half_y), 0.0])
        half_z = wrist_y / 2.0
        dq_z = np.array([np.cos(half_z), 0.0, 0.0, np.sin(half_z)])
        # Apply in body-local frame: BASE * dq_x * dq_y * dq_z
        q = _quat_mul(BASE_QUAT, dq_x)
        q = _quat_mul(q, dq_y)
        q = _quat_mul(q, dq_z)
        q = q / np.linalg.norm(q)
        data.mocap_quat[mid] = q

        # ── Direct angle retargeting (only after calibration) ────────
        q_raw    = ik.retarget(None, lm_l)
        q_smooth = joint_f(q_raw)
        for i in range(data.model.nu):
            lo, hi = data.model.actuator_ctrlrange[i]
            q_smooth[i] = np.clip(q_smooth[i], lo, hi)
        data.ctrl[:] = q_smooth

    if SHOW_CAMERA:
        tracker.draw_landmarks(frame_l, res_l)

        # Top-left: mode indicator
        cv2.putText(frame_l, hud_mode, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2)
        if hud_detail:
            cv2.putText(frame_l, hud_detail, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        # Calibration status banner
        if _wrist_ref_angle is None:
            calib_txt = "NON CALIBRE — Appuyez sur A (MuJoCo)"
            cv2.putText(frame_l, calib_txt, (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame_l, "CALIBRE (A = recalibrer)", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

        # Top-right: depth readout (large)
        if depth_cm is not None:
            depth_str = f"DEPTH: {depth_cm:.0f} cm"
            d_col = (0, 220, 0)
        else:
            depth_str = "DEPTH: ---"
            d_col = (0, 165, 255)
        txt_size = cv2.getTextSize(depth_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame_l, depth_str, (w - txt_size[0] - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_col, 2)

        # Bottom: rotation debug per axis (large text)
        rx_deg = float(np.degrees(wrist_x))
        ry_deg = float(np.degrees(wrist_y))
        rz_deg = float(np.degrees(wrist_z))
        rz_raw_deg = float(np.degrees(wrist_z_raw))

        rz_col = (0, 220, 0) if abs(rz_deg) > 0.1 else (100, 100, 100)
        cv2.putText(frame_l, f"Rz(roll) : {rz_deg:+6.1f}  (raw {rz_raw_deg:+.1f})",
                    (15, h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rz_col, 2)

        rx_col = (255, 100, 0) if abs(rx_deg) > 0.1 else (100, 100, 100)
        wz0 = lm_l[0].z; mz9 = lm_l[9].z
        cv2.putText(frame_l, f"Rx(pitch): {rx_deg:+6.1f}  wrist.z={wz0:+.2f} mid.z={mz9:+.2f}",
                    (15, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rx_col, 2)

        ry_col = (0, 220, 220) if abs(ry_deg) > 0.1 else (100, 100, 100)
        iz5 = lm_l[5].z; pz17 = lm_l[17].z
        cv2.putText(frame_l, f"Ry(yaw)  : {ry_deg:+6.1f}  idx.z={iz5:+.2f} pky.z={pz17:+.2f}",
                    (15, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ry_col, 2)

        cv2.putText(frame_l, f"SENT  Rx:{rx_deg:+.0f}  Ry:{ry_deg:+.0f}  Rz:{rz_deg:+.0f}",
                    (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 220), 3)

        # Show both cameras side by side with detection status
        if frame_r is not None:
            tracker.draw_landmarks(frame_r, res_r)
            r_det = "R: HAND" if res_r.multi_hand_landmarks else "R: NO HAND"
            r_col = (0, 220, 0) if res_r.multi_hand_landmarks else (0, 0, 255)
            cv2.putText(frame_r, r_det, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_col, 2)
            display = np.hstack([frame_l, frame_r])
        else:
            display = frame_l

    _show(display if SHOW_CAMERA else frame_l)


_wrist_ref_angle = None
_pitch_ref_angle = None
_yaw_ref_angle = None
_last_hand_time = 0.0
_calibrate_flag = False
_reset_flag = None
_frame_q = None
_show_counter = 0
_SHOW_EVERY = 5        # send 1 frame out of 5 to the viewer
_VIEWER_SCALE = 0.35   # stronger downscale for lower CPU/GPU load


def _show(frame):
    """Send a downscaled frame to the viewer subprocess every N calls."""
    global _show_counter
    if not SHOW_CAMERA or _frame_q is None:
        return
    _show_counter += 1
    if _show_counter % _SHOW_EVERY != 0:
        return
    small = cv2.resize(frame, None, fx=_VIEWER_SCALE, fy=_VIEWER_SCALE,
                       interpolation=cv2.INTER_NEAREST)
    if _frame_q.full():
        try:
            _frame_q.get_nowait()
        except Exception:
            pass
    try:
        _frame_q.put_nowait(small)
    except Exception:
        pass


def _key_callback(keycode):
    """MuJoCo viewer key callback: press A to (re-)calibrate wrist orientation."""
    global _calibrate_flag
    if keycode == 65:  # GLFW_KEY_A
        _calibrate_flag = True
        print("[CALIB] Touche A détectée — calibration au prochain frame avec main visible.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _frame_q

    # Hardware
    # Pass y_offset so vertical alignment is ready when STEREO_DEPTH is re-enabled.
    zed     = ZEDCamera(camera_id=CAMERA_ID,
                        y_offset=geo.Y_OFFSET_PX if STEREO_DEPTH else 0)
    tracker = StereoHandTracker()

    # Physics
    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data  = mujoco.MjData(model)

    # Mocap body index for hand_proxy
    mid = model.body("hand_proxy").mocapid[0]

    # Retargeter and filters
    ik       = IKRetargeter(model)
    pos_f    = OneEuroFilter(POS_FREQ,    min_cutoff=POS_MC,    beta=POS_BETA)
    joint_f  = OneEuroFilter(JOINT_FREQ,  min_cutoff=JOINT_MC,  beta=JOINT_BETA)
    orient_f = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    pitch_f  = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)
    yaw_f    = OneEuroFilter(WRIST_FREQ, min_cutoff=WRIST_MC, beta=WRIST_BETA)

    # Spawn hand at rest position
    _init_hand(model, data)

    # Camera viewer in a separate lightweight process (only imports cv2,
    # NOT mujoco — avoids the Cocoa / OpenGL conflict with mjpython on macOS).
    viewer_proc = None
    if SHOW_CAMERA:
        from _camera_viewer import viewer_loop
        ctx = _mp.get_context("spawn")
        _frame_q = ctx.Queue(maxsize=2)
        _reset_flag = ctx.Value('i', 0)
        viewer_proc = ctx.Process(target=viewer_loop, args=(_frame_q, _reset_flag), daemon=True)
        viewer_proc.start()

    print("─" * 60)
    print("  Binocular Hand Teleoperation (Direct Angle Retargeting)")
    print("  Move your right hand in front of the ZED camera.")
    print("  Press  A  dans le viewer MuJoCo pour calibrer l'orientation.")
    print("  Press  Q  in the camera window  or  ESC  in the")
    print("  MuJoCo viewer to quit.")
    print("─" * 60)

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as v:
        while v.is_running():
            _update(data, zed, tracker, ik, pos_f, joint_f, orient_f, pitch_f, yaw_f, mid)

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model, data)
            v.sync()

    # Clean shutdown
    if viewer_proc is not None and _frame_q is not None:
        _frame_q.put(None)
        viewer_proc.join(timeout=3)

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
