# scripts/collect_data.py
"""
Data collection script — enhanced version of teleop_leap.py.

Same pipeline as teleop_leap.py, plus:
  - Press 'S' in the MuJoCo viewer to Start/Stop recording.
  - Each recording session is auto-saved as ``data/demo_YYYYMMDD_HHMMSS.h5``.

Run with mjpython on macOS:
    mjpython scripts/collect_data.py
"""

import multiprocessing as _mp
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
# regardless of the working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import yaml
import mujoco
import mujoco.viewer

import src.vision.geometry as geo
from src.vision.camera import ZEDCamera
from src.vision.detectors import StereoHandTracker
from src.robots.leap_hand.teleop_interface import LeapTeleopInterface
from src.utils.data_logger import DataLogger

# ── Paths ─────────────────────────────────────────────────────────────────────
_CONFIG_PATH = _ROOT / "configs" / "teleop_config.yaml"


def main() -> None:
    # ── Load config ──────────────────────────────────────────────────────────
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    ws = config["workspace"]
    phys = config["physics"]
    hw = config["hardware"]
    stereo_enabled: bool = hw.get("stereo_depth", True)

    START_Y     = ws["start_y"]
    START_Z     = ws["start_z"]
    TRANS_SCALE = ws["trans_scale"]
    DEPTH_SCALE = ws["depth_scale"]
    DEPTH_MIN_M, DEPTH_MAX_M = ws["depth_range"]
    DEPTH_MID_M = ws["depth_mid"]
    HOLD_POSE_SEC = config.get("hold_pose_sec", 1.0)
    EPIPOLAR_TOL  = config.get("epipolar_tol", 40)
    SHOW_CAMERA   = config.get("show_camera", True)

    # ── Mutable state for key_callback ───────────────────────────────────────
    calibrate_flag = False
    record_toggle  = False       # toggled by 'S' key
    last_hand_time = 0.0

    # ── Hardware init ────────────────────────────────────────────────────────
    cam = geo.ZED2I
    zed = ZEDCamera(
        camera_id=hw["camera_id"],
        y_offset=geo.Y_OFFSET_PX if stereo_enabled else 0,
    )
    tracker = StereoHandTracker()

    # ── Physics init ─────────────────────────────────────────────────────────
    scene_xml = str(_ROOT / phys["scene_xml"])
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    robot = LeapTeleopInterface(model, config)
    mid = robot.mid

    data.mocap_pos[mid]  = np.array([0.0, START_Y, START_Z])
    data.mocap_quat[mid] = robot.base_quat.copy()
    mujoco.mj_forward(model, data)

    # ── Data logger ──────────────────────────────────────────────────────────
    save_dir = _ROOT / "data"
    logger = DataLogger(save_dir=save_dir)

    # ── Camera viewer (separate process) ─────────────────────────────────────
    frame_q = None
    viewer_proc = None
    _show_counter = 0
    _SHOW_EVERY = 10          # render preview every N frames (saves Cocoa overhead)
    _STEREO_COUNTER = 0
    _STEREO_EVERY = 3         # run right-eye MediaPipe every N frames (depth is slow-changing)
    _cached_res_r = None
    _VIEWER_SCALE = 0.35

    if SHOW_CAMERA:
        from src.vision._camera_viewer import viewer_loop
        ctx = _mp.get_context("spawn")
        frame_q = ctx.Queue(maxsize=2)
        viewer_proc = ctx.Process(
            target=viewer_loop, args=(frame_q,), daemon=True
        )
        viewer_proc.start()

    def _show(frame: np.ndarray) -> None:
        nonlocal _show_counter
        if not SHOW_CAMERA or frame_q is None:
            return
        _show_counter += 1
        if _show_counter % _SHOW_EVERY != 0:
            return
        small = cv2.resize(frame, None, fx=_VIEWER_SCALE, fy=_VIEWER_SCALE,
                           interpolation=cv2.INTER_NEAREST)
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except Exception:
                pass
        try:
            frame_q.put_nowait(small)
        except Exception:
            pass

    # ── Key callback ─────────────────────────────────────────────────────────
    def key_callback(keycode: int) -> None:
        nonlocal calibrate_flag, record_toggle
        if keycode == 65:           # 'A' — calibrate
            calibrate_flag = True
        elif keycode == 83:         # 'S' — start/stop recording
            record_toggle = True

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  LEAP Hand Data Collection — Binocular")
    print("  'A' = calibrate  |  'S' = start/stop recording")
    print(f"  Demos saved to: {save_dir.resolve()}")
    print("=" * 60)

    # ── Main loop ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        while v.is_running():
            # ---- Record toggle ----
            if record_toggle:
                record_toggle = False
                if logger.is_recording:
                    saved = logger.stop()
                    if saved:
                        print(f"[DATA] Demo saved → {saved}")
                else:
                    logger.start()

            # ---- Vision step ----
            frame_l, frame_r = zed.get_frames()
            if frame_l is None:
                continue

            h, w, _ = frame_l.shape

            # Right-eye frame-skip: stereo depth changes slowly; skip right-eye
            # MediaPipe every _STEREO_EVERY frames and cache the result.
            _STEREO_COUNTER += 1
            if _STEREO_COUNTER % _STEREO_EVERY == 0 or _cached_res_r is None:
                res_l, res_r = tracker.process(frame_l, frame_r)
                _cached_res_r = res_r
            else:
                rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
                res_l = tracker.tracker_left.process(rgb_l)
                res_r = _cached_res_r

            # ---- No hand ----
            if not res_l.multi_hand_landmarks:
                elapsed = time.monotonic() - last_hand_time
                holding = elapsed < HOLD_POSE_SEC and last_hand_time > 0
                if not holding:
                    robot.filters["roll"].reset()
                    robot.filters["pitch"].reset()
                    robot.filters["yaw"].reset()
                    data.mocap_quat[mid] = robot.base_quat.copy()
                if SHOW_CAMERA:
                    label = f"HOLD {HOLD_POSE_SEC - elapsed:.1f}s" if holding else "NO HAND"
                    col = (0, 200, 255) if holding else (0, 0, 255)
                    cv2.putText(frame_l, f"L: {label}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                    if frame_r is not None:
                        display = np.hstack([frame_l, frame_r])
                    else:
                        display = frame_l
                    _show(display)
                for _ in range(phys["n_substeps"]):
                    mujoco.mj_step(model, data)
                v.sync()
                continue

            last_hand_time = time.monotonic()
            lm_l = res_l.multi_hand_landmarks[0].landmark

            # ---- Compute sim_pos ----
            _palm_ids = (0, 5, 9, 13, 17)
            u_w = sum(lm_l[i].x for i in _palm_ids) / len(_palm_ids) * w
            v_w = sum(lm_l[i].y for i in _palm_ids) / len(_palm_ids) * h

            sim_x = (u_w - cam.cx) / cam.fx * START_Y * TRANS_SCALE
            sim_y = START_Y
            sim_z = START_Z + (-(v_w - cam.cy) / cam.fy * START_Y) * TRANS_SCALE

            if stereo_enabled and res_r.multi_hand_landmarks:
                lm_r = res_r.multi_hand_landmarks[0].landmark
                py_l = int(lm_l[0].y * h)
                py_r = int(lm_r[0].y * h)
                valid, epi_err = geo.check_epipolar_constraint(
                    py_l, py_r, tolerance_px=EPIPOLAR_TOL
                )
                if valid:
                    p3d = geo.stereo_hand_3d(
                        lm_l, lm_r, w, h,
                        depth_min_m=DEPTH_MIN_M,
                        depth_max_m=DEPTH_MAX_M,
                    )
                    if p3d is not None:
                        x_m, y_m, z_m = p3d
                        sim_x = -x_m * TRANS_SCALE
                        sim_y = START_Y + (DEPTH_MID_M - z_m) * DEPTH_SCALE * TRANS_SCALE
                        sim_z = START_Z + (-y_m) * TRANS_SCALE

            sim_pos = np.array([sim_x, sim_y, sim_z])

            # ---- Calibration ----
            if calibrate_flag:
                raw_angles = robot.compute_raw_angles(lm_l)
                robot.calibrate(raw_angles)
                calibrate_flag = False

            # ---- Teleop step ----
            robot.update(data, lm_l, sim_pos)

            # ---- Data capture (if recording) ----
            logger.record(data.qpos, data.qvel, data.ctrl)

            # ---- Camera HUD ----
            if SHOW_CAMERA:
                tracker.draw_landmarks(frame_l, res_l)
                # Status line
                if logger.is_recording:
                    status = f"REC [{logger.n_steps}]"
                    col = (0, 0, 255)
                elif robot.is_calibrated:
                    status = "READY — press S to record"
                    col = (0, 220, 0)
                else:
                    status = "UNCALIB — press A"
                    col = (0, 165, 255)
                cv2.putText(frame_l, status, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                if frame_r is not None:
                    tracker.draw_landmarks(frame_r, res_r)
                    display = np.hstack([frame_l, frame_r])
                else:
                    display = frame_l
                _show(display)

            # ---- Physics step ----
            for _ in range(phys["n_substeps"]):
                mujoco.mj_step(model, data)
            v.sync()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    # Auto-save if still recording when viewer closes
    if logger.is_recording:
        logger.stop()

    if viewer_proc is not None and frame_q is not None:
        frame_q.put(None)
        viewer_proc.join(timeout=3)
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
