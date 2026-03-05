# src/robots/leap_hand/teleop_interface.py
import numpy as np
import mujoco
from src.utils.smoother import OneEuroFilter
from src.robots.leap_hand.retargeter import IKRetargeter

class LeapTeleopInterface:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.mid = model.body("hand_proxy").mocapid[0]
        self.ik = IKRetargeter(model)
        
        # Initialize Filters
        self.filters = {
            "pos": OneEuroFilter(**config['filters']['position']),
            "joint": OneEuroFilter(**config['filters']['joint']),
            "wrist": OneEuroFilter(**config['filters']['wrist']),
            # ... add others
        }
        
        self.ref_angles = {"roll": None, "pitch": None, "yaw": None}
        self.base_quat = np.array([0.0, 1.0, 0.0, 0.0])

    def calibrate(self, raw_angles):
        """Captures the current hand orientation as the zero-reference."""
        self.ref_angles = raw_angles
        for f in self.filters.values():
            f.reset()
        print("[CALIB] Reference orientation captured.")

    def update(self, data, hand_landmarks, sim_pos):
        """Processes one frame of tracking data and updates MuJoCo control."""
        if self.ref_angles["roll"] is None:
            return

        # 1. Update Position
        new_pos = self.filters["pos"](sim_pos)
        data.mocap_pos[self.mid] = self._apply_step_limit(data.mocap_pos[self.mid], new_pos)

        # 2. Update Orientation (logic moved from teleop_leap.py)
        # ... compute wrist_x, wrist_y, wrist_z using self.ref_angles ...
        # data.mocap_quat[self.mid] = computed_quat

        # 3. Update Fingers
        q_raw = self.ik.retarget(None, hand_landmarks)
        data.ctrl[:] = self.filters["joint"](q_raw)

    def _apply_step_limit(self, current, target):
        delta = target - current
        dist = np.linalg.norm(delta)
        limit = self.config['wrist_control']['max_step_m']
        if dist > limit:
            return current + delta * (limit / dist)
        return target