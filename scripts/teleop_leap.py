# scripts/teleop_leap.py
import yaml
import mujoco.viewer
from src.vision.camera import ZEDCamera
from src.vision.detectors import StereoHandTracker
from src.robots.leap_hand.teleop_interface import LeapTeleopInterface

def main():
    # Load Config
    with open("configs/teleop_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Hardware
    zed = ZEDCamera(camera_id=config['hardware']['camera_id'])
    tracker = StereoHandTracker()

    # Initialize Physics
    model = mujoco.MjModel.from_xml_path(config['physics']['scene_xml'])
    data = mujoco.MjData(model)
    robot = LeapTeleopInterface(model, config)

    def key_callback(keycode):
        if keycode == 65: # 'A' Key
            # Logic to trigger calibration on next update
            pass

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
        while v.is_running():
            # 1. Vision Step
            frame_l, frame_r = zed.get_frames()
            res_l, res_r = tracker.process(frame_l, frame_r)
            
            # 2. Teleop Step
            if res_l.multi_hand_landmarks:
                # Extract landmarks and compute sim_pos
                robot.update(data, res_l.multi_hand_landmarks[0].landmark, sim_pos)

            # 3. Physics Step
            for _ in range(config['physics']['n_substeps']):
                mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    main()