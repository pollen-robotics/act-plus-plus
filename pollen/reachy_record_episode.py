import time
from glob import glob

import cv2
import h5py
import numpy as np
from reachy_sdk import ReachySDK

# reachy

# qpos (8dof per arm + neck) -> 16 + 3 = 19

# observations
# - images
#   - cam_head (480, 640, 3) 'uint8'
# - qpos       (19,)         'float64'
# - qvel       (19,)         'float64'

# action       (19,)         'float64' -> position of master -> goal_position
# base_action  (2,)          'float64' -> 2d vector


# returns goal_positions, present_positions
def get_goal_positions():
    goal_positions = []
    l_arm_goal_pos = []
    l_arm_goal_pos.append(reachy.l_arm.l_shoulder_pitch.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_shoulder_roll.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_arm_yaw.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_elbow_pitch.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_forearm_yaw.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_wrist_pitch.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_wrist_roll.goal_position)
    l_arm_goal_pos.append(reachy.l_arm.l_gripper.goal_position)

    r_arm_goal_pos = []
    r_arm_goal_pos.append(reachy.r_arm.r_shoulder_pitch.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_shoulder_roll.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_arm_yaw.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_elbow_pitch.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_forearm_yaw.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_wrist_pitch.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_wrist_roll.goal_position)
    r_arm_goal_pos.append(reachy.r_arm.r_gripper.goal_position)

    neck_goal_pos = []
    neck_goal_pos.append(reachy.head.neck_roll.goal_position)
    neck_goal_pos.append(reachy.head.neck_pitch.goal_position)
    neck_goal_pos.append(reachy.head.neck_yaw.goal_position)

    goal_positions.extend(l_arm_goal_pos)
    goal_positions.extend(r_arm_goal_pos)
    goal_positions.extend(neck_goal_pos)

    return np.array(goal_positions, dtype=np.float64)


def get_present_positions():
    present_positions = []
    l_arm_present_pos = []
    l_arm_present_pos.append(reachy.l_arm.l_shoulder_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_shoulder_roll.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_arm_yaw.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_elbow_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_forearm_yaw.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_wrist_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_wrist_roll.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_gripper.present_position)

    r_arm_present_pos = []
    r_arm_present_pos.append(reachy.r_arm.r_shoulder_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_shoulder_roll.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_arm_yaw.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_elbow_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_forearm_yaw.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_wrist_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_wrist_roll.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_gripper.present_position)

    neck_present_pos = []
    neck_present_pos.append(reachy.head.neck_roll.present_position)
    neck_present_pos.append(reachy.head.neck_pitch.present_position)
    neck_present_pos.append(reachy.head.neck_yaw.present_position)

    present_positions.extend(l_arm_present_pos)
    present_positions.extend(r_arm_present_pos)
    present_positions.extend(neck_present_pos)

    return np.array(present_positions, dtype=np.float64)


def get_qvel(prev_qpos, qpos, dt):
    return np.array((qpos - prev_qpos) / dt, dtype=np.float64)


record_for = 10  # seconds
# reachy = ReachySDK("localhost")
reachy = ReachySDK("192.168.1.162")


prev_qpos = get_present_positions()

data_dict = {
    "/observations/qpos": [],
    "/observations/qvel": [],
    "/observations/effort": [],
    "/observations/images/cam_head": [],
    "/action": [],
    "/base_action": [],
}

sampling_rate = 50  # Hz
start = time.time()
elapsed = time.time() - start
dt = 0.0
prev_t = start
print("Recording ...")
while elapsed < record_for:
    dt = time.time() - prev_t

    images = {}
    images["cam_head"] = cv2.resize(reachy.right_camera.last_frame, (640, 480))

    qpos = get_present_positions()
    prev_qpos = qpos

    # action = get_goal_positions() # TODO goal_positions are not updated.
    action = get_present_positions()  # For now, we use present positions as action.

    base_action = np.array([0, 0], np.float64)

    data_dict["/action"].append(action)
    data_dict["/base_action"].append(base_action)
    data_dict["/observations/qpos"].append(qpos)
    data_dict["/observations/qvel"].append(get_qvel(prev_qpos, qpos, dt))
    data_dict["/observations/effort"].append(np.zeros((19,), dtype=np.float64))
    data_dict["/observations/images/cam_head"].append(images["cam_head"])

    prev_t = time.time()
    time.sleep(1 / sampling_rate)
    elapsed = time.time() - start

print("Done recording")
max_timesteps = len(data_dict["/action"])

nb_files = len(glob("episodes/pollen_grab_bottle/*.hdf5"))

t0 = time.time()
with h5py.File(
    "episodes/pollen_grab_bottle/" + str(nb_files) + ".hdf5",
    "w",
    rdcc_nbytes=1024**2 * 2,
) as root:
    root.attrs["sim"] = False
    root.attrs["compress"] = False
    obs = root.create_group("observations")
    image = obs.create_group("images")
    for cam_name in ["cam_head"]:
        _ = image.create_dataset(
            cam_name,
            (max_timesteps, 480, 640, 3),
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )
    _ = obs.create_dataset("qpos", (max_timesteps, 19))
    _ = obs.create_dataset("qvel", (max_timesteps, 19))
    _ = obs.create_dataset("effort", (max_timesteps, 19))
    _ = root.create_dataset("action", (max_timesteps, 19))
    _ = root.create_dataset("base_action", (max_timesteps, 2))
    # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

    for name, array in data_dict.items():
        root[name][...] = array
time.sleep(2)
print("Saved in episodes/pollen_grab_bottle/" + str(nb_files) + ".hdf5")
