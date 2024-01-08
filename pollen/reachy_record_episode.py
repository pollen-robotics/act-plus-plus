import argparse
import os
import time
from glob import glob

import cv2
import h5py
import numpy as np
from reachy_utils import ReachyWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, required=False, default="192.168.1.252", help="Ip of the robot")
parser.add_argument("--taskName", type=str, required=True, help="Name of the task")
parser.add_argument("--saveDir", type=str, required=True, help="Where to save the episodes")
parser.add_argument("--nbEpisodesToRecord", type=int, required=True, help="How many episodes to record in this session")
parser.add_argument("--episodeLength", type=int, required=True, help="How many seconds per episode")
args = parser.parse_args()

SAMPLING_RATE = 30 # Hz

path = os.path.join(args.saveDir, args.taskName)
os.makedirs(path, exist_ok=True)

# reachy

# qpos (8dof per arm + neck) -> 16 + 3 = 19

# observations
# - images
#   - cam_head (480, 640, 3) 'uint8'
# - qpos       (19,)         'float64'
# - qvel       (19,)         'float64'

# action       (19,)         'float64' -> position of master -> goal_position
# base_action  (2,)          'float64' -> 2d vector

def play_sound(duration, freq):
    os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))


nb_episodes_recorded = 0
reachy_wrapper = ReachyWrapper(args.ip)
STATE = "IDLE" # Can be IDLE or RECORDING
play_sound(0.5, 600) # Setup ready sound

print("IDLE")
while True:
    if STATE == "IDLE":
        if reachy_wrapper.l_gripper_closed_for() > 2.:
            STATE = "REC"

    if STATE == "REC":
        print("Starting to record")
        data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/observations/effort": [],
            "/observations/images/cam_head": [],
            "/action": [],
            "/base_action": [],
        }

        time.sleep(1)
        dt = 0
        prev_t = time.time()
        start = time.time()
        play_sound(1.0, 440) # Start recording sound
        prev_present_position = reachy_wrapper.get_present_positions()
        elapsed = 0
        while elapsed < args.episodeLength:
            dt = time.time() - prev_t
            elapsed = time.time() - start

            present_position = reachy_wrapper.get_present_positions()
            data_dict["/action"].append(present_position) # For now present_position = goal_position
            data_dict["/base_action"].append(np.array([0, 0], np.float64)) # We don't record mobile base action for now
            data_dict["/observations/qpos"].append(present_position)
            data_dict["/observations/qvel"].append(reachy_wrapper.get_qvel(prev_present_position, present_position, dt))
            data_dict["/observations/effort"].append(np.zeros((19,), dtype=np.float64))
            data_dict["/observations/images/cam_head"].append(reachy_wrapper.get_image())

            prev_present_position = present_position

            prev_t = time.time()
            time.sleep(1 / SAMPLING_RATE)

        nb_episodes = len(glob(os.path.join(path, "*.hdf5")))
        episode_path = os.path.join(path, str(nb_episodes)+".hdf5")
        max_timesteps = len(data_dict["/action"])

        with h5py.File(episode_path,"w",rdcc_nbytes=1024**2 * 2,) as root:
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

            for name, array in data_dict.items():
                root[name][...] = array

        play_sound(0.5, 600) # Episode done
        print("Saved episode: ", episode_path)
        nb_episodes_recorded += 1

        if nb_episodes_recorded > args.nbEpisodesToRecord:
            break
        STATE = "IDLE"
        print("IDLE")

print("SESSION DONE")
play_sound(0.2, 600)
play_sound(0.2, 600)
play_sound(0.2, 600)
