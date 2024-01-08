# qpos (8dof per arm + neck) -> 16 + 3 = 19

# observations
# - images
#   - cam_head (480, 640, 3) 'uint8'
# - qpos       (19,)         'float64'
# - qvel       (19,)         'float64'
# action       (19,)         'float64' -> position of master -> goal_position
# base_action  (2,)          'float64' -> 2d vector

import collections
import time

import cv2
import dm_env
import numpy as np
from reachy_sdk import ReachySDK


class PollenEnv:
    def __init__(self, ip="localhost"):
        self.reachy = ReachySDK(ip)
        self.dt = 1e-5
        self.prev_t = time.time()
        self.qpos = self.get_qpos()
        self.prev_qpos = self.qpos

    def get_qpos(self):
        present_positions = []
        l_arm_present_pos = []
        l_arm_present_pos.append(self.reachy.l_arm.l_shoulder_pitch.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_shoulder_roll.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_arm_yaw.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_elbow_pitch.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_forearm_yaw.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_wrist_pitch.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_wrist_roll.present_position)
        l_arm_present_pos.append(self.reachy.l_arm.l_gripper.present_position)

        r_arm_present_pos = []
        r_arm_present_pos.append(self.reachy.r_arm.r_shoulder_pitch.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_shoulder_roll.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_arm_yaw.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_elbow_pitch.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_forearm_yaw.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_wrist_pitch.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_wrist_roll.present_position)
        r_arm_present_pos.append(self.reachy.r_arm.r_gripper.present_position)

        neck_present_pos = []
        neck_present_pos.append(self.reachy.head.neck_roll.present_position)
        neck_present_pos.append(self.reachy.head.neck_pitch.present_position)
        neck_present_pos.append(self.reachy.head.neck_yaw.present_position)

        present_positions.extend(l_arm_present_pos)
        present_positions.extend(r_arm_present_pos)
        present_positions.extend(neck_present_pos)

        return np.array(present_positions, dtype=np.float64)

    def set_joints(self, left_action, right_action, neck_action):
        self.reachy.l_arm.l_shoulder_pitch.goal_position = left_action[0]
        self.reachy.l_arm.l_shoulder_roll.goal_position = left_action[1]
        self.reachy.l_arm.l_arm_yaw.goal_position = left_action[2]
        self.reachy.l_arm.l_elbow_pitch.goal_position = left_action[3]
        self.reachy.l_arm.l_forearm_yaw.goal_position = left_action[4]
        self.reachy.l_arm.l_wrist_pitch.goal_position = left_action[5]
        self.reachy.l_arm.l_wrist_roll.goal_position = left_action[6]
        self.reachy.l_arm.l_gripper.goal_position = left_action[7]

        self.reachy.r_arm.r_shoulder_pitch.goal_position = right_action[0]
        self.reachy.r_arm.r_shoulder_roll.goal_position = right_action[1]
        self.reachy.r_arm.r_arm_yaw.goal_position = right_action[2]
        self.reachy.r_arm.r_elbow_pitch.goal_position = right_action[3]
        self.reachy.r_arm.r_forearm_yaw.goal_position = right_action[4]
        self.reachy.r_arm.r_wrist_pitch.goal_position = right_action[5]
        self.reachy.r_arm.r_wrist_roll.goal_position = right_action[6]
        self.reachy.r_arm.r_gripper.goal_position = right_action[7]

        # self.reachy.head.neck_roll.goal_position = neck_action[0]
        self.reachy.head.neck_pitch.goal_position = 45
        # self.reachy.head.neck_yaw.goal_position = neck_action[2]

        # self.reachy.head.neck_roll.goal_position = neck_action[0]
        # self.reachy.head.neck_pitch.goal_position = neck_action[1]
        # self.reachy.head.neck_yaw.goal_position = neck_action[2]

    def get_qvel(self, qpos, dt):
        return np.array((qpos - self.prev_qpos) / dt, dtype=np.float64)

    def get_effort(self):
        return np.zeros((19,), dtype=np.float64)

    def get_images(self):
        images = dict()
        images["cam_head"] = cv2.resize(self.reachy.right_camera.last_frame, (640, 480))
        return images

    def get_base_vel(self):
        return np.zeros([0, 0], dtype=np.float64)

    def get_observation(self):
        obs = collections.OrderedDict()
        self.prev_qpos = self.qpos
        self.qpos = self.get_qpos()
        obs["qpos"] = self.qpos
        obs["qvel"] = self.get_qvel(obs["qpos"], self.dt)
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        # obs['base_vel_t265'] = self.get_base_vel_t265()
        obs["base_vel"] = self.get_base_vel()
        return obs

    def get_reward(self):
        return 0

    def step(self, action, base_action=None):
        self.dt = time.time() - self.prev_t

        state_len = int(len(action) / 2)
        left_action = action[:8]
        right_action = action[8:16]
        neck_action = action[16:19]

        # print("step")
        self.set_joints(left_action, right_action, neck_action)

        if base_action is not None:
            pass

        obs = self.get_observation()

        self.prev_t = time.time()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def reset(self):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )


def make_pollen_env(ip="localhost"):
    env = PollenEnv(ip=ip)
    return env
