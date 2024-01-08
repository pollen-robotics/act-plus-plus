import time

import cv2
import numpy as np
from reachy_sdk import ReachySDK


class ReachyWrapper:
    def __init__(self, ip):
        self.reachy = ReachySDK(ip)
        self.l_gripper_closed = False
        self.start_l_gripper_closed = time.time()


    # Returns for how many seconds the left gripper has been closed
    def l_gripper_closed_for(self):
        was_closed = self.l_gripper_closed
        if self.reachy.l_arm.l_gripper.present_position < 0:
            self.l_gripper_closed = True
        else:
            self.l_gripper_closed = False

        if not was_closed and self.l_gripper_closed: # Was open and is closed now
            self.start_l_gripper_closed = time.time()
        
        if not self.l_gripper_closed:
            self.start_l_gripper_closed = time.time()

        return time.time() - self.start_l_gripper_closed

    def get_image(self):
        return cv2.resize(self.reachy.right_camera.last_frame, (640, 480))

    def get_present_positions(self):
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
    

    def get_qvel(self, prev_qpos, qpos, dt):
        return np.array((qpos - prev_qpos) / dt, dtype=np.float64)
    
    def get_goal_positions(self):
        return None
        # TODO not working, goal_position is not updated. Not used for now
        goal_positions = []
        l_arm_goal_pos = []
        l_arm_goal_pos.append(self.reachy.l_arm.l_shoulder_pitch.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_shoulder_roll.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_arm_yaw.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_elbow_pitch.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_forearm_yaw.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_wrist_pitch.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_wrist_roll.goal_position)
        l_arm_goal_pos.append(self.reachy.l_arm.l_gripper.goal_position)

        r_arm_goal_pos = []
        r_arm_goal_pos.append(self.reachy.r_arm.r_shoulder_pitch.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_shoulder_roll.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_arm_yaw.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_elbow_pitch.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_forearm_yaw.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_wrist_pitch.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_wrist_roll.goal_position)
        r_arm_goal_pos.append(self.reachy.r_arm.r_gripper.goal_position)

        neck_goal_pos = []
        neck_goal_pos.append(self.reachy.head.neck_roll.goal_position)
        neck_goal_pos.append(self.reachy.head.neck_pitch.goal_position)
        neck_goal_pos.append(self.reachy.head.neck_yaw.goal_position)

        goal_positions.extend(l_arm_goal_pos)
        goal_positions.extend(r_arm_goal_pos)
        goal_positions.extend(neck_goal_pos)

        return np.array(goal_positions, dtype=np.float64)
    

# if __name__ == "__main__":
#     r = ReachyWrapper("192.168.1.162")
#     while True:
#         if r.l_gripper_closed_for() > 2.:
#             print("CLOSED FOR 2 SECS")
#         if r.l_gripper_closed_for() > 2.:
#             print("CLOSED FOR 2 SECS")
