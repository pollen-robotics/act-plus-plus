import time

import numpy as np
from reachy_sdk import ReachySDK

reachy = ReachySDK("localhost")

reachy.head.neck_pitch.goal_position = 45

reachy.r_arm.r_elbow_pitch.goal_position = -90
reachy.l_arm.l_elbow_pitch.goal_position = -90
pos = 0
while True:
    p = pos + 15 * np.sin(2 * np.pi * 0.5 * time.time())
    reachy.r_arm.r_arm_yaw.goal_position = p
    time.sleep(0.001)
