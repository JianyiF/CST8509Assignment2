import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from irobot_create_msgs.msg import StopStatus

from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time


# ==============================
# ROS2 NODE
# ==============================
class RedBall(Node):

    def __init__(self):
        super().__init__('redball_node')

        self.bridge = CvBridge()

        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.image_publisher = self.create_publisher(
            Image, '/target_redball', 10
        )

        self.create_subscription(
            Image,
            '/custom_ns/camera1/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            StopStatus,
            '/stop_status',
            self.stop_callback,
            10
        )

        self.redball_position = 320
        self.create3_is_stopped = True

    # IMAGE CALLBACK
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)

            if radius > 5:
                self.redball_position = int(x)
        else:
            self.redball_position = 320

    # STOP CALLBACK
    def stop_callback(self, msg):
        self.create3_is_stopped = msg.is_stopped

    # ACTION
    def send_action(self, action):
        angle = (action - 320) / 320 * (math.pi / 2)

        twist = Twist()
        twist.angular.z = angle

        self.cmd_vel_publisher.publish(twist)

        self.create3_is_stopped = False


# ==============================
# GYM ENV
# ==============================
class CreateRedBallEnv(gym.Env):

    def __init__(self):
        super().__init__()

        rclpy.init()
        self.redball = RedBall()

        observation = int(self.redball.redball_position)

        self.action_space = spaces.Discrete(641)
        self.observation_space = spaces.Discrete(641)

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        for _ in range(5):
            rclpy.spin_once(self.redball)
        
        obs = int(self.redball.redball_position)

        return obs, {}

    def step(self, action):

        self.redball.send_action(action)

        # process ROS once
        rclpy.spin_once(self.redball, timeout_sec=0.05)

        # IMPORTANT: no infinite loop (SB3 safe)
        timeout = time.time() + 1.0
        while not self.redball.create3_is_stopped and time.time() < timeout:
            rclpy.spin_once(self.redball, timeout_sec=0.05)

        obs = int(self.redball.redball_position)

        reward = -abs(obs - 320)

        self.step_count += 1
        done = self.step_count >= 100

        return obs, float(reward), done, False, {"pos": obs}

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()