# MIT License

# Copyright (c) 2024 Joshua J. Damanik

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Joshua J. Damanik
"""

import rospy
import numpy as np

from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import LaserScan
from robot_localization.srv import SetPose
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from .utils import quaternion_to_psi


class RaceCar:
    """
    A class to represent a race car in the F1TENTH environment.

    Attributes:
        robot_name (str): The name of the robot.
        max_speed (float): The maximum speed of the car.
        max_steering_angle (float): The maximum steering angle of the car.
        laser_clip (float): The maximum distance for the laser scanner readings.
        odom (np.ndarray): The odometry data [x, y, psi, linear_velocity, angular_velocity].
        laser (np.ndarray): The laser scan data.
        collision (bool): Whether the car has collided.
        odom_topic (str): The ROS topic for odometry.
        laser_topic (str): The ROS topic for laser scans.
        collision_topic (str): The ROS topic for collision detection.
        command_topic (str): The ROS topic for sending drive commands.

    Methods:
        set_velocity(speed, steering_angle): Sets the velocity of the car.
    """

    def __init__(
            self,
            robot_name='racecar',
            max_speed=20,
            max_steering_angle=0.325,
            laser_clip=10.0,
            odom_topic='odometry/filtered',
            laser_topic='scan',
            collision_topic='collision',
            command_topic='vesc/high_level/ackermann_cmd_mux/input/nav_0',
    ):
        """
        Initializes the RaceCar object with the given parameters.

        Args:
            robot_name (str, default='racecar'): The name of the robot.
            max_speed (float, default=20): The maximum speed of the car.
            max_steering_angle (float, default=0.325): The maximum steering angle of the car.
            laser_clip (float, default=10.0): The maximum distance for the laser scanner readings.
            odom_topic (str, default='odometry/filtered'): The ROS topic for odometry.
            laser_topic (str, default='scan'): The ROS topic for laser scans.
            collision_topic (str, default='collision'): The ROS topic for collision detection.
            command_topic (str, default='vesc/high_level/ackermann_cmd_mux/input/nav_0'): The ROS topic for sending drive commands.
        """
        self.robot_name = robot_name
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.laser_clip = laser_clip

        self.odom = np.zeros((5,), dtype=np.float32)
        self.laser = np.full((1080,), laser_clip, dtype=np.float32)
        self.collision = False

        self.odom_topic = f'/{robot_name}/{odom_topic}'
        self.laser_topic = f'/{robot_name}/{laser_topic}'
        self.collision_topic = f'/{robot_name}/{collision_topic}'
        self.command_topic = f'/{robot_name}/{command_topic}'

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback)
        rospy.Subscriber(self.laser_topic, LaserScan, self._laser_callback)
        rospy.Subscriber(self.collision_topic, Bool, self._collision_callback)
        self.command_pub = rospy.Publisher(self.command_topic, AckermannDriveStamped, queue_size=1)
        self.reset_odom_srv = rospy.ServiceProxy(f'/{robot_name}/set_pose', SetPose)

    def _odom_callback(self, msg):
        """
        Callback function for odometry messages.

        Args:
            msg (Odometry): The odometry message.
        """
        pose = msg.pose.pose
        twist = msg.twist.twist
        psi = quaternion_to_psi(pose.orientation)
        self.odom = (pose.position.x, pose.position.y, psi, twist.linear.x, twist.angular.z)

    def _laser_callback(self, msg):
        """
        Callback function for laser scan messages.

        Args:
            msg (LaserScan): The laser scan message.
        """
        self.laser = np.array(msg.ranges, dtype=np.float32)
        self.laser[self.laser > self.laser_clip] = self.laser_clip

    def _collision_callback(self, msg):
        """
        Callback function for collision messages.

        Args:
            msg (Bool): The collision message.
        """
        self.collision = msg.data

    def set_velocity(self, speed, steering_angle):
        """
        Sets the velocity of the car.

        Args:
            speed (float): The desired speed as a fraction of the maximum speed [-1, 1].
            steering_angle (float): The desired steering angle as a fraction of the maximum steering angle [-1, 1].
        """
        command = AckermannDriveStamped()
        command.header = Header()
        command.drive.speed = speed * self.max_speed
        command.drive.steering_angle = steering_angle * self.max_steering_angle
        self.command_pub.publish(command)

    def reset_odom(self, x=0.0, y=0.0, angle=0.0):
        """
        Resets the odometry of the car.

        Args:
            x (float): The x-coordinate of the car.
            y (float): The y-coordinate of the car.
            angle (float): The orientation of the car in radians.
        """
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = self.robot_name + '/odom'
        pose.pose.pose.position.x = x
        pose.pose.pose.position.y = y
        pose.pose.pose.position.z = 0
        pose.pose.pose.orientation.x = 0
        pose.pose.pose.orientation.y = 0
        pose.pose.pose.orientation.z = np.sin(angle / 2)
        pose.pose.pose.orientation.w = np.cos(angle / 2)

        self.reset_odom_srv(pose)
