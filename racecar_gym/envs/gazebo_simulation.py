import rospy
import numpy as np

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Quaternion


def create_model_state(robot_name, x, y, z, angle):
    # the rotation of the angle is in (0, 0, 1) direction
    model_state = ModelState()
    model_state.model_name = robot_name
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    model_state.pose.orientation = Quaternion(0, 0, np.sin(angle / 2.), np.cos(angle / 2.))
    model_state.reference_frame = "world"

    return model_state


class GazeboSimulation():

    def __init__(self):
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

    def unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

    def reset(self, robot_name, x, y, angle):
        """
        /gazebo/reset_world or /gazebo/reset_simulation will
        destroy the world setting, here we used set model state
        to put the model back to the origin
        """
        assert robot_name is not None, "Please provide the robot name"

        init_model_state = create_model_state(robot_name, x, y, 0, angle)

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._reset(init_model_state)
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/set_model_state service call failed")

    def get_model_state(self, robot_name):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            return self._model_state(robot_name, 'world')
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/get_model_state service call failed")