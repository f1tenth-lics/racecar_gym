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

# gym imports
import gym
from gym import spaces

# ros
import rospy
import rospkg
import subprocess
from .gazebo_simulation import GazeboSimulation
from .racecar import RaceCar
from .utils import quaternion_to_psi

# others
import os
import time
import yaml
import numpy as np


class RaceCarEnv(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        seed (int, default=12345): seed for random state and reproducibility
        map (str, default='vegas'): name of the map used for the environment.
                                    Currently, available environments include: 'berlin', 'vegas', 'skirk'.
                                    You can also use the absolute path to the yaml file of your custom map.
        params (dict, default=None): dictionary of vehicle parameters.
        num_agents (int, default=1): number of agents in the environment.
        timestep (float, default=0.01): physics timestep.
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    def __init__(self, seed=12345, map='berlin', params=None, num_agents=1, timestep=0.01):
        # Initialize parameters
        if params is None:
            params = {
                'max_speed': 20.0,
                'max_steering_angle': 0.325,
                'laser_clip': 10.0,
            }

        rospack = rospkg.RosPack()
        helper_path = rospack.get_path('racecar_helper')

        self.seed = seed
        self.map = map
        self.params = params
        self.num_agents = num_agents
        self.timestep = timestep

        self.map_path = os.path.join(helper_path, 'maps', self.map, self.map + '.yaml')

        # Read map yaml file
        with open(self.map_path, 'r') as f:
            map_data = yaml.load(f, Loader=yaml.FullLoader)

        self.current_time = 0.0

        self.start_xs = np.arange(self.num_agents)
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))

        # Start simulation processes
        self.sim_process = subprocess.Popen([
            'roslaunch',
            os.path.join(helper_path, 'launch', 'simulator.launch'),
            'map:=' + self.map,
            'gui:=true',
            'teleop:=false',
        ])
        time.sleep(1)

        colors = ['blue', 'red', 'green']
        self.racecar_processes = []
        for i in range(self.num_agents):
            self.racecar_processes.append(subprocess.Popen([
                'roslaunch',
                os.path.join(helper_path, 'launch', 'spawn_racecar.launch'),
                'robot_name:=racecar' + str(i),
                'color:=' + colors[i % 3],
                'init_x:=' + str(self.start_xs[i]),
                'init_y:=' + str(self.start_ys[i])
            ]))
        time.sleep(5)

        rospy.init_node('f1tenth_gym', anonymous=True)

        self.sim = GazeboSimulation()
        self._reset_sim()

        self.racecars = []
        for i in range(self.num_agents):
            self.racecars.append(RaceCar('racecar' + str(i), **self.params))

        self.rate = rospy.Rate(1 / self.timestep)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, 2), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Discrete(1),
            'scans': spaces.Box(low=0, high=10, shape=(1080,), dtype=np.float32),
            'poses_x': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'poses_y': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'poses_theta': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'linear_vels_x': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'linear_vels_y': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'ang_vels_z': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32),
            'collisions': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)
        })

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _get_obs(self):
        ego_idx = []
        scans = []
        poses_x = []
        poses_y = []
        poses_theta = []
        linear_vels_x = []
        linear_vels_y = []
        ang_vels_z = []
        collisions = []

        for i in range(self.num_agents):
            ego_idx.append(i)
            scans.append(self.racecars[i].laser)

            odom = self.racecars[i].odom
            poses_x.append(odom[0])
            poses_y.append(odom[1])
            poses_theta.append(odom[2])
            linear_vels_x.append(odom[3])
            linear_vels_y.append(odom[4])
            ang_vels_z.append(0.0)

            collisions.append(self.racecars[i].collision)

        return {
            'ego_idx': ego_idx,
            'scans': scans,
            'poses_x': poses_x,
            'poses_y': poses_y,
            'poses_theta': poses_theta,
            'linear_vels_x': linear_vels_x,
            'linear_vels_y': linear_vels_y,
            'ang_vels_z': ang_vels_z,
            'collisions': collisions
        }

    def _check_done(self):
        """
        Check if the current rollout is done

        Returns:
            done (bool): whether the rollout is done
        """
        # Check for collisions
        for i in range(self.num_agents):
            if self.racecars[i].collision:
                return True
        return False

    def _get_reward(self, state, obs):
        reward = 0.0
        for i in range(self.num_agents):
            reward += obs['linear_vels_x'][i]
        return self.timestep

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxiliary information dictionary
        """
        # Call simulation step
        for i in range(self.num_agents):
            self.racecars[i].set_velocity(action[i][0], action[i][1])

        self.rate.sleep()

        # Get observations
        state = self._get_states()
        obs = self._get_obs()

        # Calculate reward
        reward = self._get_reward(state, obs)
        self.current_time += 1

        # Check if done
        done = self._check_done()

        return obs, reward, done, {}

    def _get_states(self):
        """
        Get the state of the robot

        Returns:
            state (dict): state of the robot
        """
        ego_idx = []
        poses_x = []
        poses_y = []
        poses_theta = []
        linear_vels_x = []
        linear_vels_y = []

        for i in range(self.num_agents):
            state = self.sim.get_model_state('racecar' + str(i))
            ego_idx.append(i)
            poses_x.append(state.pose.position.x)
            poses_y.append(state.pose.position.y)
            poses_theta.append(quaternion_to_psi(state.pose.orientation))
            linear_vels_x.append(state.twist.linear.x)
            linear_vels_y.append(state.twist.linear.y)

        return {
            'ego_idx': ego_idx,
            'poses_x': poses_x,
            'poses_y': poses_y,
            'poses_theta': poses_theta,
            'linear_vels_x': linear_vels_x,
            'linear_vels_y': linear_vels_y
        }

    def _reset_sim(self):
        for i in range(self.num_agents):
            self.sim.reset(f'racecar{i}', self.start_xs[i], self.start_ys[i], self.start_thetas[i])

    def _stop_racecar(self):
        for i in range(self.num_agents):
            self.racecars[i].set_velocity(0, 0)

    def reset(self, poses=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
        """
        if poses is not None:
            self.start_xs = poses[:, 0]
            self.start_ys = poses[:, 1]
            self.start_thetas = poses[:, 2]

        # Stop racecars
        self._stop_racecar()
        time.sleep(1)

        # Call reset to simulator
        self._reset_sim()

        for i in range(self.num_agents):
            self.racecars[i].collision = False

        # Get observations
        obs = self._get_obs()

        return obs
