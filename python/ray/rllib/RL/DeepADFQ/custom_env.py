"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.tune import grid_search

from gym import spaces, logger
from gym.utils import seeding

from numpy import linalg as LA

from envs.maps import map_utils
import envs.env_utils as util 
from envs.target_tracking.agent_models import *
from envs.target_tracking.policies import *
from envs.target_tracking.belief_tracker import KFbelief, UKFbelief

import os, copy, pdb
from envs.target_tracking.metadata import *
from envs.target_tracking.metadata import TARGET_INIT_COV

class TargetTrackingEnv0(gym.Env):
    def __init__(self, target_init_cov=TARGET_INIT_COV, q_true = 0.01, q = 0.01, 
                num_targets=1, map_name='empty', is_training=True, known_noise=True):
        gym.Env.__init__(self)
        self.seed()
        self.id = 'TargetTracking-v0'
        self.state = None 
        self.action_space = spaces.Discrete(12)
        self.action_map = {}        
        for (i,v) in enumerate([3,2,1,0]):
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)
        self.target_dim = 2
        self.num_targets = num_targets
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5 # sec
        self.q = q if num_targets==1 else 0.1*q
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(map_path=os.path.join(map_dir_path, map_name), 
                                        r_max = self.sensor_r, fov = self.fov/180.0*np.pi)
        # LIMITS
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = TARGET_INIT_COV
        self.target_noise_cov = self.q * self.sampling_period**3/3*np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = q_true*np.eye(2)
        self.targetA=np.eye(self.target_dim)
        # Build a robot        
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'], 
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
        # Build a target
        self.targets = [AgentDoubleInt2D(dim=self.target_dim, sampling_period=self.sampling_period, 
                            limit=self.limit['target'],
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL),
                            A=self.targetA, W=self.target_true_noise_sd) for _ in range(num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
                                for _ in range(num_targets)]

    def reset(self, init_random = True):
        self.state = []
        if init_random:
            if self.MAP.map is None:
                a_init = self.agent_init_pos[:2]
                self.agent.reset(self.agent_init_pos)
            else:
                isvalid = False 
                while(not isvalid):
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin     
                    isvalid = not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL)) 
                self.agent.reset([a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi])
            for i in range(self.num_targets):
                isvalid = False
                while(not isvalid):            
                    rand_ang = np.random.rand()*2*np.pi - np.pi 
                    t_r = INIT_DISTANCE
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + a_init
                    if (np.sqrt(np.sum((t_init - a_init)**2)) < MARGIN):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init, MARGIN2WALL))
                self.belief_targets[i].reset(init_state=t_init + 10*(np.random.rand(2)-0.5),
                                init_cov=self.target_init_cov)
                self.targets[i].reset(t_init)         
                r, alpha, _ = util.relative_measure(self.belief_targets[i].state, self.agent.state)
                logdetcov = np.log(LA.det(self.belief_targets[i].cov))
                self.state.extend([r, alpha, logdetcov, 0.0])
        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        return self.state

    def observation(self, target):
        r, alpha, _ = util.relative_measure(target.state, self.agent.state) # True Value       
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, self.agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0], #z[0]/self.sensor_r * self.sensor_r_sd, 0.0], 
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov
    
    def get_reward(self, obstacles_pt, observed, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = 1./max(1.0, obstacles_pt[0]**2)

        if sum(observed) == 0:
            reward = - penalty
        else:
            detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
            reward = - 0.1 * np.log(np.mean(detcov) + np.std(detcov)) - penalty
            reward = max(0.0, reward) + np.mean(observed)
        test_reward = None

        if not(is_training):
            logdetcov = [np.log(LA.det(b_target.cov)) for b_target in self.belief_targets]              
            test_reward = -np.mean(logdetcov)

        return reward, False, test_reward

    def step(self, action):
        action_val = self.action_map[action]
        boundary_penalty = self.agent.update(action_val, [t.state[:2] for t in self.targets])
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        observed = []
        for i in range(self.num_targets):
            self.targets[i].update(self.agent.state[:2])      
            # Observe
            obs = self.observation(self.targets[i])
            observed.append(obs[0])
            # Update the belief of the agent on the target using KF
            self.belief_targets[i].update(obs[0], obs[1], self.agent.state)

        reward, done, test_reward = self.get_reward(obstacles_pt, observed, self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.relative_measure(self.belief_targets[i].state, self.agent.state)
            self.state.extend([r_b, alpha_b, 
                                    np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'test_reward': test_reward}


class CustomModel(Model):
    """Example of a custom model.

    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        return self.fcnet.outputs, self.fcnet.last_layer


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    register_env(
        "TargetTrackingEnv0", lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))
    tune.run(
        "DQN",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": TargetTrackingEnv0,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "lr": grid_search([1e-2, 1e-3, 1e-4]),  # try different lrs
            "num_workers": 1,  # parallelism
            "timesteps_per_iteration": 1000,
            
            "env_config": {
                # "corridor_length": 5,
            },
        },
    )
