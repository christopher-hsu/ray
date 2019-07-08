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

import envs
from envs.maps import map_utils
import envs.env_utils as util 
from envs.target_tracking.agent_models import *
from envs.target_tracking.policies import *
from envs.target_tracking.belief_tracker import KFbelief, UKFbelief

import os, copy, pdb, argparse
from envs.target_tracking.metadata import *
from envs.target_tracking.metadata import TARGET_INIT_COV

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v0')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=0)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--nb_train_steps', type=int, default=10000)
parser.add_argument('--buffer_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nb_warmup_steps', type=int, default = 100)
parser.add_argument('--nb_epoch_steps', type=int, default = 500)
parser.add_argument('--target_update_freq', type=int, default=500) # This should be smaller than epoch_steps
parser.add_argument('--nb_test_steps',type=int, default = None)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.99)
parser.add_argument('--learning_rate_growth_factor', type=float, default=1.001)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--hiddens', type=str, default='128:128:128')
parser.add_argument('--log_dir', type=str, default='/TrainDirectories/')
parser.add_argument('--log_fname', type=str, default='model.pkl')
parser.add_argument('--eps_fraction', type=float, default=0.1)
parser.add_argument('--eps_min', type=float, default=.02)
parser.add_argument('--test_eps', type=float, default=.05)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_sd', type=float, default=30.)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--alg', choices=['adfq','adfq-v2'], default='adfq')
parser.add_argument('--act_policy', choices=['egreedy','bayesian'], default='egreedy')
parser.add_argument('--record',type=int, default=0)
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--gpu_memory',type=float, default=1.0)
parser.add_argument('--varth', type=float,default=1e-5)
parser.add_argument('--noise', type=float,default=0.0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--scope',type=str, default='deepadfq')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--ros_log', type=int, default=0)
parser.add_argument('--map', type=str, default="empty")
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--im_size', type=int, default=50)

args = parser.parse_args()


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

    env = envs.make(args.env, 
                    render = bool(args.render), 
                    record = bool(args.record), 
                    ros = bool(args.ros), 
                    # dirname=directory, 
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    im_size=args.im_size,
                    )
    tune.run(
        "DQN",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": env,  # or "corridor" if registered above
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