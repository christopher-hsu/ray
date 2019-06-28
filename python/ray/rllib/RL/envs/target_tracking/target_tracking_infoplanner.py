import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np 
from numpy import linalg as LA
import os, pdb

from envs.maps import map_utils 
from envs.target_tracking.agent_models import AgentSE2, AgentDoubleInt2D, Agent_InfoPlanner
import envs.env_utils as util 
from envs.target_tracking.belief_tracker import KFbelief

# import pyInfoGathering as IGL
from envs.target_tracking.infoplanner_binding import Configure, Policy
from envs.target_tracking.metadata import *

class BeliefWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.num_targets = num_targets
        self.dim = dim
        self.state = None
        self.cov = None

    def update(self, state, cov):
        self.state = np.reshape(state, (self.num_targets, self.dim))
        self.cov = [cov[n*self.dim: (n+1)*self.dim,n*self.dim: (n+1)*self.dim ] for n in range(self.num_targets)]

class TargetWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.state = None
        self.num_targets = num_targets
        self.dim = dim

    def reset(self, target):
        self.target = target
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

    def update(self):
        self.target.forwardSimulate(1) 
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

class TargetTrackingInfoPlanner1(gym.Env):
    """
    Double Integrator 
    """
    def __init__(self, target_init_cov=TARGET_INIT_COV, q_true = 0.01, q = 0.001, target_init_vel=0.0,
                    map_name='empty', is_training=True, known_noise=True, num_targets=1):
        gym.Env.__init__(self)
        self.seed()
        self.id = 'TargetTracking-info1'
        self.state = None 
        self.action_space = spaces.Discrete(12)
        self.action_map = {}  
        for (i,v) in enumerate([3,2,1,0]):      
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)
        self.target_dim = 4
        self.viewer = None
        self.is_training = is_training

        self.num_targets = num_targets
        tau = 0.5 # sec
        self.q = q
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(map_path=os.path.join(map_dir_path,map_name), 
                                        r_max = self.sensor_r, fov = self.fov/180.0*np.pi)
        # LIMITS for GYM
        self.vel_limit = np.array([2.0, 2.0])
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate(( self.MAP.mapmin,[-np.pi])), 
                                            np.concatenate(( self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,  -self.vel_limit)), 
                                            np.concatenate((self.MAP.mapmax,  self.vel_limit))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -self.vel_limit[0], -self.vel_limit[1], -50.0, 0.0]*num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, self.vel_limit[0], self.vel_limit[1], 50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.concatenate((self.MAP.origin, [0.0]))
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = target_init_cov
        self.target_init_vel = target_init_vel*np.ones((2,))
        self.target_noise_sd = self.q * np.concatenate((
                        np.concatenate((tau**3/3*np.eye(2), tau**2/2*np.eye(2)), axis=1),
                        np.concatenate((tau**2/2*np.eye(2), tau*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_sd
        else:
            self.target_true_noise_sd = q_true * np.concatenate((
                        np.concatenate((tau**2/2*np.eye(2), tau/2*np.eye(2)), axis=1),
                        np.concatenate((tau/2*np.eye(2), tau*np.eye(2)),axis=1) ))
        self.targetA = np.concatenate((np.concatenate((np.eye(2), tau*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))

        # Setup Ground Truth Target Simulation 
        map_nd = IGL.map_nd(self.MAP.mapmin, self.MAP.mapmax, self.MAP.res)
        if self.MAP.map is None:
            cmap_data = list(map(str, [0] * map_nd.size()[0] * map_nd.size()[1]))
        else:
            cmap_data = list(map(str, np.squeeze(self.MAP.map.astype(np.int8).reshape(-1, 1)).tolist()))
        se2_env = IGL.SE2Environment(map_nd, cmap_data, os.path.join(map_dir_path,'mprim_SE2_RL.yaml'))

        self.cfg = Configure(map_nd, cmap_data)
        sensor = IGL.RangeBearingSensor(self.sensor_r, self.fov, self.sensor_r_sd, self.sensor_b_sd, map_nd, cmap_data)
        self.agent = Agent_InfoPlanner(dim=3, sampling_period=tau, limit=self.limit['agent'], 
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL), 
                            se2_env=se2_env, sensor_obj=sensor)
        self.belief_targets = BeliefWrapper(num_targets)
        self.targets = TargetWrapper(num_targets)

    def reset(self, init_random=True):
        self.state = []
        t_init_sets = []
        t_init_b_sets = []
        if init_random:
            if self.MAP.map is None:
                a_init = self.agent_init_pos
                a_init_igl = IGL.SE3Pose(self.agent_init_pos, np.array([0, 0, 0, 1]))
            else:
                isvalid = False
                while(not isvalid):
                    a_init = np.random.random((2,))*((self.MAP.mapmax-MARGIN)-self.MAP.mapmin) + self.MAP.mapmin - .5 * MARGIN    
                    isvalid = not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL)) 
                a_init = np.concatenate((a_init, [np.random.random() * 2 * np.pi - np.pi]))
                a_init_igl = IGL.SE3Pose(a_init, np.array([0, 0, 0, 1]))

            for i in range(self.num_targets):
                isvalid = False
                while(not isvalid):            
                    rand_ang = np.random.rand()*2*np.pi - np.pi 
                    t_r = INIT_DISTANCE
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + a_init[:2]
                    if (np.sqrt(np.sum((t_init - a_init[:2])**2)) < MARGIN):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init, MARGIN2WALL))
                t_init_b_sets.append(t_init + 10*(np.random.rand(2)-0.5))
                t_init_sets.append(t_init)         
                r, alpha, _ = util.relative_measure(t_init_b_sets[-1][:2], a_init)
                logdetcov = np.log(LA.det(self.target_init_cov*np.eye(self.target_dim)))
                self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])

        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        # Build a target
        target = self.cfg.setup_integrator_targets(n_targets=self.num_targets, init_pos=t_init_sets,
                                                init_vel=self.target_init_vel, q=self.q, max_vel=self.vel_limit[0])  # Integrator Ground truth Model
        belief_target = self.cfg.setup_integrator_belief(n_targets=self.num_targets, q=self.q, 
                                                init_pos=t_init_b_sets, 
                                                cov_pos=self.target_init_cov, cov_vel=self.target_init_cov, 
                                                init_vel=(0.0, 0.0)) 
         # Build a robot 
        self.agent.reset(a_init_igl, belief_target) 
        self.targets.reset(target)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())
        return np.array(self.state)
    
    def get_reward(self, obstacles_pt, observed, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = 1./max(1.0, obstacles_pt[0]**2)

        if sum(observed) == 0:
            reward = - penalty
        else:
            cov = self.agent.get_belief_cov()
            detcov = [LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)]) for n in range(self.num_targets)] 
            #logdetcov = [np.log(LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])) for n in range(self.num_targets)]                
            #reward_past = -0.1*np.mean(logdetcov) - penalty - 0.1*np.std(logdetcov) #- 0.01*np.square(np.std(logdetcov))) 
            reward = - 0.1 * np.log(np.mean(detcov) + np.std(detcov))
            reward = max(0.0, reward) + np.mean(observed)
        test_reward = None

        if not(is_training):
            cov = self.agent.get_belief_cov()
            logdetcov = [np.log(LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])) for n in range(self.num_targets)]                
            test_reward = -np.mean(logdetcov)

        reward /= 5.0
        return reward, False, test_reward

    def step(self, action):
        self.agent.update(action, self.targets.state) 

        # Update the true target state
        self.targets.update()
        # Observe
        measurements = self.agent.observation(self.targets.target)
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        # Update the belief of the agent on the target using KF
        GaussianBelief = IGL.MultiTargetFilter(measurements, self.agent.agent, debug=False)
        self.agent.update_belief(GaussianBelief)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())

        observed = [m.validity for m in measurements]
        reward, done, test_reward = self.get_reward(obstacles_pt, observed, self.is_training)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        target_b_state = self.agent.get_belief_state()
        target_b_cov = self.agent.get_belief_cov()
        for n in range(self.num_targets):
            r_b, alpha_b, _ = util.relative_measure(target_b_state[self.target_dim*n: self.target_dim*n+2], 
                                                self.agent.state)
            rel_target_vel = util.coord_change2b(target_b_state[self.target_dim*n: self.target_dim*n+2], 
                                                        alpha_b + self.agent.state[-1])
            self.state.extend([r_b, alpha_b, 
                                rel_target_vel[0], rel_target_vel[1], 
                                    np.log(LA.det(target_b_cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])),
                                        float(observed[n])])

        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'test_reward': test_reward}

class TargetTrackingInfoPlanner2(gym.Env):
    """
    Double Integrator 
    """
    def __init__(self, target_init_cov=TARGET_INIT_COV, q_true = 0.01, q = 0.01, target_init_vel=0.0,
                    map_name='empty', is_training=True, known_noise=True, num_targets=1):
        gym.Env.__init__(self)
        self.seed()
        self.id = 'TargetTracking-info2'
        self.state = None 
        self.action_space = spaces.Discrete(12)
        self.action_map = {}  
        for (i,v) in enumerate([3,2,1,0]):      
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)
        self.target_dim = 4
        self.viewer = None
        self.is_training = is_training

        self.num_targets = num_targets
        tau = 0.5 # sec
        self.q = q
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(map_path=os.path.join(map_dir_path,map_name), 
                                        r_max = self.sensor_r, fov = self.fov/180.0*np.pi)
        # LIMITS for GYM
        self.vel_limit = np.array([2.0, 2.0])
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate(( self.MAP.mapmin,[-np.pi])), 
                                            np.concatenate(( self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,  -self.vel_limit)), 
                                            np.concatenate((self.MAP.mapmax,  self.vel_limit))]
        self.limit['state'] = [np.array([0.0, -np.pi, -self.vel_limit[0], -self.vel_limit[1], -50.0, 0.0, 0.0, -np.pi ]),
                               np.array([600.0, np.pi, self.vel_limit[0], self.vel_limit[1], 50.0, 2.0, self.sensor_r, np.pi])]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.concatenate((self.MAP.origin, [0.0]))
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = target_init_cov
        self.target_init_vel = target_init_vel*np.ones((2,))
        self.target_noise_sd = self.q * np.concatenate((
                        np.concatenate((tau**3/3*np.eye(2), tau**2/2*np.eye(2)), axis=1),
                        np.concatenate((tau**2/2*np.eye(2), tau*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_sd
        else:
            #self.target_true_noise_sd = q_true * np.array([[5,0,0,0],[0,5,0,0],[0,0,1,0],[0,0,0,1]], dtype = np.float)
            self.target_true_noise_sd = q_true * np.concatenate((
                        np.concatenate((tau**2/2*np.eye(2), tau/2*np.eye(2)), axis=1),
                        np.concatenate((tau/2*np.eye(2), tau*np.eye(2)),axis=1) ))
        self.targetA = np.concatenate((np.concatenate((np.eye(2), tau*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))

        # Setup Ground Truth Target Simulation 
        map_nd = IGL.map_nd(self.MAP.mapmin, self.MAP.mapmax, self.MAP.res)
        if self.MAP.map is None:
            cmap_data = list(map(str, [0] * map_nd.size()[0] * map_nd.size()[1]))
        else:
            cmap_data = list(map(str, np.squeeze(self.MAP.map.astype(np.int8).reshape(-1, 1)).tolist()))
        se2_env = IGL.SE2Environment(map_nd, cmap_data, os.path.join(map_dir_path,'mprim_SE2_RL.yaml'))

        self.cfg = Configure(map_nd, cmap_data)
        sensor = IGL.RangeBearingSensor(self.sensor_r, self.fov, self.sensor_r_sd, self.sensor_b_sd, map_nd, cmap_data)
        self.agent = Agent_InfoPlanner(dim=3, sampling_period=tau, limit=self.limit['agent'], 
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL), 
                            se2_env=se2_env, sensor_obj=sensor)
        self.belief_target = BeliefWrapper()
        self.target = TargetWrapper(dim=2)

    def reset(self, init_random=True):
        if init_random:
            isvalid = False
            while(not isvalid):
                if self.MAP.map is None:
                    a_init = self.agent_init_pos[:2]
                else:
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin                    
                if not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL)):
                    rand_ang = np.random.rand() * 2 * np.pi - np.pi 
                    t_r = INIT_DISTANCE
                    t_init = np.array([t_r * np.cos(rand_ang), t_r * np.sin(rand_ang)]) + a_init
                    if (np.sqrt(np.sum((t_init - a_init)**2)) < MARGIN):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init, MARGIN2WALL))
            a_init = IGL.SE3Pose(np.concatenate((a_init, [np.random.random() * 2 * np.pi - np.pi])), np.array([0, 0, 0, 1]))
        else:
            a_init = IGL.SE3Pose(self.agent_init_pos, np.array([0, 0, 0, 1]))
            t_init = self.target_init_pos

        # Build a target
        target = self.cfg.setup_se2_targets(n_targets=self.num_targets, 
                                             init_odom=[np.concatenate((t_init, [np.random.rand() * 2 * np.pi - np.pi]))],
                                                q=self.q, 
                                                policy=Policy.random_policy(0.5, 60)#, collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
                                                )  # Integrator Ground truth Model
        belief_target = self.cfg.setup_integrator_belief(n_targets=self.num_targets, q=self.q, 
                                                init_pos=[t_init + 10.*(np.random.rand(2)-0.5)], 
                                                cov_pos=self.target_init_cov, cov_vel=self.target_init_cov, 
                                                init_vel=(0.0, 0.0)) 
         # Build a robot 
        self.agent.reset(a_init, belief_target) 
        r, alpha, _ = util.relative_measure(self.agent.get_belief_state()[:2], self.agent.state)
        self.state = np.array([r, alpha, 0.0, 0.0, 
                                np.log(LA.det(self.agent.get_belief_cov())), 
                                0.0, self.sensor_r, np.pi])
        self.target.reset(target)
        self.belief_target.update(self.agent.get_belief_state(), self.agent.get_belief_cov())

        return np.array(self.state)

    def get_reward(self, obstacles_pt, observed=False, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = 1./max(1.0, obstacles_pt[0]**2)
        if not observed:
            reward = - penalty
        else:
            logdetcov = np.log(LA.det(self.agent.get_belief_cov()))
            
            reward = 1. + max(0.0, -0.1*logdetcov - penalty) #- 0.1*(np.sqrt(pos_dist)-4.0)  #np.log(np.linalg.det(self.target_cov_prev)) - logdetcov 
        test_reward = None

        if not(is_training):
            test_reward = - np.log(LA.det(self.agent.get_belief_cov()))
            
        return reward, False, test_reward

    def step(self, action):
        self.agent.update(action, self.target.state) 

        # Update the true target state
        self.target.update()
        # Observe
        measurements = self.agent.observation(self.target.target)
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        # Update the belief of the agent on the target using KF
        GaussianBelief = IGL.MultiTargetFilter(measurements, self.agent.agent, debug=False)
        self.agent.update_belief(GaussianBelief)
        self.belief_target.update(self.agent.get_belief_state(), self.agent.get_belief_cov())

        reward, done, test_reward = self.get_reward(obstacles_pt, measurements[0].validity, self.is_training)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        r_b, alpha_b, _ = util.relative_measure(self.agent.get_belief_state()[:2], self.agent.state)
        rel_target_vel = util.coord_change2b(self.agent.get_belief_state()[:2], 
                                                        alpha_b + self.agent.state[-1])
        self.state = np.array([r_b, alpha_b, rel_target_vel[0], rel_target_vel[1],
                                np.log(LA.det(self.agent.get_belief_cov())), 
                                float(measurements[0].validity), obstacles_pt[0], obstacles_pt[1]])
        return self.state, reward, done, {'test_reward': test_reward}


