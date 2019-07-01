""" envs/ folder is for openAIgym-like simulation environments
To use, 
>>> import envs
>>> env = envs.make("NAME_OF_ENV")

"""
import gym

def make(env_name, render=False, figID=0, record=False, 
                    ros=False, dirname='', map_name="empty", is_training=True, 
                    num_targets=1, T_steps=None, im_size=None):
    try:
        env = gym.make(env_name)
        if record:
            env = Monitor(env, directory=args.log_dir)
    except:
        if 'Target' in env_name:
            from gym import wrappers
            import envs.target_tracking.target_tracking as ttenv
            from envs.target_tracking.target_tracking_advanced import TargetTrackingEnvRNN
            # from envs.target_tracking.target_tracking_infoplanner import TargetTrackingInfoPlanner1, TargetTrackingInfoPlanner2
            from envs.target_tracking import display_wrapper 
            if T_steps is None:
                if num_targets > 1:
                    T_steps = 150
                else:
                    T_steps = 100
            if env_name == 'TargetTracking-v0':
                env0 = ttenv.TargetTrackingEnv0(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-v1':
                env0 = ttenv.TargetTrackingEnv1(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-v2':
                env0 = ttenv.TargetTrackingEnv2(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-v3':
                env0 = ttenv.TargetTrackingEnv3(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-v4':
                env0 = ttenv.TargetTrackingEnv4(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-v5':
                env0 = ttenv.TargetTrackingEnv5(map_name=map_name, is_training=is_training, num_targets=num_targets, im_size=im_size)
            elif env_name == 'TargetTracking-vRNN':
                env0 = TargetTrackingEnvRNN(map_name=map_name, is_training=is_training, num_targets=num_targets)
                T_steps = 200
            elif env_name == 'TargetTracking-info1':
                env0 = TargetTrackingInfoPlanner1(map_name=map_name, is_training=is_training, num_targets=num_targets)
            elif env_name == 'TargetTracking-info2':
                env0 = TargetTrackingInfoPlanner2(map_name=map_name, is_training=is_training, num_targets=num_targets)
            else:
                raise ValueError('no such environments')

            env = wrappers.TimeLimit(env0, max_episode_steps=T_steps)
            if ros:
                from envs.ros_wrapper import Ros
                env = Ros(env)
            if render:
                env = display_wrapper.Display2D(env, figID=figID)
            if record:
                env = display_wrapper.Video2D(env, dirname = dirname)
        else:
            from envs import classic_mdp
            env = classic_mdp.model_assign(env_name)
    return env
