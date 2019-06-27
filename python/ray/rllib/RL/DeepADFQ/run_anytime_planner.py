import pdb
import numpy as np
import pickle
import datetime, json, os, argparse

import envs
import pyInfoGathering as IGL

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--render', help='render', type=int, default=0)
parser.add_argument('--record', help='record', type=int, default=0)
parser.add_argument('--map', type=str, default="empty")
parser.add_argument('--env', help='environment ID', default='TargetTracking-info0')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)

args = parser.parse_args()

if __name__ == "__main__":
    # Initialize Planner
    n_controls = 5
    T = 12
    delta = 3
    eps = np.infty
    arvi_time = 1
    range_limit = np.infty
    debug = 1
    directory = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
    if not os.path.exists(directory):
            os.makedirs(directory)
    else:
            ValueError("The directory already exists...", directory)
    json.dump(vars(args), open(os.path.join(directory, 'learning_prop.json'), 'w'))

    env = envs.make(args.env, 
                    render = bool(args.render), 
                    record = bool(args.record), 
                    ros = bool(args.ros), 
                    dirname=directory, 
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    is_training=False
                    )
    timelimit_env = env
    while( not hasattr(timelimit_env, '_elapsed_steps')):
        timelimit_env = timelimit_env.env    

    ep_nlogdetdov = []
    for episode in range(args.repeat):
        planner = IGL.InfoPlanner()
        np.random.seed(args.seed + episode)
        
        # Save Planner Output
        plannerOutputs = [0] * 1

        # Main Loop
        nlogdetcov = []
        done = False
        obs = env.reset()
        t = 0
        while(not done):
            print('Timestep ', t)

            # Plan for individual Robots (Every n_controls steps)
            if t % n_controls == 0:
                plannerOutputs[0] = planner.planARVI(timelimit_env.env.agent.agent, T, delta, eps, arvi_time, debug, 0)

            # Apply Control
            obs, reward, done, info = env.step(plannerOutputs[0].action_idx[-1])
            if args.render:
                env.render()
            nlogdetcov.append(info['test_reward'])
            # Pop off last Action manually (UGLY)
            plannerOutputs[0].action_idx = plannerOutputs[0].action_idx[:-1]
            t += 1
        obs = env.reset()
        ep_nlogdetdov.append(np.sum(nlogdetcov))
        print("Ep.%d Cumulative Rewards = %.2f"%(episode, ep_nlogdetdov[-1]))

    if args.repeat > 1:
        f = open(os.path.join(directory,'reward_batch_%d'%args.repeat + '.pkl'),'wb')
        pickle.dump(ep_nlogdetdov, f)











