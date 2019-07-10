"""
This code was modified from a OpenAI baseline code - baselines0/baselines0/deepq/experiments/train_cartpole.py for running ADFQ
"""
from baselines0.common import set_global_seeds

import models
import numpy as np
import tensorflow as tf
import datetime, json, os, argparse
from gym.wrappers import Monitor
import envs
import pdb

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
parser.add_argument('--log_dir', type=str, default='.')
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

def train():
    set_global_seeds(args.seed)
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
                    im_size=args.im_size,
                    )
    hiddens = args.hiddens.split(':')
    hiddens = [int(h) for h in hiddens]
    with tf.device(args.device):
        if args.env == 'TargetTracking-v5':
            import simple_imtracking as simple
            model = models.cnn_plus_mlp(
                            convs=[(8, 4, 2), (16, 3, 1)],
                            hiddens=hiddens,
                            dueling=bool(args.dueling),
                            init_mean = args.init_mean, 
                            init_sd = args.init_sd,
            )
        else:
            import simple_tracking as simple
            model = models.mlp(hiddens, init_mean=args.init_mean, init_sd=args.init_sd)

        act, records = simple.learn(
            env,
            q_func=model,
            lr=args.learning_rate,
            lr_decay_factor=args.learning_rate_decay_factor,
            lr_growth_factor=args.learning_rate_growth_factor,
            max_timesteps=args.nb_train_steps,
            buffer_size=args.buffer_size,
            batch_size = args.batch_size,
            exploration_fraction=args.eps_fraction,
            exploration_final_eps=args.eps_min,
            target_network_update_freq=args.target_update_freq,
            print_freq=10,
            checkpoint_freq=int(args.nb_train_steps/10),
            learning_starts=args.nb_warmup_steps,
            gamma=args.gamma,
            prioritized_replay=bool(args.prioritized),
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            callback=None,#callback,
            epoch_steps = args.nb_epoch_steps,
            noise = args.noise,
            varTH=args.varth,
            alg = args.alg,
            gpu_memory=args.gpu_memory,
            act_policy=args.act_policy,
            save_dir=directory,
            nb_test_steps=args.nb_test_steps,
            scope = args.scope,
            test_eps = args.test_eps,
            render = (bool(args.render) or bool(args.ros)),
            map_name = args.map,
            num_targets = args.nb_targets,
            im_size = args.im_size,
        )
        print("Saving model to model.pkl")
        act.save(os.path.join(directory,"model.pkl"))
        plot(records, directory)
    memo = input("Memo for this experiment?: ")
    f = open(os.path.join(directory,"memo.txt"), 'w')
    f.write(memo)
    f.close()
    if args.record == 1:
        env.moviewriter.finish()
        
def test():
    set_global_seeds(args.seed)
    import json
    if args.env == 'TargetTracking-v5':
        import simple_imtracking as simple
    else:
        import simple_tracking as simple
            
    learning_prop = json.load(open(os.path.join(args.log_dir, 'learning_prop.json'),'r'))
    env = envs.make(args.env, 
                    render = bool(args.render), 
                    record = bool(args.record), 
                    ros = bool(args.ros), 
                    map_name=args.map,
                    num_targets=learning_prop['nb_targets'],
                    dirname = args.log_dir,
                    is_training=True,
                    im_size=args.im_size,
                    )
    act_params = {'scope': learning_prop['scope'], 'eps': args.test_eps}
    act = simple.load(os.path.join(args.log_dir, args.log_fname), act_params)
    
    if args.ros_log:
        from envs.target_tracking.ros_wrapper import RosLog
        log = RosLog(num_targets=args.nb_targets, wrapped_num=args.ros + args.render + args.record + 1)
    t =0
    while(t < args.nb_test_steps): # test episode
        t += 1
        obs, done = env.reset(), False
        episode_rew = 0
        env.render()
        while not done:
            if args.render:
                env.render()
                # pdb.set_trace()
            if args.ros_log:
                log.log(env)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
    if args.record :
        env.moviewriter.finish()
    if args.ros_log :
        log.save(args.log_dir)


def plot(records, directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from simple_tracking import iqr

    m = len(records['q_mean'])
    x_vals = range(0 , args.nb_epoch_steps*m, args.nb_epoch_steps)
    
    f0, ax0 = plt.subplots(3, sharex=True, sharey=False)
    _ = ax0[0].plot(x_vals, records['q_mean'])
    _ = ax0[0].set_ylabel('Average Q means')

    _ = ax0[1].plot(x_vals, np.log(records['q_sd']))
    _ = ax0[1].set_ylabel('Log of Average Q SD')

    _ = ax0[2].plot(x_vals, records['loss'])
    _ = ax0[2].set_ylabel('Loss')
    _ = ax0[2].set_xlabel('Learning Steps')
    _ = [ax0[i].grid() for i in range(3)]

    f1, ax1 = plt.subplots()
    _ = ax1.plot(x_vals, records['online_reward'])
    _ = ax1.set_ylabel('Average recent 100 rewards')
    _ = ax1.set_xlabel('Learning Steps')
    _ = ax1.grid()

    f2, ax2 = plt.subplots()
    m, ids25, ids75 = iqr(np.array(records['test_reward']).T)
    _ = ax2.plot(x_vals, m, color='b')
    _ = ax2.fill_between(x_vals, list(ids75), list(ids25), facecolor='b', alpha=0.2)
    _ = ax2.set_ylabel('Test Rewards')
    _ = ax2.set_xlabel('Learning Steps')
    _ = ax2.grid()

    f3, ax3 = plt.subplots()
    m, ids25, ids75 = iqr(np.array(records['eval_value']).T)
    _ = ax3.plot(x_vals, m, color='b')
    _ = ax3.fill_between(x_vals, list(ids75), list(ids25), facecolor='b', alpha=0.2)
    _ = ax3.set_ylabel('Test Mean -LogDetCov')
    _ = ax3.set_xlabel('Learning Steps')
    _ = ax3.grid()

    _ = f0.savefig(os.path.join(directory, "result.png"))
    _ = f1.savefig(os.path.join(directory, "online_reward.png"))
    _ = f2.savefig(os.path.join(directory, "test_reward.png"))
    _ = f3.savefig(os.path.join(directory, "test_logdetcov.png"))

if __name__ == '__main__':
    if args.mode == 'train':
        i = 0
        while(i < args.repeat):
            print("Train %d"%i)
            train()
            i += 1
    elif args.mode =='test':
        test()

