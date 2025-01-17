import brl_util as util
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle, os
import pdb
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='CartPole-v0')
parser.add_argument('--env_kind', help='environment kind?', default='classic_control')
parser.add_argument('--nb_train_steps', type=int, help='the number of training steps', default=200000)
parser.add_argument('--nb_epoch_steps', type=int, help='the number of training steps', default=1000)
parser.add_argument('--nb_files', type=int, help='the number of training steps', default=5)
parser.add_argument('--init_value', type=float, help='the number of training steps', default=0.0)

args = parser.parse_args()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

alg_names = [r'ADFQ, $\epsilon$-greedy', 'ADFQ, TS','DQN', 'Double DQN']
color_id = [2,3,0,1]
dir_names = {}

if args.env_kind == 'classic_control':
	dir_names[r'ADFQ, $\epsilon$-greedy'] = os.path.join(os.path.join('results/',args.env_kind),args.env+'_batch_1e-3_eg_200K')
	dir_names['ADFQ, TS'] = os.path.join(os.path.join('results/',args.env_kind),args.env+'_batch_1e-3_bayes_200K')
	dir_names['DQN'] = os.path.join(os.path.join('results/',args.env_kind+'/baselines'),args.env+'_batch_200K')
	dir_names['Double DQN'] = os.path.join(os.path.join('results/',args.env_kind+'/baselines'),args.env+'_batch_200K_DDQN')
else:
	dir_names[r'ADFQ, $\epsilon$-greedy'] = os.path.join(os.path.join('results/',args.env_kind), 'ADFQ-eg')
	dir_names['ADFQ, TS'] = os.path.join(os.path.join('results/',args.env_kind), 'ADFQ-ts')
	dir_names['DQN'] = os.path.join(os.path.join('results/',args.env_kind), 'DQN')
	dir_names['Double DQN'] = os.path.join(os.path.join('results/',args.env_kind), 'DDQN')

fnames = {}
for alg in alg_names:
	fnames[alg] = [x for x in os.listdir(dir_names[alg]) if (x!= '.DS_Store') and (args.env in x)]
	if len(fnames[alg]) < args.nb_files:
		print("The number of datasets is not %d!"%args.nb_files)
		pdb.set_trace()
	fnames[alg] = fnames[alg][:args.nb_files]

data_keys = ['online_reward', 'test_reward']#, 'q_vals']
datasets = {k:{a:[] for a in alg_names} for k in data_keys}
plot_sets = {k:[] for k in data_keys}

for alg in alg_names:
	for f in fnames[alg]:
		X = pickle.load(open(os.path.join(os.path.join(dir_names[alg],f), 'records.pkl'),'rb'))
		for k in data_keys:
			if 'ADFQ' in alg and k=='q_vals':
				k0 = 'q_mean'
			else:
				k0 = k
			X[k0] = np.asarray(X[k0])
			if k == 'test_reward':
				datasets[k][alg].append(np.asarray(util.smoothing(
					np.mean(X[k0][:100], axis=-1),6)))
					#np.concatenate(([args.init_value],np.mean(X[k0][:100], axis=-1))),6)))
			elif k == 'online_reward':
				datasets[k][alg].append(np.asarray(util.smoothing(
					X[k0][:100],6)))
					#np.concatenate(([args.init_value],X[k0][:100])),6)))
			else:
				datasets[k][alg].append(np.asarray(util.smoothing(
					X[k0][:100],6)))
					#np.concatenate(([0],X[k0][:100])),6)))
	for k in data_keys:
		plot_sets[k].append(np.asarray(datasets[k][alg]))
		if plot_sets[k][-1].shape == (2,):
			pdb.set_trace()
		
for k in data_keys:
	plot_sets[k] = np.asarray(plot_sets[k])

alg_names_b = alg_names
figure0, figure1 = None, None

pdb.set_trace()
title = args.env.split('NoFrameskip-v4')[0]
x_label = 'Training Steps (millions)'
#x_vals = range(0, int(args.nb_train_steps/args.nb_epoch_steps)+1, int(int(args.nb_train_steps/args.nb_epoch_steps)/(plot_sets['test_reward'].shape[-1]-1)))
x_vals = np.arange(0, 5, 0.05)
f, ax = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['test_reward'], 
	labels =alg_names_b, legend=(True, 'lower right'), x_label=x_label, y_label='Average Reward per Episode', 
	title =title, x_vals = x_vals, colors = [colors[i] for i in color_id], figure=figure0)

f1, ax1 = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['online_reward'], 
	labels =alg_names_b, legend=(True, 'lower right'), x_label=x_label, y_label='Average Reward per Episode', 
	title=title, x_vals = x_vals, colors =[colors[i] for i in color_id], figure=figure1)
	
# f2, ax2 = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['q_vals'], alg_names, 
# 	legend=(True, 'lower right'), x_label=x_label, y_label='Average Q values', title=title,
# 	x_vals = x_vals, colors =[colors[i] for i in color_id])

if 'CartPole' in args.env:
	ax.axhline(y=200, color='k')
	ax1.axhline(y=199, color='k')

f.savefig('_'.join([args.env,'test_reward']),  bbox_inches='tight', pad_inches=0)
f1.savefig('_'.join([args.env,'online_reward']),  bbox_inches='tight', pad_inches=0)
f2.savefig('_'.join([args.env,'q_vals']),  bbox_inches='tight', pad_inches=0)
plt.show()