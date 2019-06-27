import brl_util as util
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle, os
import pdb
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v0')
parser.add_argument('--env_kind', help='environment kind?', default='classic_control')
parser.add_argument('--nb_train_steps', type=int, help='the number of training steps', default=200000)
parser.add_argument('--nb_epoch_steps', type=int, help='the number of training steps', default=1000)
parser.add_argument('--nb_files', type=int, help='the number of training steps', default=5)
parser.add_argument('--init_value', type=float, help='the number of training steps', default=0.0)
parser.add_argument('--baselines', type=str, default='')

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

pdb.set_trace()
data_keys = ['online_reward', 'test_reward_t', 'q_vals']
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
			if k == 'test_reward_t':
				datasets[k][alg].append(np.asarray(util.smoothing(
					np.mean(X[k0][:100], axis=-1),6)))
			elif k == 'online_reward':
				datasets[k][alg].append(np.asarray(util.smoothing(
					X[k0][:100],6)))
			else:
				datasets[k][alg].append(np.asarray(util.smoothing(
					X[k0][:100],6)))
	for k in data_keys:
		plot_sets[k].append(np.asarray(datasets[k][alg]))
		if plot_sets[k][-1].shape == (2,):
			pdb.set_trace()
		
for k in data_keys:
	plot_sets[k] = np.asarray(plot_sets[k])

if args.baselines:
	f, ax = plt.subplots()
	f1, ax1 = plt.subplots()
	b_data = pickle.load(open(os.path.join(os.path.join('results', args.env_kind), args.baselines), 'rb'))
	b_rewards = b_data 
	b_mean = np.mean(b_rewards)
	b_std = np.std(b_rewards)
	tmp = ax.axhline(y=b_mean, color='k', linestyle='--', linewidth=1.5)
	ax.axhspan(b_mean - 0.5 * b_std, b_mean + 0.5 * b_std, alpha=0.2, color='k')
	#tmp1 = ax1.axhline(y=b_mean, color='k', linestyle='--', linewidth=1.5)
	#ax1.axhspan(b_mean - 0.5 * b_std, b_mean + 0.5 * b_std, alpha=0.2, color='k')
	figure0 = (f, ax, [tmp])
	figure1 = (f1, ax1, [])#[tmp1])
	alg_names_b = ['ARVI'] + alg_names

else:
	alg_names_b = alg_names
	figure0, figure1 = None, None

title = 'Single - Empty Environment'
T_H = 100
x_label = 'Learning trajectories'
x_vals = range(0, int(args.nb_train_steps/args.nb_epoch_steps), int(int(args.nb_train_steps/args.nb_epoch_steps)/(plot_sets['test_reward_t'].shape[-1]-1)))
x_vals = x_vals[:49]
pdb.set_trace()
f, ax = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['test_reward_t'], 
	labels =alg_names_b, legend=(True, 'lower right', (1.0, 0.0)), x_label=x_label, y_label=r'Average Cumulative $-\log \det \Sigma$', 
	title =title, x_vals = x_vals, colors = [colors[i] for i in color_id], figure=figure0)

f1, ax1 = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['online_reward'], 
	labels =alg_names_b[1:], legend=(True, 'lower right', (1.0, 0.0)), x_label=x_label, y_label=r'Average Cumulative $-\log \det \Sigma$', 
	title=title, x_vals = x_vals, colors =[colors[i] for i in color_id], figure=figure1)
	
f2, ax2 = util.plot_sd(int(args.nb_train_steps/args.nb_epoch_steps), plot_sets['q_vals'], alg_names, 
	legend=(True, 'upper left', (0.0, 1.0)), x_label=x_label, y_label='Average Q values', title=title,
	x_vals = x_vals, colors =[colors[i] for i in color_id])

if 'CartPole' in args.env:
	ax.axhline(y=200, color='k')
	ax1.axhline(y=199, color='k')

f.savefig('_'.join([args.env,'test_nlogdetcov']),  bbox_inches='tight', pad_inches=0)
f1.savefig('_'.join([args.env,'online_reward']),  bbox_inches='tight', pad_inches=0)
f2.savefig('_'.join([args.env,'q_vals']),  bbox_inches='tight', pad_inches=0)
plt.show()