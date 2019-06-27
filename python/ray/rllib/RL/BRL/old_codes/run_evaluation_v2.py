import brl_new as brl
import tabularRL as trl
import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import brl_util_new as util
import pdb
import seeding
import pickle
import argparse
import os
from collections import OrderedDict

colors = ['r','k','b','g','c','m','y','burlywood','chartreuse','0.8']

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=10, help='the number of trials to repeat')
parser.add_argument('--discount', type=float, default=0.95, help= 'discount factor')
parser.add_argument('--learning_rate_init', type=float, default=0.5, help ='initial learning rate for Q-learning')
parser.add_argument('--function', type=str)
parser.add_argument('--param_dir', type=str, default="params/")
parser.add_argument('--result_dir', type=str, default='./')
parser.add_argument('--scene', type=str, default='')

args = parser.parse_args()

"""""""""
EVALUATION ADFQ-Numeric, ADFQ-Approx, KTD-Q (egreedy / behavior policy), and, Watkin's Q-learning

* ALGORITHMS :
	- ADFQ-Numeric : ADFQ with numerically computed mean and variance
	- ADFQ-Approx  : ADFQ with small variance approximation
	- KTD-Q		   : KTD-Q
	- Q-learning   : Q-learning 

* Action Selection
	- Off-Policy for RMSE results
	- ADFQ : Egreedy, Eg+Bayesian, Bayesian
	- KTD-Q : Egreedy, Behavior Policy
	- Q-learning : Egreedy, Boltzmann

* TASKs : LOOP, GRID5, GRID10, Dearden's MAZE, MiniMaze 
* Evaluations:
	1) RMSE Plots
	2) Total Rewards Charts
	3) Optimal Policy Route Plots - Grids
	4) Uncertainty Map - Grids?
	5) Greedy Evaluation Graphs 

* Initial Mean Policy - Determined after observing the first reward for all domains
* Initial Vairances for ADFQ 
* KTD-Q : parameters - kappa(1, 2/n, n), evolution noise, observation noise
* Q-learning, Learning Rate - LSPI approach starting from 0.5
"""""""""

Nrun = args.num_trials
discount = args.discount
learning_rate_init = args.learning_rate_init
variances = [1.0, 10.0, 100.0]
epsilons = [0.0, 0.1, 0.2, 0.5]
boltz_temp = [0.1, 0.3, 0.5]
stochasticity = [0.0, 0.1, 0.5]
ktd_noise = [0.0, 0.1, 1.0]
alpha = 0.5

update_policies = ['SoftApprox', 'Approx']
labels = ["Q-learning"]+["ADFQ-"+policy_name for policy_name in update_policies]+["KTD-Q"]
labels_act = ["Q-learning, eg ", "Q-learning, btz "] \
			+ ["ADFQ-"+policy_name+", eg" for policy_name in update_policies] \
			+["ADFQ-"+policy_name+", eg+bayes" for policy_name in update_policies] \
			+["ADFQ-"+policy_name+", bayes" for policy_name in update_policies] \
			+ ["KTD-Q, eg ", "KTD-Q, active " ]
labels_display = [r'Q-learning, $\epsilon$-greedy', "Q-learning, boltzmann ",] \
			+['ADFQ-'+policy_name+r', $\epsilon$-greedy' for policy_name in update_policies] \
			+ ["ADFQ-"+policy_name+", semi-BS" for policy_name in update_policies] \
			+["ADFQ-"+policy_name+", BS" for policy_name in update_policies] \
			+ [r'KTD-Q, $\epsilon$-greedy', "KTD-Q, active " ]

scene_set={'loop':(2,util.T_loop,9*2), 'grid5':(4,util.T_grid5,25*4),'grid10':(4,util.T_grid10,100*4), 
		'minimaze':(4,util.T_minimaze,112*4), 'maze':(4,util.T_maze,264*4)}
scene_set = OrderedDict(sorted(scene_set.items(), key = lambda t:t[1][1]))

def off_policy_helper(scene, models, result, alg_name, del_t):
	best_key = models.keys()[np.argmin([sum([x.Q_err[-1] for x in y]) for y in models.values()])]
	result['Q_err'].append(np.array([x.Q_err for x in models[best_key]]))
	result['tab'].append([alg_name, str(round((sum(result['Q_err'][-1])/float(Nrun))[-1],2)), str(del_t), str(best_key[0])])
	return np.mean([x.test_rewards for x in models[best_key]], axis=0)

def off_policy(eval_scenes=None):	
	if eval_scenes == None: 
		eval_scenes = scene_set.keys()
	variance = 100.0

	print("Off-Policy Learning...")
	np.random.seed(0)
	test_rewards = {}
	test_qerr = {}
	for scene in eval_scenes:
		test_rewards[scene] = {}
		test_qerr[scene] = {}
		f_rmse = open(os.path.join(args.result_dir,"rmse.txt"),"a")		
		v = scene_set[scene]
		alg_num = 0
		result = {'Q_err':[], 'tab':[]}
		print("Domain:" + scene)
		actionSet = [np.random.choice(v[0],v[1]) for i in range(Nrun)]
		if scene == 'loop': # Non-episodic
			init_means = 1.0/(1-discount)*np.ones((Nrun,))
		else:
			init_means = np.ones((Nrun,))
	
		# Q-learning (fixed learning rate)
		print(labels[alg_num])
		t_start = time.time()
		model = [trl.Qlearning(scene, alpha, discount, init_means[i], init_policy=False) for i in range(Nrun)]
		[model[i].learning('offline',actionSet[i], rate_decay=True, eval_greedy=True) for i in range(Nrun)]
		test_rewards[scene][labels[alg_num]] = np.mean([x.test_rewards for x in model], axis=0)
		test_qerr[scene][labels[alg_num]] = np.mean([x.Q_err for x in model], axis=0)
		print(time.time()-t_start)
		alg_num+=1

		for useScale in [False, True]:
			for batch_size in [0, 20]:
				for (noise_c, noise_n) in [(0.0,1e-8), (0.0,0.0), (0.0,1e-3), (0.0, 1.0), (1e-8,0.0), (1e-3,0.0), (1.0, 0.0)]:
					m_r, m_q = {}, {}
					alg_num = 1
					"""
					# ADFQ - Numeric
					print(labels[alg_num])
					t_start = time.time()
					models = {}
					model = [brl.adfq(scene, discount, init_means[i], variance, init_policy=False) for i in range(Nrun)]
					[model[i].learning('Numeric', 'offline', actionSet[i], eval_greedy = True, useScale=useScale, 
						batch_size=batch_size, noise = noise_n, noise_c = noise_c) for i in range(Nrun)]
					m_r[labels[alg_num]] = np.mean([x.test_rewards for x in model], axis=0)
					m_q[labels[alg_num]] = np.mean([x.Q_err for x in model], axis=0)
					print(time.time()-t_start)
					alg_num+=1
					"""
					# ADFQ - SoftApprox
					print(labels[alg_num])
					t_start = time.time()
					model = [brl.adfq(scene, discount, init_means[i], variance, init_policy=False) for i in range(Nrun)]
					[model[i].learning('SoftApprox', 'offline', actionSet[i], eval_greedy = True, useScale=useScale, 
						batch_size=batch_size, noise = noise_n, noise_c = noise_c) for i in range(Nrun)]
					m_r[labels[alg_num]] = np.mean([x.test_rewards for x in model], axis=0)
					m_q[labels[alg_num]] = np.mean([x.Q_err for x in model], axis=0)
					print(time.time()-t_start)
					alg_num+=1

					# ADFQ - Approx
					print(labels[alg_num])
					t_start = time.time()
					model = [brl.adfq(scene, discount, init_means[i], variance, init_policy=False) for i in range(Nrun)]
					[model[i].learning('Approx', 'offline', actionSet[i], eval_greedy = True, useScale=useScale, 
						batch_size=batch_size, noise = noise_n, noise_c = noise_c) for i in range(Nrun)]
					m_r[labels[alg_num]] = np.mean([x.test_rewards for x in model], axis=0)
					m_q[labels[alg_num]] = np.mean([x.Q_err for x in model], axis=0)
					print(time.time()-t_start)
					alg_num+=1

					test_rewards[scene][(useScale, batch_size, noise_c, noise_n)] = m_r
					test_qerr[scene][(useScale, batch_size, noise_c, noise_n)] = m_q
					pickle.dump(test_rewards, open("test_rewards.pkl","wb"))
					pickle.dump(test_qerr, open("test_qerr.pkl","wb"))

def save_to_file(scene, dataset, reward_based=True):
	for (k,v) in dataset.items():
		if k != 'tab':
			if reward_based:
				np.save(os.path.join(args.result_dir,k+"_"+scene), np.array(v))
			else:
				np.save(os.path.join(args.result_dir,k+"_v2_"+scene), np.array(v))

def analysis(fname, fun, evaluation_ids= None):
	data = np.load(fname)
	if args.scene == 'loop':
		T = util.T_loop
	elif args.scene == 'grid5':
		T = util.T_grid5
	elif args.scene == 'grid10':
		T = util.T_grid10
	elif args.scene == 'minimaze':
		T = util.T_minimaze
	elif args.scene == 'maze':
		T = util.T_maze

	if fun == 'off_policy':
		util.plot_IQR(T, data, labels, False, 'Learning Step', 'RMSE')
	else:
		if evaluation_ids:
			ids = evaluation_ids
		else:
			ids = range(len(data))
			
		util.plot_smoothing(T, data[ids,:,:], [labels_display[i+2] for i in ids], 
			'Learning Step', 'Total Cumulative Rewards', legend=(True,(1.0,0.0)))

if __name__ == "__main__":
	if not args.scene:
		eval_scene = None
	else:
		eval_scene = args.scene.split(':')

	if args.function == 'off_policy':
		off_policy(eval_scene)
	elif args.function == 'action_policy':
		action_policy(eval_scene)
	elif args.function == 'analysis_off':
		analysis(os.path.join(args.result_dir, 'err_'+args.scene+'.npy'), 'off_policy')
	elif args.function == 'analysis_action':
		analysis(os.path.join(args.result_dir, 'reward_v2_'+args.scene+'.npy'), 'action')

