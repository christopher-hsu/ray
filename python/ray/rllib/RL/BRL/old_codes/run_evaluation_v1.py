import brl 
import tabularRL as tq
import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import brl_util as util
import pdb
import seeding
import pickle
import argparse
import os
from tabulate import tabulate
from collections import OrderedDict

colors = ['r','k','b','g','c','m','y','burlywood','chartreuse','0.8']

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=10, help='the number of trials to repeat')
parser.add_argument('--discount', type=float, default=0.9, help= 'discount factor')
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
alphas = [0.1, 0.3, 0.5]

update_policies = ['SoftApprox']
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

def off_policy(eval_scenes=None):	
	if eval_scenes == None: 
		eval_scenes = scene_set.keys()

	print("Off-Policy Learning...")
	for scene in eval_scenes:
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
		models = {}
		for alpha in alphas:
			peq = [tq.Qlearning(scene, alpha, discount, init_means[i], init_policy=False) for i in range(Nrun)]
			[peq[i].learning('offline',actionSet[i], rate_decay=False) for i in range(Nrun)]
			models[(alpha,)] = peq
		off_policy_helper(scene, models, result, labels[alg_num], round((time.time()-t_start)/len(models) ,2))
		alg_num+=1

		# ADFQ - Numeric
		print(labels[alg_num])
		t_start = time.time()
		models = {}
		for var in variances:
			adfq = [brl.adfq(scene, discount, init_means[i], var, init_policy=False) for i in range(Nrun)]
			[adfq[i].learning(updatePolicy='Numeric', actionPolicy='offline', actionParam=actionSet[i]) for i in range(Nrun)]
			models[(var,)] = adfq
		off_policy_helper(scene, models, result, labels[alg_num], round((time.time()-t_start)/len(models) ,2))
		alg_num+=1

		# ADFQ - Approx
		print(labels[alg_num])
		t_start = time.time()
		models = {}
		for var in variances:
			adfq = [brl.adfq(scene, discount, init_means[i], var, init_policy=False) for i in range(Nrun)]
			[adfq[i].learning(updatePolicy='Approx', actionPolicy='offline', actionParam=actionSet[i]) for i in range(Nrun)]
			models[(var,)] = adfq
		off_policy_helper(scene, models, result, labels[alg_num], round((time.time()-t_start)/len(models) ,2))
		alg_num+=1

		# KTD-Q
		if not(scene == 'maze'):
			print(labels[alg_num])
			t_start = time.time()
			models = {}
			for var in variances:
				for kappa in [1.0, 0.5*v[2], v[2]]:
					ktd = [brl.ktd_Q(scene, discount, init_means[i], var, init_policy = False) for i in range(Nrun)]
					[ktd[i].learning(kappa, actionPolicy="offline", actionParam=actionSet[i]) for i in range(Nrun)]
					models[(var,kappa)] = ktd
			off_policy_helper(scene, models, result, labels[alg_num], round((time.time()-t_start)/len(models) ,2))
			alg_num+=1

		np.save(os.path.join(args.result_dir,"err_set_"+scene),result['Q_err'])
		f_rmse.write(scene+"\n")
		f_rmse.write(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
		f_rmse.write('\n')

def action_policy_helper(scene, models, result, alg_name, del_t, reward_based =True):
	if reward_based:
		rewards = [sum([x.get_total_reward() for x in y])/float(Nrun) for y in models.values()]
		best_key = models.keys()[np.argmax(rewards)]
	else:
		max_v = 0.0
		for (k,vec) in models.items():
			val = np.mean([np.mean(x.test_rewards[-20:]) for x in vec])
			if val > max_v:
				max_v = val
				best_key = k

	total_rewards = np.array([x.get_total_reward() for x in models[best_key]])
	result['reward'].append(np.array([x.test_rewards for x in models[best_key]]))
	result['count'].append(np.array([x.test_counts for x in models[best_key]]))
	try:
		result['Q'].append(np.array([x.means for x in models[best_key]]))
	except:
		result['Q'].append(np.array([x.Q for x in models[best_key]]))

	result['tab'].append([alg_name, str(round(sum(total_rewards)/float(Nrun),2)), str(np.std(total_rewards))] + [str(del_t)]+[str(v) for v in best_key])
	save_to_file(scene, result, reward_based = reward_based)

def action_policy(eval_scenes=None, useParam=True):
	if eval_scenes == None: 
		eval_scenes = scene_set.keys()
	headers = ["ALG. Rew_based", "Tot rewards", "SD","Elapsed T", "HyperParam1", "HyperParam2"]
	print("Aciton Policy Learning...")
	for scene in eval_scenes:
		v = scene_set[scene] 
		alg_num = 0
		f_rew = open(os.path.join(args.result_dir,"total_rewards_range.txt"),"a")
		f_rew.write("\n"+scene+"\n")
		print("Domain:" + scene)
		#f_rew.close()
		result = {'reward':[], 'count':[], 'Q':[], 'tab':[]}
		result2 = {'reward':[], 'count':[], 'Q':[], 'tab':[]}
				
		# # Q-learning_fixed, egreedy
		# print(labels_act[alg_num])
		# t_start = time.time()
		# models = {}
		# for alpha in alphas:
		# 	for es in epsilons: 
		# 		peq = [tq.Qlearning(scene, alpha, discount, 0.0, init_policy=True, TH=v[1]) for i in range(Nrun)]
		# 		[peq[i].learning('egreedy',es, eval_greedy =True, rate_decay=False) for i in range(Nrun)]
		# 		models[(alpha,es)] = peq
		# elapsed_t = round( (time.time()-t_start)/len(models) ,2)
		# action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
		# action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
		# print(result['tab'][-1])
		alg_num += 1

		# # Q-learning_fixed, boltzmann
		# print(labels_act[alg_num])
		# t_start = time.time()
		# models = {}
		# for alpha in alphas:
		# 	for tau in boltz_temp:
		# 		peq = [tq.Qlearning(scene, alpha, discount, 0.0, init_policy=True, TH=v[1]) for i in range(Nrun)]
		# 		[peq[i].learning('softmax',tau,eval_greedy =True, rate_decay=False) for i in range(Nrun)]
		# 		models[(alpha,tau,)] = peq
		# elapsed_t = round( (time.time()-t_start)/len(models) ,2)
		# action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
		# action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
		# print(result['tab'][-1])
		alg_num +=1

		# ADFQs - Egreedy
		for policy in update_policies:
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for var in  variances: 
				for es in epsilons:
					adfq = [brl.adfq(scene, discount, 0.0, var, init_policy=True, TH=v[1]) for i in range(Nrun)]
					[adfq[i].learning(updatePolicy=policy, actionPolicy='egreedy', actionParam = es, eval_greedy =True, updateParam=0.01) for i in range(Nrun)]
					models[(var,es)] = (adfq)
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			alg_num += 1
	
		# ADFQs - Eg+Bayesian
		for policy in update_policies:
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for var in variances: 
				for es in epsilons:
					adfq = [brl.adfq(scene, discount, 0.0, var, init_policy=True, TH=v[1]) for i in range(Nrun)]
					[adfq[i].learning(updatePolicy=policy, actionPolicy='semi-Bayes', actionParam = es, eval_greedy =True, updateParam=0.01) for i in range(Nrun)]
					models[(var,es)] = (adfq)
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			alg_num+=1

		# ADFQs - Bayesian
		for policy in update_policies:
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for var in variances:
				adfq = [brl.adfq(scene, discount, 0.0, var, init_policy=True, TH=v[1]) for i in range(Nrun)]
				[adfq[i].learning(updatePolicy=policy, actionPolicy='Bayes', actionParam = None, eval_greedy =True, updateParam=0.01) for i in range(Nrun)]
				models[(var,)] = (adfq)
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			alg_num+=1

		if False:# not(scene == 'maze'):
			# KTD-Q with Egreedy
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for var in [1.0,10.0]:
				for es in [0.05,0.1,0.15]:
					ktd = [brl.ktd_Q(scene, discount, 0.0, var, init_policy = True, TH=v[1]) for i in range(Nrun)]
					[ktd[i].learning(1, actionPolicy="egreedy", actionParam = es, eval_greedy =True) for i in range(Nrun)] #bp[scene][labels[4]]['kappa']
					models[(var,es)] = ktd
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			alg_num += 1

			# KTD-Q with Active Learning
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for var in [1.0, 10.0]:
				ktd = [brl.ktd_Q(scene, discount, 0.0, var, init_policy = True, TH=v[1]) for i in range(Nrun)]
				[ktd[i].learning(1, actionPolicy="active", actionParam=None ,eval_greedy =True) for i in range(Nrun)]
				models[(var,)] = ktd
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])

		f_rew.write(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
		f_rew.write('\n')
		f_rew.write(tabulate(result2['tab'], headers=["ALG. Eval_based"]+headers[1:], tablefmt='orgtbl'))
		f_rew.write('\n')
		f_rew.close()

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

