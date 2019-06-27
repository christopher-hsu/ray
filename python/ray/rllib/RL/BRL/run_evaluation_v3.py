import brl
import tabularRL as tq
import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

import time
import brl_util as util
import pdb
import seeding
import pickle
import argparse
import os
from tabulate import tabulate
from collections import OrderedDict
import models as MDPs

colors = ['r','k','b','g','c','m','y','burlywood','chartreuse','0.8']

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=10, help='the number of trials to repeat')
parser.add_argument('--discount', type=float, default=0.95, help= 'discount factor')
parser.add_argument('--learning_rate_init', type=float, default=0.5, help ='initial learning rate for Q-learning')
parser.add_argument('--function', type=str)
parser.add_argument('--log_dir', type=str, default='./')
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--slip', type=float, default=0.0)

args = parser.parse_args()

"""""""""
EVALUATION ADFQ-Numeric, ADFQ, KTD-Q (egreedy / behavior policy), and, Watkin's Q-learning

* ALGORITHMS :
	- ADFQ-Numeric : ADFQ with numerically computed mean and variance
	- ADFQ : ADFQ with small variance approximation
	- ADFQ-V2  : ADFQ with small variance approximation - additional veersion
	- KTD-Q		   : KTD-Q
	- Q-learning   : Q-learning 

* Action Selection
	- Off-Policy for RMSE results
	- ADFQ : Egreedy, Eg+Thompson_Sampling, Thompson_Sampling
	- KTD-Q : Egreedy, Behavior Policy
	- Q-learning : Egreedy, Boltzmann

* TASKs : CHAIN, LOOP, GRID5, GRID10, Dearden's MAZE, MiniMaze 
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
variances = [100.0]
epsilons = [0.0, 0.1, 0.2]
boltz_temp = [0.1, 0.3, 0.5]
ktd_noise = [0.0, 0.1, 1.0]
alphas = [args.learning_rate_init]#[0.1, 0.3, 0.5]
batch_sizes = [30]
noises = [1e-11]

update_policies = ['adfq', 'numeric','adfq-v2']
policy_labels = {'adfq': 'ADFQ', 'numeric': 'ADFQ-Numeric', 'adfq-v2': 'ADFQ-V2'}
labels_off = ["Q-learning"]+ [policy_labels[k] for k in update_policies] +["KTD-Q"]
labels_act = [r'Q-learning, $\epsilon$-greedy', "Q-learning, boltzmann ",] \
			+[policy_labels[k] + r', $\epsilon$-greedy' for k in update_policies] \
			+[policy_labels[k] + ', BS' for k in update_policies] \
			+[r'KTD-Q, $\epsilon$-greedy', "KTD-Q, active" ]

scene_set={'chain':(2, util.T_chain, 5*2),'loop':(2,util.T_loop,9*2), 'grid5':(4,util.T_grid5,25*4),'grid10':(4,util.T_grid10,100*4), 
		'minimaze':(4,util.T_minimaze,112*4), 'maze':(4,util.T_maze,264*4)}
scene_set = OrderedDict(sorted(scene_set.items(), key = lambda t:t[1][1]))

def off_policy_helper(scene, models, result, alg_name, del_t):
	best_key = models.keys()[np.argmin([sum([x.Q_err[-1] for x in y]) for y in models.values()])]
	result['Q_err'].append(np.array([x.Q_err for x in models[best_key]]))
	result['tab'].append([alg_name, str(round((sum(result['Q_err'][-1])/float(Nrun))[-1],2)), str(del_t)]+[str(v) for v in best_key])

def off_policy(eval_scenes=None):	
	if eval_scenes == None: 
		eval_scenes = scene_set.keys()

	headers = ["ALG. Q_err based", "RMSE mean ","Elapsed T", "HyperParam1", "HyperParam2", "HyperParam3"]
	print("Off-Policy Learning...")
	for scene in eval_scenes:
		f_rmse = open(os.path.join(args.log_dir,"rmse.txt"),"a")		
		v = scene_set[scene]
		alg_num = 0
		result = {'Q_err':[], 'tab':[]}
		print("Domain:%s with slip %.2f"%(scene, args.slip))
		actionSet = [np.random.choice(v[0],v[1]) for i in range(Nrun)]
		if MDPs.model_assign(scene).episodic: # Non-episodic
			init_means = np.ones((Nrun,))
		else:
			init_means = 1.0/(1-discount)*np.ones((Nrun,))
	
		# Q-learning (fixed learning rate)
		print(labels_off[alg_num])
		t_start = time.time()
		models = {}
		for alpha in alphas:
			algs = [tq.Qlearning(scene, alpha, discount, initQ=init_means[i]) for i in range(Nrun)]
			[algs[i].env.set_slip(args.slip) for i in range(Nrun)]
			[algs[i].learning('offline', actionSet[i], rate_decay=True) for i in range(Nrun)]
			models[(alpha,)] = algs
		off_policy_helper(scene, models, result, labels_off[alg_num], round((time.time()-t_start)/len(models) ,2))
		alg_num+=1
		print(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))

		# ADFQ 
		useScale = False
		for policy in update_policies:
			print(labels_off[alg_num])
			t_start = time.time()
			models = {}
			if args.slip == 0.0:
				algs = [brl.adfq(scene, discount, init_mean=init_means[i]) for i in range(Nrun)]
				[algs[i].env.set_slip(args.slip) for i in range(Nrun)]
				[algs[i].learning(updatePolicy=policy, actionPolicy='offline', actionParam=actionSet[i], useScale=useScale) for i in range(Nrun)]
				models[(-1,)] = algs
			else:
				for batch_size in batch_sizes:
					for noise in noises:
						algs = [brl.adfq(scene, discount, init_mean=init_means[i]) for i in range(Nrun)]
						[algs[i].env.set_slip(args.slip) for i in range(Nrun)]
						[algs[i].learning(updatePolicy=policy, actionPolicy='offline', actionParam=actionSet[i], batch_size=batch_size, noise=noise, useScale=useScale) for i in range(Nrun)]
						models[(batch_size, noise,)] = algs
			off_policy_helper(scene, models, result, labels_off[alg_num], round((time.time()-t_start)/len(models) ,2))
			print(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
			alg_num+=1
		np.save(os.path.join(args.log_dir,"err_set_"+scene),result['Q_err'])
		# KTD-Q
		if not(scene == 'maze'):
			print(labels_off[alg_num])
			t_start = time.time()
			models = {}
			for kappa in [1.0, 0.5*v[2], v[2]]:
				print("kappa %.2f"%kappa)
				algs = [brl.ktd_Q(scene, discount, init_mean = init_means[i]) for i in range(Nrun)]
				[algs[i].env.set_slip(args.slip) for i in range(Nrun)]
				[algs[i].learning(kappa = kappa, actionPolicy="offline", actionParam=actionSet[i]) for i in range(Nrun)]
				models[(kappa,)] = algs
			off_policy_helper(scene, models, result, labels_off[alg_num], round((time.time()-t_start)/len(models) ,2))
			alg_num+=1
		print(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
		np.save(os.path.join(args.log_dir,"err_set_"+scene),result['Q_err'])
		f_rmse.write(scene+"\n")
		f_rmse.write(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
		f_rmse.write('\n')

def action_policy_helper(scene, models, result, alg_name, del_t, reward_based =True):
	if reward_based:
		rewards = [sum([x.get_total_reward() for x in y])/float(Nrun) for y in models.values()]
		best_key = models.keys()[np.argmax(rewards)]
	else:
		max_v = -1000.0
		for (k,vec) in models.items():
			#val = np.mean([np.mean(x.test_rewards[-20:]) for x in vec])
			val = np.mean([np.sum(x.test_rewards) for x in vec])
			if val > max_v:
				max_v = val
				best_key = k

	total_rewards = np.array([x.get_total_reward() for x in models[best_key]])
	result['reward'].append(np.array([x.test_rewards for x in models[best_key]]))
	result['count'].append(np.array([x.test_counts for x in models[best_key]]))
	try:
		result['Q'].append(np.array([x.means.flatten() for x in models[best_key]]))
	except:
		result['Q'].append(np.array([x.Q.flatten() for x in models[best_key]]))
	result['tab'].append([alg_name, str(round(sum(total_rewards)/float(Nrun),2)), str(np.std(total_rewards))] + [str(del_t)]+[str(v) for v in best_key])
	save_to_file(scene, result, reward_based = reward_based)

def action_policy(eval_scenes=None, useParam=True):
	if eval_scenes == None: 
		eval_scenes = scene_set.keys()
	headers = ["ALG. Rew_based", "Tot rewards", "SD","Elapsed T", "HyperParam1", "HyperParam2", "HyperParam3"]
	print("Aciton Policy Learning...")
	init_mean = 3.0
	test_rewards_set = {}
	for scene in eval_scenes:
		v = scene_set[scene] 
		alg_num = 0
		f_rew = open(os.path.join(args.log_dir,"total_rewards_range.txt"),"a")
		f_rew.write("\n"+scene+"\n")
		print("Domain:" + scene)
		#f_rew.close()
		result = {'reward':[], 'count':[], 'Q':[], 'tab':[]}
		result2 = {'reward':[], 'count':[], 'Q':[], 'tab':[]}
		
		# Q-learning_fixed, egreedy
		print(labels_act[alg_num])
		t_start = time.time()
		models = {}
		tmp = {}
		for alpha in alphas:
			for es in epsilons: 
				peq = [tq.Qlearning(scene, alpha, discount, initQ=init_mean, TH=v[1]) for i in range(Nrun)]
				[peq[i].env.set_slip(args.slip) for i in range(Nrun)]
				[peq[i].learning('egreedy',es, eval_greedy =True, rate_decay=True) for i in range(Nrun)]
				models[(alpha,es)] = peq
				tmp[(alpha, es)] = [x.test_rewards for x in peq]
		test_rewards_set[labels_act[alg_num]] = tmp
		elapsed_t = round( (time.time()-t_start)/len(models) ,2)
		action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
		action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
		print(result['tab'][-1])
		alg_num += 1
		
		# Q-learning_fixed, boltzmann
		print(labels_act[alg_num])
		t_start = time.time()
		models = {}
		tmp = {}
		for alpha in alphas:
			for tau in boltz_temp:
				peq = [tq.Qlearning(scene, alpha, discount, initQ=init_mean, TH=v[1]) for i in range(Nrun)]
				[peq[i].env.set_slip(args.slip) for i in range(Nrun)]
				[peq[i].learning('softmax',tau, eval_greedy =True, rate_decay=True) for i in range(Nrun)]
				models[(alpha,tau,)] = peq
				tmp[(alpha, tau)] = [x.test_rewards for x in peq]
		test_rewards_set[labels_act[alg_num]] = tmp
		elapsed_t = round( (time.time()-t_start)/len(models) ,2)
		action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
		action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
		print(result['tab'][-1])
		alg_num +=1

		#model_set = {}
		# ADFQs - Egreedy
		for policy in update_policies:
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			tmp = {}
			if args.slip == 0.0:
				for es in epsilons:
					for var in variances:
						adfq = [brl.adfq(scene, discount, init_mean=init_mean, init_var=var, TH=v[1]) for i in range(Nrun)]
						[adfq[i].env.set_slip(args.slip) for i in range(Nrun)]
						[adfq[i].learning(updatePolicy=policy, actionPolicy='egreedy', actionParam = es, eval_greedy =True) for i in range(Nrun)]
						models[(es, var)] = adfq
						tmp[(es, var)] = [x.test_rewards for x in adfq]
			else:
				for noise in noises:
					for batch_size in  batch_sizes: 
						for es in epsilons:
							adfq = [brl.adfq(scene, discount, init_mean=init_mean, TH=v[1]) for i in range(Nrun)]
							[adfq[i].env.set_slip(args.slip) for i in range(Nrun)]
							[adfq[i].learning(updatePolicy=policy, actionPolicy='egreedy', actionParam = es, eval_greedy =True, 
								noise = noise, batch_size=batch_size) for i in range(Nrun)]
							models[(noise, batch_size, es,)] = adfq
							tmp[(noise, batch_size, es,)] = [x.test_rewards for x in adfq]
			test_rewards_set[labels_act[alg_num]] = tmp
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			#model_set[(policy, 'egreedy')] = models
			alg_num += 1

		# ADFQs - Thompson_Sampling
		for policy in update_policies:

			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			tmp = {}
			if args.slip == 0.0:
				for var in variances:
					adfq = [brl.adfq(scene, discount, init_mean=init_mean, init_var=var, TH=v[1]) for i in range(Nrun)]
					[adfq[i].env.set_slip(args.slip) for i in range(Nrun)]
					[adfq[i].learning(updatePolicy=policy, actionPolicy='ts', actionParam = None, eval_greedy =True) for i in range(Nrun)]
					models[(var,)] = adfq
					tmp[(var,)] = [x.test_rewards for x in adfq]
			else:
				for noise in noises:
					for batch_size in  batch_sizes: 
						adfq = [brl.adfq(scene, discount, init_mean=init_mean, TH=v[1]) for i in range(Nrun)]
						[adfq[i].env.set_slip(args.slip) for i in range(Nrun)]
						[adfq[i].learning(updatePolicy=policy, actionPolicy='ts', actionParam = None, eval_greedy =True, 
							noise = noise, batch_size=batch_size) for i in range(Nrun)]
						models[(noise, batch_size,)] = adfq
						tmp[(noise, batch_size,)] = [x.test_rewards for x in adfq]
			test_rewards_set[labels_act[alg_num]] = tmp
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			pickle.dump(test_rewards_set, open(os.path.join(args.log_dir, "set_rewards.pkl"), "wb"))
			alg_num+=1

		if False:#not(scene == 'maze'):
			# KTD-Q with Egreedy
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			tmp = {}
			for kappa in [1.0, 0.5*v[2], v[2]]:
				for es in epsilons:
					ktd = [brl.ktd_Q(scene, discount, init_mean=init_mean, TH=v[1]) for i in range(Nrun)]
					[ktd[i].env.set_slip(args.slip) for i in range(Nrun)]
					[ktd[i].learning(kappa=kappa, actionPolicy="egreedy", actionParam = es, eval_greedy =True) for i in range(Nrun)] #bp[scene][labels[4]]['kappa']
					models[(kappa,es)] = ktd
					tmp[(kappa,es)] = [x.test_rewards for x in ktd]
			test_rewards_set[labels_act[alg_num]] = tmp
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])
			alg_num += 1

			# KTD-Q with Active Learning
			print(labels_act[alg_num])
			t_start = time.time()
			models = {}
			for kappa in [1.0, 0.5*v[2], v[2]]:
				ktd = [brl.ktd_Q(scene, discount, init_mean=init_mean, TH=v[1]) for i in range(Nrun)]
				[ktd[i].env.set_slip(args.slip) for i in range(Nrun)]
				[ktd[i].learning(kappa=kappa, actionPolicy="active", actionParam=None ,eval_greedy =True) for i in range(Nrun)]
				models[(kappa,)] = ktd
				tmp[(kappa,)] = [x.test_rewards for x in ktd]
			test_rewards_set[labels_act[alg_num]] = tmp
			elapsed_t = round( (time.time()-t_start)/len(models) ,2)
			action_policy_helper(scene, models, result, labels_act[alg_num], elapsed_t)
			action_policy_helper(scene, models, result2, labels_act[alg_num], elapsed_t, reward_based=False)
			print(result['tab'][-1])

		f_rew.write(tabulate(result['tab'], headers=headers, tablefmt='orgtbl'))
		f_rew.write('\n')
		f_rew.write(tabulate(result2['tab'], headers=["ALG. Eval_based"]+headers[1:], tablefmt='orgtbl'))
		f_rew.write('\n')
		f_rew.close()
		pickle.dump(test_rewards_set, open(os.path.join(args.log_dir, "set_rewards.pkl"), "wb"))

def save_to_file(scene, dataset, reward_based=True):
	for (k,v) in dataset.items():
		if k != 'tab':
			if reward_based:
				np.save(os.path.join(args.log_dir,k+"_"+scene), np.array(v))
			else:
				np.save(os.path.join(args.log_dir,k+"_v2_"+scene), np.array(v))

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
		if len(data) > 4:
			ids = [0,2,1,4]
			#data = data[ids,:,:]
		labels = labels_off
		xids = range(0,200,2)
		pic_name = 'rmse'
		pdb.set_trace()
		#util.plot_IQR(100, data[:,:,xids], labels, 'Learning Step', 'RMSE', x_vals = range(0,T,T/100), save=True, pic_name=pic_name)
		util.plot_sd(100, data, labels, 'Learning Step', 'RMSE', x_vals = range(0,T,T/100), save=False, pic_name=pic_name)
	else:
		pdb.set_trace()
		if evaluation_ids:
			ids = evaluation_ids
		else:
			ids = range(len(data))
		
		labels_ids = ids
		pic_name = 'evaluation '+args.scene
		interval = T/100
		util.plot_sd(T, data[ids,:,:], [labels_act[i] for i in labels_ids], 'Learning Step', 'Average Episode Rewards', 
			x_vals=range(0,T,interval) ,legend=(True,'lower right'), save=False, pic_name = pic_name, smoothed=True)
		#util.plot_smoothing(T, data[ids,:,:], [labels_act[i] for i in labels_ids], 
		#	'Learning Step', 'Average Episode Rewards', legend=(True,'lower right'), save=False, pic_name = pic_name)

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
		analysis(os.path.join(args.log_dir, 'err_set_'+args.scene+'.npy'), 'off_policy')
	elif args.function == 'analysis_action':
		analysis(os.path.join(args.log_dir, 'reward_v2_'+args.scene+'.npy'), 'action')
	else:
		raise ValueError('%s is not available'%args.function)

