""" 
<ADFQ and other BRL implementations>

Author: Heejin Chloe Jeong (chloe.hjeong@gmail.com)
Affiliation: University of Pennsylvania
"""

import models as mdl
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
import time
import brl_util as util
import util_GPTD
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import pdb
import random
import seeding
import gym
import copy

class BRL(object):
	def __init__(self,scene,discount,init_policy, TH, useGym, memory_size):
		self.scene = scene
		self.obj = gym.make(scene) if useGym else mdl.model_assign(scene)
		self.discount = discount
		self.states = []
		self.actions = []
		self.rewards = []
		self.init_policy = init_policy
		self.np_random,_  = seeding.np_random(None)
		self.test_counts = []
		self.test_rewards = []
		self.Q_err = []
		self.visits = np.zeros((self.obj.snum,self.obj.anum))
		self.useGym = useGym
		self.replayMem = []
		self.memory_size = memory_size
		if not(TH==None):
			self.obj.set_time(TH)

	def get_visits(self):
		return self.visits

	def get_total_reward(self):
		return sum(self.rewards)

	def err(self):
		mean_eval = np.reshape(self.means, (self.obj.snum, self.obj.anum) )
		return np.sqrt(np.mean((self.Q_target[self.obj.eff_states,:] - mean_eval[self.obj.eff_states,:])**2))

	def draw(self,s,a,t,r):
			
		print("s:",s,"t:",t,"Reward:",r,"Total Reward:",sum(self.rewards)+r)
		self.obj.plot(s,a)
		print("=====")
		time.sleep(0.5)	

	def greedy_policy(self, get_action_func, episodic,
			step_bound = None, it_bound = util.EVAL_RUNS):
		"""
			lambda x : get_action_eps(x,kappa,epsilon)
			or
			lambda x : get_action_egreedy(x,epsilon)
		"""
		if step_bound is None:
			step_bound = self.obj.timeH/util.EVAL_STEPS
		counts = [] 
		rewards = []
		it = 0 
		while(it<it_bound):
			t = 0
			#np_random_local, _  = seeding.np_random(None)
			state = self.obj.reset(self.np_random) #np_random_local.choice(range(self.obj.anum))
			reward = 0.0
			done = False
			while((not done) and (t<step_bound)):
				action = get_action_func(state)
				r, state_n, done = self.obj.observe(state,action,self.np_random)
				state = state_n
				reward += r
				t +=1
			rewards.append(reward)
			counts.append(t)
			it += 1
		return np.mean(counts), np.mean(rewards), np.std(counts), np.std(rewards)

	def test(self, eps = util.EVAL_EPS, mean= None, var = None):
		if mean:
			self.means = mean
		if var:
			self.vars = var
		count, rew, _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, eps), self.obj.episodic)
		return count, rew

	def init_params(self):
		s = self.obj.reset(self.np_random)
		while(True):
			a = self.np_random.choice(range(self.obj.anum))
			r, s_n, done = self.obj.observe(s,a,self.np_random)
			if r > 0: # First nonzero reward
				if self.obj.episodic:
					self.means = r*np.ones(self.means.shape,dtype=np.float)
				else:
					self.means = r/(1-self.discount)*np.ones(self.means.shape, dtype=np.float)
				break
			else:
				if done:
					self.means = np.zeros(self.means.shape,dtype=np.float)
					break
				s = s_n
	def store(self, causality_tup):
		if (len(self.replayMem) == self.memory_size):
			self.replayMem.pop(0)
			self.replayMem.append(causality_tup)
		else:
			self.replayMem.append(causality_tup)

	def get_batch(self, batch_size):
		minibatch = {'state':[], 'action':[], 'reward':[], 'state_n':[], 'terminal':[]}
		for _ in range(batch_size):
			d = self.replayMem[random.randint(0,len(self.replayMem)-1)]
			for (k,v) in minibatch.items():
				v.append(d[k])
		return minibatch

class adfq(BRL):
	def __init__(self,scene, discount, means, variances, init_policy=False, TH=None, useGym=False, memory_size = 50):

		BRL.__init__(self, scene, discount, init_policy, TH, useGym=useGym, memory_size = memory_size)
	
		self.means = means*np.ones((self.obj.snum,self.obj.anum),dtype=np.float)
		if init_policy:
			self.init_params()
		self.vars = variances*np.ones((self.obj.snum,self.obj.anum),dtype=np.float)
		self.step = 0
		
		self.mean_history = []
		self.var_history = []	

	def learning(self, updatePolicy, actionPolicy, actionParam, eval_greedy = False, 
				draw = False, varTH = 1e-10, updateParam=None, asymptotic=False, useRatio=False, useScale=False, noise=0.0):
		""""
		INPUT:
			updatePolicy: 'Numeric' for ADFQ-Numeric, 'Approx' for ADFQ-Approx, 'Fixed' for fixed variance update
			eval_greedy: True or 1, if you want to evaluate greedily during the learning process
			draw: True or 1, if you want visualization
		"""
		if len(self.rewards)==self.obj.timeH:
			print("The object has already learned")
			return None

		if (actionPolicy == 'offline') and (len(actionParam) != self.obj.timeH):
			raise ValueError('The given action trajectory does not match with the number of learning steps.')
		self.Q_target = np.array(self.obj.optQ(self.discount))
		s = 0
		self.varTH = varTH
		log_scale = 0.0
		enable_ratio = False

		while(self.step < self.obj.timeH):
			if self.step%(self.obj.timeH/200) == 0:
				self.Q_err.append(self.err())
				self.mean_history.append(copy.deepcopy(self.means))
				self.var_history.append(copy.deepcopy(self.vars))

			a = self.action_selection(s, actionPolicy, actionParam)
			self.states.append(s)
			self.actions.append(a)
			self.visits[s][a] += 1

			# Observation
			r, s_n, done = self.obj.observe(s,a,self.np_random)
			if updatePolicy == 'Numeric' :
				new_mean, new_var, _ = util.posterior_numeric( self.means[s_n], self.vars[s_n], self.means[s][a], 
					self.vars[s][a], r, self.discount, done, varTH = self.varTH, noise=noise)	

			elif (updatePolicy == 'SoftApprox'):
				new_mean, new_var, _ = util.posterior_soft_approx(self.means[s_n], self.vars[s_n], 
						self.means[s][a], self.vars[s][a], r, self.discount, done, varTH =self.varTH, matching=True, 
						ratio=enable_ratio, scale=useScale, c_scale=np.exp(log_scale, dtype=np.float64), asymptotic=asymptotic, noise=noise)

			elif (updatePolicy == 'Approx'):
				new_mean,new_var, _ = util.posterior_approx( self.means[s_n], self.vars[s_n], self.means[s][a], 
					self.vars[s][a], r, self.discount, done, varTH = self.varTH, asymptotic=asymptotic, noise=noise)	
				
			elif updatePolicy == 'All' :
				new_mean_numeric, new_var_numeric, _ = util.posterior_numeric( self.means[s_n], self.vars[s_n], 
						self.means[s][a], self.vars[s][a], r, self.discount, done)	
				new_mean, new_var, _ = util.posterior_soft_approx(self.means[s_n], self.vars[s_n], 
						self.means[s][a], self.vars[s][a], r, self.discount, done, matching=True)
				new_mean_approx, new_var_approx, _ = util.posterior_approx( self.means[s_n], self.vars[s_n], 
					self.means[s][a], self.vars[s][a], r, self.discount, done, varTH = self.varTH, asymptotic=asymptotic)				
				if abs(new_mean - new_mean_numeric) > 5.0 or abs(new_mean - new_mean_approx) > 5.0:
					util.plot_posterior_all(self.means[s_n], self.vars[s_n], self.means[s][a], self.vars[s][a], r, self.discount, done)
					pdb.set_trace()
			else:
				raise ValueError("No such update policy")

			self.means[s][a] = new_mean
			if new_var <= 0.0:
				print("Updated variance is negative")
				pdb.set_trace()
			self.vars[s][a] = new_var #np.maximum(self.varTH, new_var)

			if enable_ratio or useScale:
				delta =  np.log(np.mean(self.vars[self.obj.eff_states,:]))
				log_scale = np.maximum( -500.0, log_scale + delta)
				self.vars[self.obj.eff_states,:] = np.exp(np.log(self.vars[self.obj.eff_states,:]) - delta, dtype = np.float64)
				
			if useRatio and not(enable_ratio) and (np.mean(self.vars[self.obj.eff_states,:]) <= varTH*1000):
				print("Ratio Update from step %d"%self.step)
				log_scale = np.log(np.mean(self.vars[self.obj.eff_states,:]))
				self.vars[self.obj.eff_states,:] = np.exp(np.log(self.vars[self.obj.eff_states,:]) - log_scale, dtype = np.float64)
				enable_ratio = True

			self.rewards.append(r)
			if draw:
				self.draw(s,a,self.step,r)

			if eval_greedy and ((self.step+1)%(self.obj.timeH/util.EVAL_NUM) == 0):
				count, rew , _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS), self.obj.episodic)
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			s = 0 if done else s_n
			self.step += 1
		print(log_scale)
		self.mean_history = np.array(self.mean_history)
		self.var_history = np.array(self.var_history)

	def action_selection(self, state, action_policy, param):
		"""Action Selection
			1: e-greedy action
			3: e-greedy Bayesian action - greedy and Bayesian sampling with epsilon probability
			4: Bayesian sampling 
			5: uniform random
			6: offline
			7: VPI (Dearden's et al, 1998)
		"""		
		if action_policy == 'egreedy':
			action = self.get_action_egreedy(state,param)
		elif action_policy == 'semi-Bayes':
			if self.np_random.rand(1)[0] < param:
				action = self.get_action_Bayesian(state)
			else:
				action = np.argmax(self.means[state])		
		elif action_policy == 'Bayes': 
			action = self.get_action_Bayesian(state)
		elif action_policy == 'random':
			action = self.np_random.choice(range(self.obj.anum))
		elif action_policy == 'offline':
			action = param[self.step]
		elif action_policy == 'vpi':
			action = self.vpi(state)

		return action

	def get_action_Bayesian(self,state):
		if (self.vars[state] < self.varTH).any():
			return np.argmax(self.means[state])
		else:
			if len(set(self.means[state]))==1:
				return int(self.np_random.choice(range(self.obj.anum)))
			else:
				tmp  = self.np_random.normal(self.means[state],np.sqrt(self.vars[state]))
				return np.argmax(tmp)

 	def get_action_egreedy(self,state,epsilon):
 		if self.np_random.rand(1)[0] < epsilon: 
			return int(self.np_random.choice(range(self.obj.anum)))
		else:
			return np.argmax(self.means[state])

	def get_action_eB(self,state,epsilon):
		# epsilon-greedy inspired
		if self.np_random.rand(1)[0] > (1-epsilon): 
			return int(self.np_random.choice(range(self.obj.anum)))
		else:
			if (self.vars[state] < self.varTH).any():
				return np.argmax(self.means[state])
			if len(set(self.means[state]))==1:
				return  int(self.np_random.choice(range(self.obj.anum)))
			else:
				tmp  = self.np_random.normal(self.means[state],np.sqrt(self.vars[state]))
				return np.argmax(tmp)
	def vpi(self,state):
		#pdb.set_trace()
		vpi_vals = np.zeros((self.obj.anum,),dtype=np.float32)
		id_sorted = np.argsort(self.means[state,:])
		if self.means[state,id_sorted[-1]] == self.means[state,id_sorted[-2]]:
			if np.random.rand(1)[0] < 0.5:
				tmp = id_sorted[-1]
				id_sorted[-1] = id_sorted[-2]
				id_sorted[-2] = tmp
		# a = a_1
		best_a = id_sorted[-1]
		mu = self.means[state, best_a]
		sig  = np.sqrt(self.vars[state, best_a])
		vpi_vals[best_a] = self.means[state,id_sorted[-2]]* norm.cdf(self.means[state,id_sorted[-2]], mu, sig) \
			- mu*norm.cdf(self.means[state,id_sorted[-2]],mu, sig) + sig*sig*norm.pdf(self.means[state,id_sorted[-2]], mu, sig)
					#- mu + sig*sig*norm.pdf(self.means[state,id_sorted[-2]], mu, sig)/max(0.0001,norm.cdf(self.means[state,id_sorted[-2]],mu, sig))
					
		for a_id in id_sorted[:-1]:
			mu = self.means[state, a_id]
			sig = np.sqrt(self.vars[state, a_id])
			vpi_vals[a_id] = mu*(1-norm.cdf(self.means[state,best_a], mu, sig)) + sig*sig*norm.pdf(self.means[state, best_a], mu, sig) \
				- self.means[state, best_a]*(1-norm.cdf(self.means[state,best_a], mu, sig))
			#mu + sig*sig*norm.pdf(self.means[state, best_a], mu, sig)/max(0.0001,(1-norm.cdf(self.means[state,best_a], mu, sig))) \
					
		a_orders = np.argsort(vpi_vals)
		if vpi_vals[a_orders[-1]] == vpi_vals[a_orders[-2]]:
			return np.random.choice(a_orders[-2:])
		else:
			return np.argmax(vpi_vals+self.means[state,:])

class adfq_log(BRL):
	def __init__(self,scene, discount, means, variances, init_policy=False, TH=None, useGym=False):

		BRL.__init__(self, scene, discount, init_policy, TH, useGym=useGym)
	
		self.means = means*np.ones((self.obj.snum,self.obj.anum),dtype=np.float)
		if init_policy:
			self.init_params()
		self.logvars = (np.log(variances))*np.ones((self.obj.snum,self.obj.anum),dtype=np.float)
		self.step = 0
		
		self.mean_history = [copy.deepcopy(self.means)]
		self.var_history = [copy.deepcopy(self.logvars)]	

	def learning(self, updatePolicy, actionPolicy, actionParam, eval_greedy = False, 
				draw = False, logEps = -1e+20, logvarTH = -100.0):
		""""
		INPUT:
			updatePolicy: 'Numeric' for ADFQ-Numeric, 'Approx' for ADFQ-Approx, 'Fixed' for fixed variance update
			eval_greedy: True or 1, if you want to evaluate greedily during the learning process
			draw: True or 1, if you want visualization
		"""
		if len(self.rewards)==self.obj.timeH:
			print("The object has already learned")
			return None

		if (actionPolicy == 'offline') and (len(actionParam) != self.obj.timeH):
			raise ValueError('The given action trajectory does not match with the number of learning steps.')
		self.Q_target = np.array(self.obj.optQ(self.discount))
		s = 0
		while(self.step < self.obj.timeH):
			self.Q_err.append(self.err())
			a = self.action_selection(s, actionPolicy, actionParam)
			self.states.append(s)
			self.actions.append(a)
			self.visits[s][a] += 1

			# Observation
			r, s_n, done = self.obj.observe(s,a,self.np_random)
			if (updatePolicy == 'Approx'):# or (updatePolicy == 'Numeric' and (self.logvars[s_n] < self.varTH).any()):
				new_mean,new_logvar, _ = util.posterior_approx_log_v2( self.means[s_n], self.logvars[s_n], 
					self.means[s][a], self.logvars[s][a], r, self.discount, done, logEps = logEps, logvarTH=logvarTH)	
				#m, v, _ = util.posterior_approx( self.means[s_n], np.exp(self.logvars[s_n]), 
				#	self.means[s][a], np.exp(self.logvars[s][a]), r, self.discount, done, varTH=1e-10)
				#if abs(new_mean - m)> 1.0 or abs(new_logvar - np.log(v))>3.0:
				#	pdb.set_trace()

			else:
				raise ValueError("No such update policy")

			self.means[s][a] = new_mean
			self.logvars[s][a] = new_logvar

			self.rewards.append(r)
			self.mean_history.append(copy.deepcopy(self.means))
			self.var_history.append(copy.deepcopy(self.logvars))

			if draw:
				self.draw(s,a,self.step,r)

			if eval_greedy and ((self.step+1)%(self.obj.timeH/util.EVAL_NUM) == 0):
				count, rew, _, _= self.greedy_policy(lambda x : self.get_action_egreedy(x, util.EVAL_EPS), self.obj.episodic)
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			s = 0 if done else s_n
			self.step += 1

		self.mean_history = np.array(self.mean_history)
		self.var_history = np.array(self.var_history)

	def action_selection(self, state, action_policy, param):
		"""Action Selection
			1: e-greedy action
			3: e-greedy Bayesian action - greedy and Bayesian sampling with epsilon probability
			4: Bayesian sampling 
			5: uniform random
			6: offline
			7: VPI (Dearden's et al, 1998)
		"""		
		if action_policy == 'egreedy':
			action = self.get_action_egreedy(state,param)
		elif action_policy == 'semi-Bayes':
			if self.np_random.rand(1)[0] < param:
				action = self.get_action_Bayesian(state)
			else:
				action = np.argmax(self.means[state])		
		elif action_policy == 'Bayes': 
			action = self.get_action_Bayesian(state)
		elif action_policy == 'random':
			action = self.np_random.choice(range(self.obj.anum))
		elif action_policy == 'offline':
			action = param[self.step]
		elif action_policy == 'vpi':
			action = self.vpi(state)

		return action

	def get_action_Bayesian(self,state):
		if len(set(self.means[state]))==1:
			return int(self.np_random.choice(range(self.obj.anum)))
		else:
			tmp  = self.np_random.normal(self.means[state],np.sqrt(np.exp(self.logvars[state])))
			return np.argmax(tmp)

 	def get_action_egreedy(self,state,epsilon):
 		if self.np_random.rand(1)[0] < epsilon: 
			return int(self.np_random.choice(range(self.obj.anum)))
		else:
			return np.argmax(self.means[state])

	def get_action_eB(self,state,epsilon):
		# epsilon-greedy inspired
		if self.np_random.rand(1)[0] > (1-epsilon): 
			return int(self.np_random.choice(range(self.obj.anum)))
		else:
			if len(set(self.means[state]))==1:
				return  int(self.np_random.choice(range(self.obj.anum)))
			else:
				tmp  = self.np_random.normal(self.means[state],np.sqrt(np.exp(self.logvars[state])))
				return np.argmax(tmp)

class ktd_Q(BRL): 
	def __init__(self,scene, discount, means=0.0, variances =10.0, init_policy=False, TH=None, useGym=False):
		BRL.__init__(self, scene, discount, init_policy, TH, useGym=useGym)

		self.phi_func = self.obj.phi[0]
		self.dim = self.obj.phi[1]
		self.means = means*np.ones(self.dim,dtype=np.float) # row vector
		if init_policy:
			self.init_params()
		self.cov = variances*np.eye(self.dim)
		self.eval = []
		self.step = 0
		self.t_history = []  
		self.avgmeans = []
		self.avgcovs = []

	def update(self, state, action, state_n, reward, done, epsilon):
		# Prediction Step
		pre_mean = self.means
		pre_cov = self.cov + self.eta*np.eye(self.dim)

		"""Sigma Point Computation:
		"""
		sig_th, W = sigma_points(pre_mean,pre_cov,self.kappa)
		#sig_th02, W02 = sample_sigma_points(pre_mean, pre_cov, kappa)
		#diff = np.sqrt(np.mean((sig_th-sig_th02)**2) + np.mean((W-W02)**2))
		#if diff > epsilon:
		#	pdb.set_trace()

		#sig_R = np.matmul(sig_th, self.phi_func(state,action)) \
		#		 - int(not done)*self.discount*np.max([np.matmul(sig_th, self.phi_func(state_n, b)) for b in range(self.obj.anum)], axis=0)
		sig_R = np.matmul(sig_th, self.phi_func(state,action)) \
				 - self.discount*np.max([np.matmul(sig_th, self.phi_func(state_n, b)) for b in range(self.obj.anum)], axis=0)
		r_est = np.dot(W, sig_R)     
		cov_th_r = np.matmul(W*(sig_R-r_est),(sig_th-pre_mean))
		cov_r = self.obs_noise + np.dot(W, (sig_R-r_est)**2)

		"""Correction Step:
		"""
		K = cov_th_r/cov_r
		self.means = pre_mean + K*(reward-r_est)
		self.cov = pre_cov - cov_r*np.outer(K,K)
		self.cov = 0.5*self.cov +0.5*np.transpose(self.cov) + epsilon*np.eye(self.dim)
		self.avgmeans.append((self.means))
		self.avgcovs.append((self.cov))

	def learning(self,kappa, actionPolicy, actionParam, eta=0.0, obs_noise=1.0, eval_greedy=False, draw = False):
		if len(self.eval)==self.obj.timeH:
			print("The object has already learned")
			return None
		self.Q_target = np.array(self.obj.optQ(self.discount))
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		state = self.obj.reset(self.np_random)
		t=0 # This is "step" in Inv_pendulum and self.step is episode.
		while( self.step < self.obj.timeH):
			if actionPolicy == "active":
				action = self.active_learning(state,kappa)
			elif actionPolicy == "egreedy":
				action = self.get_action_eps(state, kappa, actionParam)
			elif actionPolicy == "offline":
				action = actionParam[self.step]
			elif actionPolicy == "uniform":
				action = self.np_random.choice(range(self.obj.anum))
			else:
				print("You must choose between egreedy, active, or offline for the action selection.")
				break  
			reward, state_n, done = self.obj.observe(state,action,self.np_random)          
			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			if draw:
				self.draw(state,action,t,r)
			
			self.visits[state][action] += 1
			self.Q_err.append(self.err())
			if eval_greedy and ((self.step+1)%(self.obj.timeH/util.EVAL_NUM) == 0):
				count, rew, _, _= self.greedy_policy(lambda x : self.get_action_eps(x, kappa, util.EVAL_EPS), self.obj.episodic)
				self.test_counts.append(count)
				self.test_rewards.append(rew)
			state = self.obj.reset(self.np_random) if done else state_n  
			self.step += 1

	def learning_cartpole(self,kappa, eta=0.0, obs_noise=1.0, epsilon = 1e-05):
		assert(self.obj.name == 'inv_pendulum')

		state = self.obj.reset(self.np_random)
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		step = 0 
		episode = 0
		while(episode<self.obj.timeH):
			action = np.random.choice(self.obj.anum,)
			reward, state_n, done = self.obj.observe(state,action,self.np_random) 
			self.update(state, action, state_n, reward, done, epsilon = epsilon)

			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			state = state_n
			step += 1
			if done or (step > self.obj.step_bound):
				self.t_history.append(step)
				state = self.obj.reset(self.np_random)
				if episode%50 == 0:
					count, rew, count_sd, _ = self.greedy_policy(lambda x : self.get_action_eps(x, kappa, 0.0), self.obj.episodic, self.obj.step_bound, 100)
					self.test_counts.append(count)
					self.test_rewards.append(rew)
					print("After %d steps, Episode %d : %.2f, SD: %.2f"%(step, episode, count, count_sd))
				episode += 1
				step = 0

	def learning_cartpole_gym(self,kappa, eta=0.0, obs_noise=1.0):
		env = gym.make('CartPole-v0')
		state = env.reset()
		self.kappa = float(kappa)
		self.eta = eta
		self.obs_noise = obs_noise
		step = 0 
		episode = 0
		it_bound = 100
		while(episode<self.obj.timeH):
			action = np.random.choice(self.obj.anum,)
			env.render()
			state_n, reward, done, _ = env.step(action)
			self.update(state[-2:], action, state[-2:], reward, done)

			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)
			state = state_n
			step += 1
			if done or (step > self.obj.step_bound):
				self.t_history.append(step)
				state = env.reset()
				if episode%50 == 0:
					test_env = gym.make('CartPole-v0')
					step_bound = self.obj.step_bound
					t_total, reward_total, it = 0,0,0
					while(it<it_bound):
						t = 0
						s_test = test_env.reset() #np_random_local.choice(range(self.obj.anum))
						r_test = 0.0
						done = False
						while((not done) and (t<step_bound)):
							a_test = np.argmax([np.dot(self.means, self.phi_func(s_test[-2:], a)) for a in range(self.obj.anum)])
							sn_test, r, done, _ = test_env.step(a_test)
							s_test = sn_test
							r_test += r
							t +=1
						reward_total += r_test
						t_total += t
						it += 1
					self.test_counts.append(t_total/float(it_bound))
					self.test_rewards.append(reward_total/float(it_bound))
					print("After %d steps, Episode %d : %d"%(step, episode, self.test_counts[-1]))
				episode += 1
				step = 0				
	
	def get_action_eps(self,state,kappa,eps):
		if self.np_random.rand() < eps:
			return self.np_random.choice(range(self.obj.anum))
		else:
			Q = [np.dot(self.means, self.phi_func(state, a)) for a in range(self.obj.anum)]
			return np.argmax(Q)

	def active_learning(self, state, kappa):
		sig_th, W = sigma_points(self.means, self.cov, kappa)
		if sig_th is None:
			return None
		Q_mean=[np.dot(W,np.matmul(sig_th, self.phi_func(state,a))) for a in range(self.obj.anum)]
		Q_var =[np.dot(W,(np.matmul(sig_th, self.phi_func(state,a)) - Q_mean[a])**2) for a in range(self.obj.anum)]
		rand_num = np.random.rand(1)[0] * sum(np.sqrt(Q_var))
		cumsum = 0
		for (i,v) in enumerate(np.sqrt(Q_var)):
			cumsum += v
			if rand_num <= cumsum:
				action = i
				break
		return action

	def get_total_reward(self):
		return sum(self.rewards)

	def get_visits(self):
		return self.visits

def sample_sigma_points(mean, variance, kappa):
    n = len(mean)
    X = np.empty((2 * n + 1, n))
    X[:, :] = mean[None, :]
    C = np.linalg.cholesky((kappa + n) * variance)
    for j in range(n):
        X[j + 1, :] += C[:, j]
        X[j + n + 1, :] -= C[:, j]
    W = np.ones(2 * n + 1) * (1. / 2 / (kappa + n))
    W[0] = (kappa / (kappa + n))
    return X, W

def sigma_points(mean, cov_in, k):
	cov = copy.deepcopy(cov_in)
	n = np.prod(mean.shape)
	count = 0
	#while(not(np.all(np.linalg.eigvals(cov)>0))) :
	#	cov += 0.1*np.eye(n)
	#	count+=1
	#	if count > 2000:
	#		print("Matrix is not positive definite")
	#		pdb.set_trace()
	chol_t = (cholesky((n+k)*cov)).T # array form cov 
	m = np.reshape(mean, (n,1))
	sigs = np.concatenate((m, m+chol_t),axis=1)
	sigs = np.concatenate((sigs, m-chol_t),axis=1)
	#sigs = np.zeros((2 * n + 1, n))
	#sigs[0,:] = mean
	#for j in range(n):
	#	sigs[j+1,:] = mean + chol_t[:,j]
	#	sigs[j+1+n,:] =mean - chol_t[:,j]

	W = 0.5/(k+n)*np.ones(n*2+1)
	W[0] = k / float(k + n)

	return sigs.T, W

class Dearden_BQL(BRL):
	"""
	 	Bayesian Q-learning of Dearden's et al. 
	 	Experimental. Not completed.
	"""
	def __init__(self, scene, discount, init_param = (0.0,1.0,1.1,20), init_policy=False, TH=None):
		BRL.__init__(self, scene, discount, init_policy, TH)
		self.ng_param = np.ones((self.obj.snum, self.obj.anum, 4), dtype=np.float)
		for i in range(4):
			self.ng_param[:,:,i] = init_param[i]
		self.step = 0

	def learning(self, actionPolicy, actionParam, eval_greedy=False):
		from scipy.special import digamma
		from scipy.stats import t as studentTdist

		state = self.obj.reset(self.np_random)
		deltaR = 0.1
		deltaM = 0.2
		deltaT = 0.02
		epsilon = 0.0001
		while(self.step <= self.obj.timeH):
			if actionPolicy == 'egreedy':
				if self.np_random.rand() < actionParam:
					action = self.np_random.choice(range(self.obj.anum))
				else:
					action = np.argmax(self.ng_param[state,:,0])
			elif actionPolicy == 'Q-sample':
				p = self.np_random.rand(self.obj.anum)
				xs = np.arange(min(self.ng_param[state,:,0])-10.0, max(self.ng_param[state,:,0])+10.0,0.01)
				p_table = []
				for x in xs:
					p_table.append(studentTdist.cdf((x-self.ng_param[state,:,0])*np.sqrt(self.ng_param[state,:,1]*self.ng_param[state,:,2]/self.ng_param[state,:,3]), 
						2*self.ng_param[state,:,2]))
				ids = np.argmin(np.abs(p-np.asarray(p_table)),axis=0)
				action = np.argmax([xs[ids[i]] for i in range(self.obj.anum)])
			else:
				raise ValueError('Not provided action policy.')

			cur_param = self.ng_param[state,action]
			reward, state_n, done = self.obj.observe(state,action,self.np_random)

			std = np.sqrt((cur_param[1]+1)*cur_param[3]/cur_param[1]/(cur_param[2]-1))
			R_samples = np.arange(cur_param[0]-3*std, cur_param[0]+3*std, deltaR, dtype=np.float32)
			R_pdf = norm.pdf(R_samples, cur_param[0],std)
			M1 = reward + self.discount*R_samples
			M2 = M1**2
			mu0_new = (cur_param[1]*cur_param[0] + 1*M1)/(cur_param[1]+1)
			l_new = (cur_param[1]+1)*np.ones(R_samples.shape)
			a_new = (cur_param[2]+0.5)*np.ones(R_samples.shape)
			b_new = cur_param[3]+0.5*(M2-M1**2)+0.5*cur_param[1]*(M1-cur_param[0])**2/l_new
			x_mu = np.arange(-20.0+cur_param[0],20.0+cur_param[0], deltaM, dtype=np.float32)
			x_tau = np.arange(0.01+cur_param[0],0.2+cur_param[0], deltaT, dtype=np.float32)

			e_mt, e_t, e_m2t, e_logt = 0.0, 0.0, 0.0, 0.0
			tot_prob = 0.0
			tmp_mu = []
			for x_m in x_mu:
				tmp_tau = []
				for x_t in x_tau:
					p_mix = deltaR*np.dot(util.normalGamma(x_m, x_t, mu0_new, l_new, a_new, b_new),R_pdf)
					tmp_tau.append(p_mix)							
					tot_prob += p_mix
					e_mt += x_m*x_t*p_mix
					e_t += x_t*p_mix
					e_m2t += x_m*x_m*x_t*p_mix
					e_logt += np.log(x_t)*p_mix
				tmp_mu.append(tmp_tau)
			tot_prob *= deltaT*deltaM
			if abs(tot_prob-1.0) > 0.1:
				pdb.set_trace()
			e_mt *= deltaT*deltaM/tot_prob
			e_t *= deltaT*deltaM/tot_prob
			e_m2t *= deltaT*deltaM/tot_prob
			e_logt *= deltaT*deltaM/tot_prob

			if (np.log(e_t)-e_logt)<0:
				pdb.set_trace()
			# UPDATE
			self.ng_param[state,action,0] = e_mt/e_t
			self.ng_param[state,action,1] = 1.0/(e_m2t - e_t*self.ng_param[state,action,0]**2)
			self.ng_param[state,action,2] = max(1+epsilon, self.finverse(np.log(e_t)-e_logt, digamma))
			self.ng_param[state,action,3] = self.ng_param[state,action,2]/e_t


	def finverse(self, x, fun):
		for y in np.arange(0.01, 10.0, 0.01):
			if abs(x-(np.log(y)-fun(y))) < 0.01:
				return y
		pdb.set_trace()



def isPostiveDefinite(x):
	return np.all(np.linalg.eigvals(x) > 0)


