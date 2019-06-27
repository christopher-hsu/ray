from BRL import brl_util_new as util
import random
import pdb
import numpy as np
import tensorflow as tf
from BRL.DRL.alg.qnetwork import BaseQnetwork
import time

varTH = np.float32(1e-5)

class ADFQ(BaseQnetwork):
	def __init__(
		self,
		env,
		optimizer,
		max_gradient_norm,
		init_bias = (1.0 ,50.0),
		**kwargs
		):
		action_dim = env.action_space.n
		output_bias=[init_bias[0]]*action_dim+[init_bias[1]]*action_dim
		BaseQnetwork.__init__(self, env=env, output_dim=2*action_dim, output_bias=output_bias, **kwargs)

		# if action_policy == 'egreedy':
		# 	self.best_action = tf.argmax(tf.reshape(self.train_network.get_output()[:,:self.action_dim], (self.action_dim,)), output_type = tf.int32)
		# 	#self.curr_action = tf.cond(tf.random_uniform(shape=()) < self.epsilon, 
		# 	#	lambda:tf.random_uniform(shape=(),dtype=tf.int32, maxval=self.action_dim), 
		# 	#	lambda:tf.argmax(tf.reshape(self.train_network.get_output()[:,:self.action_dim], (self.action_dim,)), output_type = tf.int32))
		# elif action_policy == 'semi-Bayes':
		# 	self.curr_action = tf.cond(tf.random_uniform(shape=()) < self.epsilon, 
		# 		lambda:self.bayes_sample(self.train_network.get_output()[:,:self.action_dim], 
		# 									tf.log(1+tf.exp(self.train_network.get_output()[:,self.action_dim:]))), #tf.random_uniform(shape=(),dtype=tf.int32, maxval=self.action_dim), 
		# 		lambda:tf.argmax(tf.reshape(self.train_network.get_output()[:,:self.action_dim], (self.action_dim,)), output_type = tf.int32))
		self.targets = tf.placeholder(tf.float32, shape=[None,2], name = "targets")
		self.Qs = self.train_network.get_output()[:,:self.action_dim]
		action_one_hot = tf.one_hot(self.actions, self.action_dim, 1.0, 0.0, name ="action_one_hot")
		mean_curr = tf.reduce_sum(tf.multiply(self.train_network.get_output()[:, :self.action_dim],action_one_hot), axis = 1)
		std_curr = tf.maximum(np.sqrt(varTH), tf.log(1+tf.exp(tf.reduce_sum(tf.multiply(self.train_network.get_output()[:,self.action_dim:], action_one_hot), axis = 1))))
		self.loss = tf.reduce_mean(tf.contrib.distributions.kl_divergence(tf.distributions.Normal(loc=self.targets[:,0], scale=self.targets[:,1]),
					 tf.distributions.Normal(loc=mean_curr, scale=std_curr)), name = 'loss')

		tf.summary.scalar('loss',self.loss)
		params = tf.trainable_variables()
		param_target = {}
		param_train = {}
		for x in params:
			name = x.name.split("/")
			if "target_network" in x.name:
				param_target["/".join(name[1:])] = x
			elif "train_network" in x.name:
				param_train["/".join(name[1:])] = x
		self.update_target_op = {}
		for (name,variable) in param_train.items():
			self.update_target_op[name] = tf.assign(param_target[name], variable)

		opt = optimizer(self.learning_rate)
		gradients = tf.gradients(self.loss, params)
		clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step = self.global_step)
		self.saver = tf.train.Saver(tf.global_variables())
		self.merged = tf.summary.merge_all()

	def step(self, session, discount, forward = True):
		# Action Selection
		#action  = int(session.run( self.curr_action, 
		#						{self.train_network.get_input_layer(): np.reshape(self.state,(1,)+self.state_dim )}))
		#						#self.train_network.keep_prob : 1.0}))
		action = self.get_egreedy_action(session, self.state, self.epsilon.eval())
		next_obs, reward, done, _ = self.env.step(action)
		next_state = self.get_state(next_obs)

		self.store({'state': self.state,'action':action,'state_n':next_state,'reward':reward,'terminal': int(done)})
		self.state = next_state
		if forward:
			return reward, done, None, None
		else:
			session.run(self.epsilon_decay_op)
			batch = self.get_batch()
			stats = session.run(self.target_network.get_output(),
							{self.target_network.get_input_layer(): batch['state'],})
							#{self.target_network.keep_prob : 1.0 })
			stats_n = session.run(self.target_network.get_output(),
							{self.target_network.get_input_layer(): batch['state_n'],})
							 # self.target_network.keep_prob : 1.0})
			n_means = stats_n[:,:self.action_dim]
			n_vars  = np.maximum(varTH, np.log(1+np.exp(stats_n[:,self.action_dim:]))**2) 
			
			target_mean, target_var, _ = util.posterior_soft_approx(n_means,
										n_vars,
										stats[np.arange(self.batch_size), batch['action']],
										np.maximum(varTH, np.log(1+np.exp(stats[np.arange(self.batch_size), np.array(batch['action'])+self.action_dim]))**2),
										batch['reward'],
										discount, batch['terminal'], batch=True)
			target_mean = np.reshape(target_mean, (self.batch_size,1))
			target_var = np.reshape(target_var, (self.batch_size,1))
			targets = np.concatenate((target_mean, np.sqrt(target_var)), axis=1)
			input_feed = {self.states : batch['state'],
						self.actions : batch['action'],
						self.targets : targets }
						#self.train_network.keep_prob: 0.5 }
			outputs = session.run([self.update, self.loss, self.merged, self.train_network.layers], input_feed)
			if outputs[1] > 5000.0:
				pdb.set_trace()
			return reward, done, outputs[1], outputs[2], [np.mean(x) for x in outputs[-1]], np.mean(target_var)

	def bayes_sample(self, means, stds):
		return tf.argmax(tf.random_normal((self.action_dim,), means, stds), axis=1, output_type=tf.int32)









