from BRL import brl_util as util
import random
import pdb
import numpy as np
from collections import deque
import tensorflow as tf
from BRL.DRL.network import MLP, ConvNet

class BaseQnetwork():
	def __init__(
		self,
		env,
		network_params,
		batch_size,
		learning_rate,
		learning_rate_decay_factor,
		learning_rate_range,
		epsilon,
		epsilon_min, 
		epsilon_decay_factor,
		mem_size,
		action_policy,
		output_dim,
		output_bias = None
		):
		self.env = env
		self.state = np.array([])
		self.epsilon = tf.Variable(float(epsilon), trainable=False, dtype=tf.float32)
		self.epsilon_decay_op = self.epsilon.assign(tf.maximum(epsilon_min, self.epsilon*epsilon_decay_factor))
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype = tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(tf.maximum(self.learning_rate*learning_rate_decay_factor,learning_rate_range[0]))
		self.learning_rate_grow_op = self.learning_rate.assign(tf.minimum(self.learning_rate/learning_rate_decay_factor,learning_rate_range[1]))
		self.replayMem = []
		self.mem_size = mem_size
		self.state_dim = env.observation_space.low.shape[0]
		self.action_dim = env.action_space.n
		self.global_step = tf.Variable(0, trainable=False)
		self.action_policy = action_policy
		
		if network_params['name'] == 'mlp':
			self.isCNN = False
			self.state_dim = env.observation_space.low.shape
			self.train_network = MLP("train_network", 
				self.state_dim, 
				(output_dim,), 
				network_params['num_layers'], 
				network_params['hidden_size'], 
				hidden_nonlinearity = network_params['hidden_nonlinearity'],
				output_nonlinearity = network_params['output_nonlinearity'],
				output_bias = output_bias,
				)
			self.target_network = MLP("target_network", 
				self.state_dim, 
				(output_dim,), 
				network_params['num_layers'], 
				network_params['hidden_size'], 
				hidden_nonlinearity = network_params['hidden_nonlinearity'],
				output_nonlinearity = network_params['output_nonlinearity'],
				output_bias = output_bias,
				)
		elif network_params['name'] == 'convnet':
			self.isCNN = True
			self.state_dim = (84,84,4)
			num_conv = len(network_params['conv_filters'])
			self.train_network = ConvNet("train_network", 
					self.state_dim, #env.observation_space.low.shape,
					(output_dim,),
					hidden_sizes = (network_params['hidden_size'],)*network_params['num_layers'],
					conv_filters = network_params['conv_filters'],
					patch_sizes = network_params['patch_sizes'],
					conv_strides = network_params['conv_strides'],
					hidden_nonlinearity = network_params['hidden_nonlinearity'],
					output_nonlinearity = network_params['output_nonlinearity'],
					output_bias = output_bias,
					)
			self.target_network = ConvNet("target_network", 
					self.state_dim, #env.observation_space.low.shape,
					(output_dim,),
					hidden_sizes = (network_params['hidden_size'],)*network_params['num_layers'],
					conv_filters = network_params['conv_filters'],
					patch_sizes = network_params['patch_sizes'],
					conv_strides = network_params['conv_strides'],
					hidden_nonlinearity = network_params['hidden_nonlinearity'],
					output_nonlinearity = network_params['output_nonlinearity'],
					output_bias = output_bias,
					)
		else:
			raise NotImplementedError 
		self.states = self.train_network.get_input_layer()
		self.actions = tf.placeholder(tf.int32, shape=[None],name = "actions")
		

	def reset(self):
		obs = self.env.reset()
		self.state = self.get_state(obs, new = True)
		return obs

	def get_state(self, obs, new =False):
		if self.isCNN:
			phi = util.img_preprocess(obs)
			if new:
				return np.stack((phi, phi, phi, phi), axis=2)
			else:
				return np.append(self.state[:,:,1:], np.reshape(phi, phi.shape+(1,)), axis=2)
		else:
			return obs

	def eval(self, session, test_env, eps, num_iter):
		obs = test_env.reset()
		state = self.get_state(obs, new = True)
		rewards = []
		tot_reward = 0.0
		for _ in range(num_iter):
			action = self.get_egreedy_action(session, state, eps)
			obs, reward, done, _ = test_env.step(action)
			tot_reward += reward
			if done :
				rewards.append(tot_reward)
				tot_reward = 0.0
				obs = test_env.reset()
				state = self.get_state(obs, new = True)
			else:
				state = self.get_state(obs)
		return np.mean(rewards, dtype=np.float32)

	def eval_q(self, session, sample_states):
		return np.mean(np.max(session.run(self.Qs, {self.train_network.get_input_layer(): sample_states } ), axis=1))

	def get_egreedy_action(self, session, state, eps):
		if np.random.rand() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(session.run(self.Qs,
				{self.train_network.get_input_layer(): np.reshape(state,(1,)+self.state_dim)} ))

	def step(self, session, discount, forward=True):
		pass

	def store(self, causality_tup):
		if (len(self.replayMem) == self.mem_size):
			self.replayMem.pop(0)
			self.replayMem.append(causality_tup)
		else:
			self.replayMem.append(causality_tup)		

	def get_batch(self, n = None):
		if n is None:
			n = self.batch_size
		minibatch = {'state':[], 'action':[], 'reward':[], 'state_n':[], 'terminal':[]}
		for _ in range(n):
			d = self.replayMem[random.randint(0,len(self.replayMem)-1)]
			for (k,v) in minibatch.items():
				v.append(d[k])
		return minibatch




		








