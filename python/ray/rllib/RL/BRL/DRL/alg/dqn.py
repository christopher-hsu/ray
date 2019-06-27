import random
import pdb
import numpy as np
import tensorflow as tf
from BRL.DRL.alg.qnetwork import BaseQnetwork

class DQN(BaseQnetwork):
	def __init__(
		self,
		env,
		optimizer,
		max_gradient_norm,
		**kwargs
		):

		BaseQnetwork.__init__(self,env=env, output_dim=env.action_space.n,**kwargs)
		
		#self.curr_action = tf.cond(tf.random_uniform(shape=()) < self.epsilon, 
		#	lambda:tf.random_uniform(shape=(),dtype=tf.int32, maxval=self.action_dim), 
		#	lambda:tf.argmax(tf.reshape(self.train_network.get_output(), (self.action_dim,)), output_type = tf.int32))
		#self.best_action = tf.argmax(tf.reshape(self.train_network.get_output(), (self.action_dim,)), output_type = tf.int32)
		self.targets = tf.placeholder(tf.float32, shape=[None], name = "targets")
		self.Qs = self.train_network.get_output()
		action_one_hot = tf.one_hot(self.actions, self.action_dim, 1.0, 0.0, name ="action_one_hot")
		q_curr = tf.reduce_sum(tf.multiply(self.train_network.get_output(),action_one_hot), axis = 1)
		self.loss = tf.reduce_mean(tf.square(self.targets - q_curr), name = "loss")
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
		with tf.name_scope("target_network_update"):
			for (name,variable) in param_train.items():
				self.update_target_op[name] = tf.assign(param_target[name], variable)

		opt = optimizer(self.learning_rate)
		gradients = tf.gradients(self.loss, params)
		clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
		self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step = self.global_step)
		self.saver = tf.train.Saver(tf.global_variables())
		self.merged = tf.summary.merge_all()

	def step(self, session, discount, forward=True):
		# Action Selection
		#action  =  int(session.run( self.curr_action, 
		#						{self.train_network.get_input_layer(): np.reshape(self.state,(1,)+self.state_dim)}))
								 #self.train_network.keep_prob : 1.0 })
		action = self.get_egreedy_action(session, self.state, self.epsilon.eval())
		next_obs, reward, done, _ = self.env.step(action)
		next_state = self.get_state(next_obs)

		self.store({'state': self.state,'action':action,'state_n': next_state,'reward':reward,'terminal': int(done)})
		self.state = next_state
		if forward:
			return reward, done, None, None
		else:
			session.run(self.epsilon_decay_op)
			batch = self.get_batch()
			v_n = np.max(session.run(self.target_network.get_output(),
							{self.target_network.get_input_layer(): batch['state_n']}), axis=1)
							 #self.target_network.keep_prob : 1.0})
			targets = np.array(batch['reward']) + discount*(1.-np.array(batch['terminal']))*v_n

			input_feed = {self.states : batch['state'],
							self.actions : batch['action'],
							self.targets : targets,
							 }#self.train_network.keep_prob: 0.5}
			outputs = session.run([self.update, self.loss, self.merged, self.train_network.layers], input_feed)

			return reward, done, outputs[1], outputs[2], [np.mean(x) for x in outputs[-1]]






