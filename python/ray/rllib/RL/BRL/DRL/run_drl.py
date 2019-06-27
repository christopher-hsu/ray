from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from BRL.DRL.alg.dqn import DQN
from BRL.DRL.alg.adfq import ADFQ
import gym
import os, pickle, pdb, time
import numpy as np
import datetime

tf.app.flags.DEFINE_string("algorithm", "dqn", "learning algorithm")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decay factor")
tf.app.flags.DEFINE_float("learning_rate_min", 0.00001, "Minimum learning rate")
tf.app.flags.DEFINE_float("learning_rate_max", 0.5, "Maximum learning rate")
tf.app.flags.DEFINE_string("action_policy","egreedy", "Action policy")
tf.app.flags.DEFINE_float("epsilon", 0.5, "Epsilon for eps-greedy action policy")
tf.app.flags.DEFINE_float("epsilon_min",0.01, "Minimum epsilon value")
tf.app.flags.DEFINE_float("epsilon_decay_factor", 0.9997, "epsilon decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum Gradient norm")
tf.app.flags.DEFINE_integer("batch_size", 100 , "Minibatch size")
tf.app.flags.DEFINE_integer("memory_size",10000, "Experience memory size")
tf.app.flags.DEFINE_integer("num_layers", 2, "The number of hidden layers")
tf.app.flags.DEFINE_integer("hidden_size", 128, "The number of units per layer")
tf.app.flags.DEFINE_float("discount", 0.9, "Discount factor in MDP")
tf.app.flags.DEFINE_integer("num_episodes", 300, "The number of episodes to run")
tf.app.flags.DEFINE_integer("max_iterations", 250, "The maximum number of iteration per episode")
tf.app.flags.DEFINE_boolean("render",False,"openAI gym render.")
tf.app.flags.DEFINE_string("log_dir",".","Logging directory")
tf.app.flags.DEFINE_string("summaries_dir", ".", "Summary Directory")
tf.app.flags.DEFINE_integer("log_itr", 10, "Every # iterations, it updates.")
tf.app.flags.DEFINE_integer("target_update_itr", 10, "Every # iterations, it updates the target network")
tf.app.flags.DEFINE_string("env_name","CartPole-v0","OpenAI gym, environment name")
tf.app.flags.DEFINE_string("optimizer","adam","Optimizer")
tf.app.flags.DEFINE_string("device","/cpu:0", "device name")
 
FLAGS = tf.app.flags.FLAGS
optimizer_dict = {"gradientdescent": tf.train.GradientDescentOptimizer, 
				"adadelta": tf.train.AdadeltaOptimizer,
				"adam": tf.train.AdamOptimizer,
				"rmsprop": tf.train.RMSPropOptimizer}
env_solved = {'CartPole-v0': {'bound':195.0, 'network':'mlp'}, 
			'MountainCar-v0': {'bound':-110.0, 'network':'mlp'},
			'Breakout-v0':{'bound':200.0, 'network':'convnet'},
			}

def run(alg):
	time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
	IS_CNN = False
	env = gym.make(FLAGS.env_name)
	if env_solved[FLAGS.env_name]['network'] == 'mlp':
		network_params = {'name': 'mlp',
						'num_layers' : FLAGS.num_layers,
						'hidden_size' : FLAGS.hidden_size,
						'hidden_nonlinearity' : tf.nn.tanh,
						'output_nonlinearity' : tf.identity,
						}
	elif env_solved[FLAGS.env_name]['network'] == 'convnet':
		network_params = {'name': 'convnet',
						'conv_filters': (16,32),
						'patch_sizes' : (8,4),
						'conv_strides': (4,2),
						'num_layers' : FLAGS.num_layers,
						'hidden_size' : FLAGS.hidden_size,
						'hidden_nonlinearity' : tf.nn.tanh,
						'output_nonlinearity' : tf.identity,
						}
	
	with tf.Session() as sess:
		model = alg(env=env, 
					network_params=network_params,
					optimizer=optimizer_dict[FLAGS.optimizer],
					batch_size=FLAGS.batch_size,
					learning_rate=FLAGS.learning_rate,
					learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
					learning_rate_range=(FLAGS.learning_rate_min,FLAGS.learning_rate_max),
					max_gradient_norm=FLAGS.max_gradient_norm,
					epsilon=FLAGS.epsilon,
					epsilon_min=FLAGS.epsilon_min,
					epsilon_decay_factor=FLAGS.epsilon_decay_factor,
					mem_size=FLAGS.memory_size,
					action_policy=FLAGS.action_policy,
					)
		train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, time_stamp),  sess.graph)
		sess.run(tf.global_variables_initializer())
		# Fill the experience replay memory
		state = model.reset()
		for i in range(FLAGS.memory_size):
			_, done, _, _ = model.step(sess,FLAGS.discount)
			if done:
				state = model.reset()
		assert(len(model.replayMem)==FLAGS.memory_size)
		print("START LEARNING")
		prev_loss = 0.0
		del_loss = [0.0,0.0]
		ep_reward = []
		avg_rewards = []
		mean_weights = []
		for m in range(FLAGS.num_episodes):
			# Initialize
			itr = 0
			done = False
			state = model.reset()
			loss_episode, total_reward = 0.0, 0.0
			elapsed_time_itr = 0.0
			mean_var_tot = 0.0
			while( (itr < FLAGS.max_iterations) and not(done)):
				if FLAGS.render: model.env.render()
				start_time = time.time()
				reward, done, loss, summary, mean_w, mean_var = model.step(sess, FLAGS.discount, forward = False)
				mean_weights.append(mean_w)
				if model.global_step.eval()%FLAGS.target_update_itr == 0:
					sess.run(model.update_target_op)
				elapsed_time_itr += (time.time()-start_time)
				total_reward += reward
				loss_episode += loss
				mean_var_tot += mean_var
				itr += 1
			elapsed_time_itr /= itr
			# Loss Evaluation
			train_writer.add_summary(summary, m)
			del_loss.append(loss_episode - prev_loss) if m > 1 else del_loss.append(0.0)
			prev_loss = loss_episode
			if all([(x>0) for x in del_loss[-3:]]):
				model.learning_rate_decay_op.eval()
			elif np.std(del_loss[-3:]) < 0.1:
				model.learning_rate_grow_op.eval()
			# Total Reward Evaluation
			ep_reward.append(total_reward)
			if len(ep_reward) < 100:
				avg_rew = np.mean(ep_reward)
			else:
				avg_rew = np.mean(ep_reward[-100:])
				if avg_rew > env_solved[FLAGS.env_name]['bound']:
					print("SOLVED!!!")
					break
			print("Ep: %d, t: %d, tot_Reward: %.2f, AvgReward_ep: %.2f, AvgLoss_ep: %.2f,  eps: %.4f, LR: %.4f, del_t: %.4f, mean_var: %.4f"%
				(m, itr, total_reward,  avg_rew, loss_episode/itr, model.epsilon.eval(), model.learning_rate.eval(), elapsed_time_itr, mean_var_tot/itr))
			# SAVE
			avg_rewards.append(avg_rew)
			if m%FLAGS.log_itr == 0:
				pickle.dump({"flags": FLAGS, 
							"reward":ep_reward, 
							"property": network_params,
							"environment":FLAGS.env_name,
							"ElapsedTime": elapsed_time_itr,
							"avg_rewards" : avg_rewards}, 
					open(os.path.join(os.path.join(FLAGS.log_dir,time_stamp),"result.pkl"),"wb"))
		import matplotlib.pyplot as plt
		mw = np.asarray(mean_weights)
		[plt.plot(mw[:,i]) for i in range(mw.shape[-1])]
		plt.show()
		

def main(_):
	if FLAGS.algorithm == 'dqn':
		run(DQN)
	elif FLAGS.algorithm == 'adfq':
		run(ADFQ)

if __name__ == "__main__":
	tf.app.run()
