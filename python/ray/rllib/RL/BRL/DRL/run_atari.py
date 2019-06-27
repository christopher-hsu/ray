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
tf.app.flags.DEFINE_float("learning_rate", 0.00025, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decay factor")
tf.app.flags.DEFINE_float("learning_rate_min", 0.00001, "Minimum learning rate")
tf.app.flags.DEFINE_float("learning_rate_max", 0.001, "Maximum learning rate")
tf.app.flags.DEFINE_string("action_policy","egreedy", "Action policy")
tf.app.flags.DEFINE_float("epsilon", 1.0, "Epsilon for eps-greedy action policy")
tf.app.flags.DEFINE_float("epsilon_min",0.1, "Minimum epsilon value")
tf.app.flags.DEFINE_float("epsilon_decay_factor", 0.9999976, "epsilon decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0, "Maximum Gradient norm")
tf.app.flags.DEFINE_integer("batch_size", 32 , "Minibatch size")
tf.app.flags.DEFINE_integer("memory_size",10**5, "Experience memory size")
tf.app.flags.DEFINE_integer("warmup_itr", 10000, "The number of iterations for warming up")
tf.app.flags.DEFINE_integer("num_layers", 1, "The number of hidden layers")
tf.app.flags.DEFINE_integer("hidden_size", 512, "The number of units per layer")
tf.app.flags.DEFINE_float("discount", 0.99, "Discount factor in MDP")
tf.app.flags.DEFINE_integer("max_iterations", 10**7, "The maximum number of iterations")
tf.app.flags.DEFINE_boolean("render",False,"openAI gym render.")
tf.app.flags.DEFINE_string("log_dir",".","Logging directory")
tf.app.flags.DEFINE_string("summaries_dir", ".", "Summary Directory")
tf.app.flags.DEFINE_integer("training_interval",4,"training interval.")
tf.app.flags.DEFINE_integer("log_itr", 10000, "Every # iterations, it stores.")
tf.app.flags.DEFINE_integer("target_update_itr", 10000, "Every # iterations, it updates the target network")
tf.app.flags.DEFINE_integer("epoch_itr", 5*(10**4), "One epoch corresponds to #")
tf.app.flags.DEFINE_string("env_name","Breakout-v0","OpenAI gym, environment name")
tf.app.flags.DEFINE_string("optimizer","adam","Optimizer")
tf.app.flags.DEFINE_string("device","/cpu:0", "device name")
 
FLAGS = tf.app.flags.FLAGS
optimizer_dict = {"gradientdescent": tf.train.GradientDescentOptimizer, 
				"adadelta": tf.train.AdadeltaOptimizer,
				"adam": tf.train.AdamOptimizer,
				"rmsprop": tf.train.RMSPropOptimizer}
STEP_PER_EPOCH = FLAGS.epoch_itr
EVAL_EPS = 0.05
EVAL_STEPS = 10000

def run(alg):
	time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
	env = gym.make(FLAGS.env_name)
	network_params = {'name': 'convnet',
						'conv_filters': (32, 64, 64),
						'patch_sizes' : (8, 4, 3),
						'conv_strides': (4, 2, 1),
						'num_layers' : FLAGS.num_layers,
						'hidden_size' : FLAGS.hidden_size,
						'hidden_nonlinearity' : tf.nn.relu,
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
		for i in range(FLAGS.warmup_itr):
			_, done, _, _ = model.step(sess,FLAGS.discount)
			if done:
				state = model.reset()
		q_eval_samples = model.get_batch(1000)
		q_eval_samples = q_eval_samples['state']

		assert(len(model.replayMem)==FLAGS.warmup_itr)
		print("START LEARNING")
		prev_loss = 0.0
		del_loss = [0.0,0.0]
		state = model.reset()
		episode, n_episode, train_itr  = 0, 0, 0
		tot_ep_reward, loss_episode, elapsed_time = 0.0, 0.0, 0.0
		epoch_rewards, epoch_losses, eval_rewards, eval_qs = [], [], [], []
		pickle.dump({"flags": FLAGS, 
					"property": network_params,
					"environment":FLAGS.env_name,
					}, 
					open(os.path.join(FLAGS.log_dir,time_stamp+"_prop.pkl"),"wb"))
		for itr in range(FLAGS.max_iterations):
			if FLAGS.render: model.env.render()
			
			if itr % FLAGS.training_interval == 0:
				start_time = time.time()
				reward, done, loss, summary = model.step(sess, FLAGS.discount, forward = False)
				elapsed_time = (time.time()-start_time)
				loss_episode += loss
				train_itr += 1
			else:
				reward, done, _, _ = model.step(sess, FLAGS.discount, forward = True)

			if itr%FLAGS.target_update_itr == 0:
				sess.run(model.update_target_op)
			
			tot_ep_reward += reward
			if (itr != 0) and (itr%STEP_PER_EPOCH) == 0:
				test_env = gym.make(FLAGS.env_name)
				eval_rewards.append(model.eval(sess, test_env, EVAL_EPS, EVAL_STEPS))
				eval_qs.append(model.eval_q(sess, q_eval_samples))
				print(">>>>> %d Epoch: %d episodes, Episode AVGs: Rewards %.4f, Loss %.4f"%(itr/STEP_PER_EPOCH, episode-n_episode, np.mean(epoch_rewards), np.mean(epoch_losses)))
				print("EVALUATION: %.4f (CURRENT: eps %.4f, learning_rate %.6f, del_t %.4f)"%(eval_rewards[-1], model.epsilon.eval(), model.learning_rate.eval(), elapsed_time))
				
				with  open(os.path.join(FLAGS.log_dir,time_stamp+"_reward.pkl"),"wb") as logFile:
					pickle.dump(eval_rewards, logFile)
				with  open(os.path.join(FLAGS.log_dir,time_stamp+"_qs.pkl"),"wb") as logFile:
					pickle.dump(eval_qs, logFile)
				n_episode = episode

			if done :
				train_writer.add_summary(summary, episode)
				epoch_rewards.append(tot_ep_reward)
				epoch_losses.append(1.0*loss_episode/train_itr)
				if episode > 1:
					del_loss.append(loss_episode - prev_loss) 
				if all([(x>0) for x in del_loss[-3:]]):
					model.learning_rate_decay_op.eval()
				elif np.std(del_loss[-3:]) < 0.1:
					model.learning_rate_grow_op.eval()

				# UPDATE & INITIALIZATION
				prev_loss = loss_episode			
				episode += 1
				train_itr = 0
				state = model.reset()
				tot_ep_reward, loss_episode, elapsed_time = 0.0, 0.0, 0.0

def main(_):
	if FLAGS.algorithm == 'dqn':
		run(DQN)
	elif FLAGS.algorithm == 'adfq':
		run(ADFQ)

if __name__ == "__main__":
	tf.app.run()
