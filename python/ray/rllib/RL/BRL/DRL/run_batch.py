import os, pdb, pickle
import tensorflow as tf
import numpy as np
import brl_util as util
learning_rate = [0.001, 0.005, 0.0001]
target_update = [1, 10, 30]

Nrun = 5

tf.app.flags.DEFINE_string("data_dir", ".", "data directory")
tf.app.flags.DEFINE_string("function", "experiment", "experiment or analysis")

FLAGS = tf.app.flags.FLAGS

def experiment():

	for ap in ['egreedy','semi-Bayes']:	
		for lr in learning_rate:
			for tu in target_update:
				for i in range(Nrun):
					print("ADFQ_GEN with LR: %.4f, Updating target %d, N: %d"%(lr, tu, i))
					command = ["python run_drl.py",
								"--algorithm", "adfq",
								"--log_dir", "result/ADFQ_GEN/",
								"--summaries_dir", "result/ADFQ_GEN/",
								"--learning_rate", str(lr),
								"--target_update_itr", str(tu),
								"--action_policy", ap]
					os.system(" ".join(command))

	for lr in learning_rate:
		for tu in target_update:
			for i in range(Nrun):
				print("DQN with LR: %.4f, Updating target %d, N: %d"%(lr, tu, i))
				command = ["python run_drl.py",
							"--algorithm", "dqn",
							"--log_dir", "result/DQN/",
							"--summaries_dir", "result/DQN/",
							"--learning_rate", str(lr),
							"--target_update_itr", str(tu)]
				os.system(" ".join(command))

def analysis(data_dir):
	import matplotlib.pyplot as plt
	
	flist = os.listdir(data_dir)
	sets= {'semi-Bayes':{}, 'egreedy':{}}
	for fname in flist:
		X = pickle.load(open(os.path.join(data_dir,fname), "rb"))
		fX = X['flags'].__flags
		if fX['learning_rate'] > 0.0001:
			key = ' '.join([str(fX['learning_rate']), str(fX['target_update_itr'])])
			if key in sets[fX['action_policy']].keys():
				sets[fX['action_policy']][key].append(X['reward'])
			else:
				sets[fX['action_policy']][key] = [X['reward']]
	f1, ax1 = plt.subplots()
	f2, ax2 = plt.subplots()
	
	for (fig, policy) in [(ax1, 'egreedy'), (ax2, 'semi-Bayes')]: 
		l_legend = []
		idx = 0
		for (k,v) in sets[policy].items():
			l_legend.append(k)
			max_len = max([len(each_v) for each_v in v])
			ave_rewards = []
			for each_v in v:
				ave_rew = [np.mean(each_v[:i+1]) for i in range(99)] \
							+ [np.mean(each_v[i:i+100]) for i in range(len(each_v)-99)] \
							+ [195.0]*(max_len-len(each_v))
				ave_rewards.append(ave_rew)
			m, ids25, ids75 = util.iqr(np.asarray(ave_rewards))
			ts = range(0,len(m))
			fig.plot(ts, m, color=util.COLORS[idx], linewidth=1.5)
			fig.fill_between(ts, list(ids25), list(ids75), facecolor=util.COLORS[idx], alpha = 0.2)
			idx+=1
			#fig.plot(np.mean(ave_rewards,axis=0))
		fig.legend(l_legend,loc='upper left')
		fig.set_title(policy)
	plt.show()




def main(_):
	if FLAGS.function == "experiment":
		experiment()
	elif FLAGS.function == "analysis":
		analysis(FLAGS.data_dir)

if __name__ == "__main__":
	tf.app.run()