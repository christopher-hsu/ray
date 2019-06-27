import brl
import numpy as np
import matplotlib.pyplot as plt
import tabularRL as trl
import brl_util as util

discount = 0.95
Nrun = 10
timeH = 500

adfq_eg = [brl.adfq_dynamic('movingmaze', discount, TH=timeH) for _ in range(Nrun)]
rec_eg = [x.learning('egreedy',0.1,eval_greedy=True) for x in adfq_eg]

adfq_eg_pen = [brl.adfq_dynamic('movingmaze', discount, TH=timeH) for _ in range(Nrun)]
rec_eg_pen = [x.learning('egreedy',0.1,eval_greedy=True, beta=1.0) for x in adfq_eg_pen]

adfq_bayes = [brl.adfq_dynamic('movingmaze', discount, TH=timeH) for _ in range(Nrun)]
rec_bayes = [x.learning('Bayes',None, eval_greedy=True) for x in adfq_bayes]

adfq_bayes_pen = [brl.adfq_dynamic('movingmaze', discount, TH=timeH) for _ in range(Nrun)]
rec_bayes_pen = [x.learning('Bayes', None, eval_greedy=True, beta=1.0) for x in adfq_bayes_pen]

#ql02 = [trl.Qlearning_dynamic('movingmaze', 0.2, discount, TH=timeH) for _ in range(Nrun)]
#rec_ql02 = [x.learning('egreedy', 0.1, eval_greedy=True)for x in ql02]
ql05 = [trl.Qlearning_dynamic('movingmaze',0.5, discount, TH=timeH) for _ in range(Nrun)]
rec_ql05 = [x.learning('egreedy',0.1, eval_greedy=True, rate_decay=True) for x in ql05]
t = range(0, timeH, int(timeH/util.EVAL_NUM))

q_errs = np.asarray([[x.Q_err for x in adfq_eg],[x.Q_err for x in adfq_eg_pen],[x.Q_err for x in adfq_bayes], [x.Q_err for x in adfq_bayes_pen],
	[x.Q_err for x in ql05]])
test_rewards = np.asarray([[x.test_rewards for x in adfq_eg],[x.test_rewards for x in adfq_eg_pen],[x.test_rewards for x in adfq_bayes], [x.test_rewards for x in adfq_bayes_pen],
	[x.test_rewards for x in ql05]])
test_counts = np.asarray([[x.test_counts for x in adfq_eg],[x.test_counts for x in adfq_eg_pen],[x.test_counts for x in adfq_bayes], [x.test_counts for x in adfq_bayes_pen],
	[x.test_counts for x in ql05]])

changePt = adfq_eg[0].env.changePt
labels =[r'ADFQ, $\epsilon$-greedy, 0.1',r'ADFQ, $\epsilon$-greedy, 0.1 w/ penalty', 'ADFQ, Bayes', 'ADFQ, Bayes w/ penalty', r'Q-learning, $\epsilon$-greedy, 0.1']
f1, ax1 = util.plot_IQR(timeH, q_errs, labels=labels, x_vals=t, x_label='Learning Step', y_label='RMSE of Q', 
	title='Moving Maze w/ Change point %d'%changePt, legend=(True,'upper left'))
ax1.axvline(x=adfq_eg[0].env.changePt, color='k')
f2, ax2 = util.plot_IQR(timeH, test_rewards, labels=labels, x_vals=t, x_label='Learning Step', y_label='Average Episode Reward', 
	title='Moving Maze w/ Change point %d'%changePt, legend=(True,'lower left'))
ax2.axvline(x=adfq_eg[0].env.changePt, color='k')
f3, ax3 = util.plot_IQR(timeH, test_counts, labels=labels, x_vals=t, x_label='Learning Step', y_label='Average Episode Counts', 
	title='Moving Maze w/ Change point %d'%changePt, legend=(True,'upper right'))
ax3.axvline(x=changePt, color='k')
plt.show()

"""
plt.figure(0)
plt.title('Moving Maze w/ Change point 50')
plt.plot(t, np.mean([x.Q_err for x in adfq_eg], axis=0))
plt.plot(t, np.mean([x.Q_err for x in adfq_bayes], axis=0))
plt.plot(t, np.mean([x.Q_err for x in adfq_bayes_pen], axis=0))
plt.plot(t, np.mean([x.Q_err for x in ql02], axis=0))
plt.plot(t, np.mean([x.Q_err for x in ql05], axis=0))
plt.axvline(x=adfq_eg[0].env.changePt, color='k')
plt.ylabel('RMSE of Q')
plt.xlabel('Learning Step')
plt.legend([r'ADFQ, $\epsilon$-greedy, 0.1', 'ADFQ, Bayes', 'ADFQ, Bayes w/ penalty', r'Q-learning, $\alpha$=0.2, $\epsilon$-greedy, 0.1', r'Q-learning, $\alpha$=0.5, $\epsilon$-greedy, 0.1'])

plt.figure(1)
plt.title('Moving Maze w/ Change point 50')
plt.plot(t, np.mean([x.test_rewards for x in adfq_eg], axis=0))
plt.plot(t, np.mean([x.test_rewards for x in adfq_bayes], axis=0))
plt.plot(t, np.mean([x.test_rewards for x in adfq_bayes_pen], axis=0))
plt.plot(t, np.mean([x.test_rewards for x in ql02], axis=0))
plt.plot(t, np.mean([x.test_rewards for x in ql05], axis=0))
plt.axvline(x=adfq_eg[0].env.changePt, color='k')
plt.ylabel('Average Episode Reward')
plt.xlabel('Learning Step')
plt.legend([r'ADFQ, $\epsilon$-greedy, 0.1', 'ADFQ, Bayes','ADFQ, Bayes w/ penalty', r'Q-learning, $\alpha$=0.2, $\epsilon$-greedy, 0.1', r'Q-learning, $\alpha$=0.5, $\epsilon$-greedy, 0.1'])

plt.show()
"""