import numpy as np
import brl_util as util 
import adfq_fun
import pdb
from scipy.stats import norm
import matplotlib.pyplot as plt

GAMMA = 0.9
ALPHA = 0.2

def rsme(N):
    alpha = 1.0
    gamma = 0.9
    TH = 15000
    init_q = 20.0
    init_var = 100.0
    scene = 'simple'
    #test_names = ['ql-random', 'ql-egreedy 0.1', 'adfq-random', 'adfq-egreedy 0.1', 'adfq-ts', 'adfq-numeric-random']
    test_names = ['Q-learning', 'ADFQ', 'ADFQ-Numeric']
    qerr = [[],[],[]]
    evalr = [[],[],[]]
    for i in range(N):
        np.random.seed(i)
        ql = Qlearning(alpha=alpha,discount=gamma, scene=scene, initQ=init_q, TH=TH)
        ql.learning('random', eval_greedy=True,rate_decay=True)
        qerr[0].append(ql.Q_err)
        evalr[0].append(ql.test_rewards)
        adfq = brl.adfq(scene=scene, discount=gamma, init_mean=init_q,init_var=init_var, TH=TH)
        adfq.learning('random', eval_greedy=True, noise = 0.0, useScale=True, varTH=1e-40)
        qerr[1].append(adfq.Q_err)
        evalr[1].append(adfq.test_rewards)
        adfq3 = brl.adfq(scene=scene, discount=gamma, init_mean=init_q,init_var=init_var, TH=TH)
        adfq3.learning('random', updatePolicy='numeric', eval_greedy=True, noise = 0.0, useScale=True, varTH=1e-40)
        qerr[2].append(adfq3.Q_err)
        evalr[2].append(adfq3.test_rewards)

    markers = ['*-', '+-','d-','o-','x-','s-','2-','3-']
   
    qerr = np.array(qerr)
    util.plot_sd(TH, qerr, labels=test_names, x_label='Learning Steps', y_label='RMSE of Q',
     x_vals=np.arange(0,TH,TH/100))
   
    util.plot_sd(TH, np.array(evalr), labels=test_names, x_label='Learning Steps', y_label='Average Reward per Episode',
     x_vals=np.arange(0,TH,TH/100), legend=(True, 'lower right', (1.0, 0.0)))

    print("OptQ:")
    print(ql.Q_target)
    plt.show()

def td_err_effect():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s2_means = np.array([0.0, 5.0])
    s2_vars = np.array([10.0, 10.0])
    c_var = 1.0
    td_errs = []
    var_hist = []
    comp = []
    for c_mean in np.arange(-10.0, 20.0, 0.5):
        #s2_means[1] = c_mean -10.0
        # if len(comp) > 1: 
        #     if not(comp[-1][0]) and not(comp[-1][1]):
        #         pdb.set_trace()
        outputs = adfq_fun.posterior_adfq(s2_means, s2_vars, c_mean, c_var, 0.0, 0.9, terminal=0, varTH=1e-10)
        td_errs.append(0.0 + 0.9*s2_means - c_mean)
        td_targets =  0.9*s2_means
        comp.append([outputs[2][0][0] < td_targets[1], outputs[2][0][1] < td_targets[0]])
        var_hist.append(outputs[1])

    #td_errs = np.abs(td_errs)
    td_errs = np.array(td_errs)
    pdb.set_trace()
    ax.plot(td_errs[:,0], td_errs[:,1], var_hist, 'bo-')
    ax.set_xlabel('TD error 1')
    ax.set_ylabel('TD error 2')
    ax.set_zlabel('Variance update')
    plt.show()

def weight_example():
    import matplotlib.pyplot as plt
    # s2_means = np.array([-100.0, 10.0])
    # s2_vars = np.array([1.0, 10.0])
    # c_mean = 5.0
    # hist = []
    # var_avg = []
    # mean_disp = []
    # for log_c_var in np.arange(-2.5, 5.0, 0.1):
    #     outputs = adfq_fun.posterior_adfq(s2_means, s2_vars, c_mean, np.exp(log_c_var), 0.0, 0.9, terminal=0, varTH=1e-10)
    #     hist.append(outputs[1])
    #     var_avg.append(np.sum(outputs[2][1]*outputs[2][2]))

    # f, ax = plt.subplots()
    # ax.plot(np.exp(np.arange(-2.5, 5.0, 0.1)), np.exp(np.arange(-2.5, 5.0, 0.1)), 'k--')
    # ax.plot(np.exp(np.arange(-2.5, 5.0, 0.1)), hist)
    # ax.plot(np.exp(np.arange(-2.5, 5.0, 0.1)), var_avg)
    # ax.plot(np.exp(np.arange(-2.5, 5.0, 0.1)), np.array(hist)-np.array(var_avg))
    # ax.legend(['prior','new varaince', 'avg variance', 'mean dispersion'])
    # plt.show()
    # pdb.set_trace()
    hist = []
    
    X = np.arange(-20, 20, 0.1)
    Y = np.arange(-20, 20, 0.1)
    n_vars = np.array([100.0, 100.0])
    for n1 in X:
        for n2 in Y:
            n_means = np.array([n1, n2])
            outputs = adfq_fun.posterior_adfq(n_means, n_vars, 0.0, 1000.0, 0.0, 0.9, terminal=0, varTH=1e-10)
            hist.append(outputs[1])
    hist = np.reshape(hist, (X.shape[0], Y.shape[0]))
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_wireframe(X, Y, hist,rstride=10, cstride=10)
    ax.set_xlabel('s_tp1 mean 1')
    ax.set_ylabel('s_tp1 mean 2')
    ax.set_zlabel('Variance update')
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    pdb.set_trace()

def fun(rewards, test_num):
    alpha = 0.5
    print("Test%d: r=%.2f, r=%.2f"%(test_num, rewards[0], rewards[1]))
    s2_means = np.array([0.0, 0.0])
    s2_vars = np.array([10.0, 10.0])
    c_mean = s2_means[0]
    c_var = s2_vars[0]
    Q = np.zeros((3,2))

    outputs = adfq_fun.posterior_adfq(s2_means, s2_vars, c_mean, c_var, rewards[0], 0.9, terminal=0, varTH=1e-10)
    s2_means[0] = outputs[0]
    s2_vars[0] = outputs[1]
    print("t=1 mean: ", s2_means)
    print("t=1 var: ", s2_vars)
    outputs = adfq_fun.posterior_adfq(s2_means, s2_vars, c_mean, c_var, rewards[1], 0.9, terminal=0, varTH=1e-10)
    s2_means[0] = outputs[0]
    s2_vars[0] = outputs[1]
    print("t=2 mean: ", s2_means)
    print("t=2 var: ", s2_vars)

    print("Before Q: ", Q[1,:])
    for r in rewards:
        Q[1,0] = (1-alpha)*Q[1,0] + alpha*(r + 0.9*max(Q[1,:]))
    print("After Q: ", Q[1,:])

def q_update(sars, q_tab, alpha=ALPHA):
    return (1-alpha)*q_tab[sars[0], sars[1]] + alpha*(sars[2] + 0.9*max(q_tab[sars[-1],:]))

def display_update(prior, targets, stats):
    x = np.arange(0,8,0.05)
    f, ax = plt.subplots()
    #f1, ax1 = plt.subplots()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    ax.plot(x, norm.pdf(x,prior[0], np.sqrt(prior[1])), '-.', color=colors[0], linewidth=2)
    #ax1.plot(x, norm.pdf(x,prior[0], np.sqrt(prior[1])), '-.', color=colors[0])
    legend = ['prior']
    for i in range(len(targets)):
        ax.plot(x, norm.pdf(x, targets[0][i], np.sqrt(targets[1][i])), '--', color=colors[i+1], linewidth=2)
        ax.plot(x, stats[2][2][i]*norm.pdf(x, stats[2][0][i], np.sqrt(stats[2][1][i])), color=colors[i+1], marker= '+', markersize=7)
        legend.append('TD target b=%d'%i)
        #legend.append(r'\mu_{\tau, b=%d}^*'%i)
    ax.plot(x, norm.pdf(x, stats[0], np.sqrt(stats[1])), colors[len(targets)+1], linewidth=2)
    legend.append(['posterior'])
    #ax.legend(legend)
    ax.grid()
    #ax1.grid()
    ax.set_xlim((2,5.5))
    ax.set_ylim((0.0,0.95))
    ax.tick_params(axis='both',which='major',labelsize=15)
    ax.set_xlabel('q',fontsize=18, fontname="Times New Roman")
    ax.set_ylabel('p(q)',fontsize=18, fontname="Times New Roman")
    plt.show()

    pdb.set_trace()

def adfq_update(q_means, q_vars, sars_tuples):
    print("\nInitial ADFQ Mean and Variance")
    print("means:")
    print(q_means[:2,:])
    print("vars:")
    print(q_vars[:2,:])
    visits = np.zeros(q_means.shape)
    for (i,sars) in enumerate(sars_tuples):
        visits[sars[0], sars[1]] += 1.0
        print("t=%d"%i)
        prior = [q_means[sars[0], sars[1]], q_vars[sars[0], sars[1]]]
        targets = [sars[2] + GAMMA*q_means[sars[-1],:], GAMMA*GAMMA*q_vars[sars[-1],:]]
        td_err = sars[2] + GAMMA*q_means[sars[-1],:] -  q_means[sars[0], sars[1]]
        outputs = adfq_fun.posterior_adfq(q_means[sars[-1], :], q_vars[sars[-1],:], q_means[sars[0], sars[1]],
            q_vars[sars[0], sars[1]], sars[2], GAMMA, terminal = 0, varTH=1e-10)

        print("sars:", sars)
        q_means[sars[0], sars[1]] = outputs[0]
        q_vars[sars[0], sars[1]] = outputs[1]
        print("means:")
        print(q_means[:2,:])
        print("vars:")
        print(q_vars[:2,:])
        print("mu_bs:", outputs[2][0].astype(np.float16))
        print("target:", sars[2] + GAMMA*q_means[sars[-1],:] )
        print("var_bs:", outputs[2][1].astype(np.float16))
        print("k_bs:", outputs[2][2].astype(np.float16))
        print("TD error:", td_err)
        if i > 11:
            display_update(prior, targets, outputs)
            q_tab[sars[0], sars[1]] = q_update(sars, q_tab, alpha=1./(visits[sars[0], sars[1]]))
            print("If Q-learning with alpha:%.2f"%(1./(visits[sars[0], sars[1]])))
            print(q_tab)
        else:
            q_tab = np.copy(q_means)
    return q_means, q_vars, q_tab

def qlearning_update(q_tab, sars_tuples):
    print("\nInitial Q table")
    print("Q vals:")
    print(q_tab[:2,:])
    visits = np.zeros(q_tab.shape)
    for (i,sars) in enumerate(sars_tuples):
        visits[sars[0], sars[1]] += 1.0
        print("t=%d"%i)
        q_tab[sars[0], sars[1]] = q_update(sars, q_tab, alpha=1./(visits[sars[0], sars[1]]))
        print("sars:", sars, "alpha:", 1./(visits[sars[0], sars[1]]))
        print("Q vals:")
        print(q_tab[:2,:])

    return q_tab

if __name__ == "__main__":
   
    snum = 4
    anum = 2
    terminal_states = [2,3]
    sars_tuples = [(0, 0, 0.0, 1),
                   (1, 0, 10.0, 2), #v
                   (0, 0, 0.0, 1),
                   (1, 1, 5.0, 3),
                   (0, 0, 0.0, 1),
                   (1, 0, 0.0, 2), #v
                   (0, 0, 0.0, 1),
                   (1, 1, 5.0, 3),
                   (0, 0, 0.0, 1),
                   (1, 0, 0.0, 2), #v
                   (0, 0, 0.0, 1),
                   (1, 1, 5.0, 3),
                   (0, 0, 0.0, 1),
                   # (1, 0, 10.0, 2), #v
                   # (0, 0, 0.0, 1),
                   (1, 1, 5.0, 3),
                   (0, 0, 0.0, 1),
    #                 ]
    # sars_tuples = [
    #                (0, 0, 0.0, 1),
    #                (1, 0, 0.0, 2), #2
    #                (0, 0, 0.0, 1), #3
    #                (1, 1, 5.0, 3), #4
    #                (0, 0, 0.0, 1),
    #                (1, 0, 0.0, 2),
    #                (0, 0, 0.0, 1),
    #                (1, 1, 5.0, 3), #9
    #                (0, 0, 0.0, 1),
    #                (1, 1, -5.0, 2), 
    #                (0, 0, 0.0, 1),
    #                (1, 0, 10.0, 3), 
    #                (0, 0, 0.0, 1),
                   #(1, 0, 0.0, 2),
                   #(0, 0, 0.0, 1),
                   #(1, 1, 5.0, 3), #4
                   #(0, 0, 0.0, 1),
                   ]
    # sars_tuples = [(0, 0, 0.0, 1),
    #                (1, 1, 5.0, 3),
    #                (0, 0, 0.0, 1),
    #                (1, 1, 5.0, 3),
    #                (0, 0, 0.0, 1),
    #                (1, 1, 5.0, 3),
    #                (0, 0, 0.0, 1),
    #                (1, 1, -5.0, 3),
    #                (0, 0, 0.0, 1)]

    #q_means_init = np.array([[4.5,4.5],[2.5, 3.5], [0.0, 0.0],[0.0,0.0]]) #0.0*np.ones((snum,anum))
    q_means_init = 0.0*np.ones((snum,anum))
    q_vars_init = 10.0*np.ones((snum,anum))
    q_tab_init = 0.0*np.ones((snum,anum))
    #q_tab_init = np.array([[4.5,4.5],[2.5, 3.5], [0.0, 0.0],[0.0,0.0]])
    for s in terminal_states:
        q_means_init[s,:] = 0.0
        q_tab_init[s,:] = 0.0

    optQ = np.array([[2.7, 2.7],[2.0,3.0],[0.0, 0.0],[0.0,0.0]])
    print("ADFQ:")
    q_means, q_vars, q_tab_adfq = adfq_update(q_means_init, q_vars_init, sars_tuples)
    print("Q err", np.sqrt(np.mean((optQ-q_means)**2)))
    print("Q err after q learning took over",  np.sqrt(np.mean((optQ-q_tab_adfq)**2)))

    print("\n\nQ-Learnings")
    q_tab = qlearning_update(q_tab_init, sars_tuples)
    print("Q err", np.sqrt(np.mean((optQ-q_tab)**2)))
    pdb.set_trace()




