import matplotlib.pyplot as plt
import numpy as np
import brl as brl
import tabularRL as trl
import pickle
import pdb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--discount', type=float, default=0.95, help= 'discount factor')
parser.add_argument('--slip_p', type=float, default=0.1, help='stohcasticity')
parser.add_argument('--result_dir', type=str, default='./')
parser.add_argument('--noise', type=float, default=1e-25)
parser.add_argument('--a_trigger', type=float, default=1e-20)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=30)
args = parser.parse_args()

batch_size = args.batch_size
useScale = bool(args.scale)
discount = args.discount 
slip_p = args.slip_p

scene_set= {'minimaze':(4,20000, 0.001, False)}#'chain':(2, 5000, 160.0, False),'grid5':(4,15000, 0.001, False),'loop':(2,5000, 10.0, False)}
if slip_p > 0.0:
	scene_set.pop('loop',None)
if batch_size > 0:
	model_name =  ['SoftApprox', 'Approx' ,'Q-learning']
else:
	model_name = ['Numeric', 'SoftApprox', 'SoftApprox-Asymptotic', 'Approx', 'Approx-Asymptotic','Q-learning']
 
markers = ['^', '*', 'x','+','|','1']
colors = ['y','b','g','r','c','m']

results_rew ={k:[] for k in scene_set.keys()}
results_qerr = {k:[] for k in scene_set.keys()}
results_count = {k:[] for k in scene_set.keys()}
models = {}
for scene, (anum, T, init_mean, init_policy) in scene_set.items():
	print("Learning in %s domain ..."%scene)
	print("Stochasticity: %.2f"%slip_p)
	actions = np.random.choice(anum, T)
	models[scene] = []
	f1, ax1 = plt.subplots()
	f2, ax2 = plt.subplots()
	f3, ax3 = plt.subplots()
	f4, ax4 = plt.subplots()
	"""
	# Numeric
	x_num = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
	x_num.obj.set_time(T)
	x_num.obj.set_slip(slip_p)
	x_num.learning('Numeric', 'offline', actions, eval_greedy = True, useScale=useScale, batch_size=batch_size, noise = args.noise)
	models[scene].append(x_num)
	"""
	# SoftApprox
	x_softapp = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
	x_softapp.obj.set_time(T)
	x_softapp.obj.set_slip(slip_p)
	x_softapp.learning('SoftApprox', 'offline', actions, eval_greedy=True, useScale=useScale, batch_size=batch_size, noise = args.noise)
	models[scene].append(x_softapp)
	if batch_size == 0:
		# SoftApprox with Asymptotic
		x_softapp_asym = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
		x_softapp_asym.obj.set_time(T)
		x_softapp_asym.obj.set_slip(slip_p)
		x_softapp_asym.learning('SoftApprox', 'offline', actions, eval_greedy=True, useScale=useScale, 
			asymptotic=True, asymptotic_trigger=args.a_trigger, batch_size=batch_size, noise = args.noise)
		models[scene].append(x_softapp_asym)
	# HardApprox
	x_app = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
	x_app.obj.set_time(T)
	x_app.obj.set_slip(slip_p)
	x_app.learning('Approx', 'offline', actions, eval_greedy=True, useScale=useScale, batch_size=batch_size, noise = args.noise)
	models[scene].append(x_app)
	if batch_size == 0:
		# HardApprox with Asymptotic
		x_app_asym = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
		x_app_asym.obj.set_time(T)
		x_app_asym.obj.set_slip(slip_p)
		x_app_asym.learning('Approx', 'offline', actions, eval_greedy=True, useScale=useScale, asymptotic=True, 
			asymptotic_trigger=args.a_trigger, batch_size=batch_size, noise = args.noise)
		models[scene].append(x_app_asym)
	# Qlearning
	x_q = trl.Qlearning(scene, 0.5, discount, init_mean )
	x_q.obj.set_time(T)
	x_q.obj.set_slip(slip_p)
	x_q.learning('offline', actions, eval_greedy=True, rate_decay=False)
	models[scene].append(x_q)

	ts1 = np.arange(0,T, T/100)
	ts2 = np.arange(0, T, T/200)
	max_rewards = 0.0
	for (i,v) in enumerate(models[scene][:len(model_name)]):
		ax1.plot(ts2, v.Q_err, marker=markers[i], color=colors[i], markeredgecolor=colors[i])
		ax2.plot(ts1, v.test_rewards, marker=markers[i], color=colors[i], markeredgecolor=colors[i])
		max_rewards = max(max_rewards, max(v.test_rewards))
		if model_name[i] != 'Q-learning':
			ax3.plot(ts2, np.mean(np.mean(v.mean_history[:,v.obj.eff_states,:], axis=-1), axis=-1), marker=markers[i], color=colors[i], markeredgecolor=colors[i])
			ax4.plot(ts2, np.log(np.mean(np.mean(v.var_history[:,v.obj.eff_states,:], axis=-1),axis=-1)),marker=markers[i], color=colors[i], markeredgecolor=colors[i])
		else:
			ax3.plot(ts2, np.mean(np.mean(v.Q_history[:,v.obj.eff_states,:], axis=-1), axis=-1), marker=markers[i], color=colors[i], markeredgecolor=colors[i])

	ax1.set_xlabel('Learning Steps')
	ax1.set_ylabel('RMSE')
	ax1.legend(model_name)
	ax1.set_title(' '.join([scene, str(slip_p), str(discount)]))
	f1.savefig(scene+'_'+str(int(slip_p*10))+'_qerr_cnoise001.png')

	ax2.set_xlabel('Learning Steps')
	ax2.set_ylabel('Cumulative Rewards')
	ax2.set_ylim(0.0, max_rewards*1.1)
	ax2.legend(model_name,loc='lower right')
	ax2.set_title(' '.join([scene, str(slip_p), str(discount)]))
	f2.savefig(scene+'_'+str(int(slip_p*10))+'_eval_cnoise001.png')

	ax3.set_ylabel('Average Mean')
	ax3.legend(model_name)
	ax3.set_title(' '.join([scene, str(slip_p), str(discount)]))
	f3.savefig(scene+'_'+str(int(slip_p*10))+'_mean_cnoise001.png')

	ax4.set_ylabel('Average Variance (Log)')
	ax4.legend(model_name[:-1])
	ax4.set_title(' '.join([scene, str(slip_p), str(discount)]))
	f4.savefig(scene+'_'+str(int(slip_p*10))+'_variance_cnoise001.png')

	if False:#args.slip_p > 0.0:
		x_num = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
		x_num.obj.set_time(T)
		x_num.obj.set_slip(slip_p)
		x_num.learning('Numeric', 'offline', actions, eval_greedy = True, batch_size = batch_size, noise = args.noise)
		models[scene].append(x_num)
		# SoftApprox
		x_softapp = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
		x_softapp.obj.set_time(T)
		x_softapp.obj.set_slip(slip_p)
		x_softapp.learning('SoftApprox', 'offline', actions, eval_greedy=True, useScale=useScale, batch_size = batch_size, noise = args.noise)
		models[scene].append(x_softapp)
		if batch_size == 0:
			# SoftApprox with Asymptotic
			x_softapp_asym = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
			x_softapp_asym.obj.set_time(T)
			x_softapp_asym.obj.set_slip(slip_p)
			x_softapp_asym.learning('SoftApprox', 'offline', actions, eval_greedy=True, useScale=useScale, asymptotic=True, 
				asymptotic_trigger=args.a_trigger, batch_size = batch_size, noise = args.noise)
			models[scene].append(x_softapp_asym)
		# HardApprox
		x_app = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
		x_app.obj.set_time(T)
		x_app.obj.set_slip(slip_p)
		x_app.learning('Approx', 'offline', actions, eval_greedy=True, useScale=useScale, batch_size = batch_size, noise = args.noise)
		models[scene].append(x_app)
		if batch_size == 0:
			# HardApprox with Asymptotic
			x_app_asym = brl.adfq(scene, discount, init_mean, 100.0, init_policy=init_policy)
			x_app_asym.obj.set_time(T)
			x_app_asym.obj.set_slip(slip_p)
			x_app_asym.learning('Approx', 'offline', actions, eval_greedy=True, useScale=useScale, asymptotic=True, asymptotic_trigger=args.a_trigger, 
				batch_size = batch_size, noise = args.noise)
			models[scene].append(x_app_asym)

		# Qlearning
		x_q = trl.Qlearning(scene, 0.5, discount, init_mean )
		x_q.obj.set_time(T)
		x_q.obj.set_slip(slip_p)
		x_q.learning('offline', actions, eval_greedy=True, rate_decay=False, batch_size = batch_size)
		models[scene].append(x_q)

		f5, ax5 = plt.subplots()
		f6, ax6 = plt.subplots()
		f7, ax7 = plt.subplots()
		f8, ax8 = plt.subplots()

		for (k,v) in enumerate(models[scene][len(model_name):]):
			ax5.plot(ts2, v.Q_err, marker=markers[k], color=colors[k], markeredgecolor=colors[k])
			ax6.plot(ts1, v.test_rewards, marker=markers[k], color=colors[k], markeredgecolor=colors[k])
			if model_name[k] != 'Q-learning':
				ax7.plot(ts2, np.mean(np.mean(v.mean_history[:,v.obj.eff_states,:], axis=-1), axis=-1), marker=markers[k], color=colors[k], markeredgecolor=colors[k])
				ax8.plot(ts2, np.log(np.mean(np.mean(v.var_history[:,v.obj.eff_states,:], axis=-1),axis=-1)),marker=markers[k], color=colors[k], markeredgecolor=colors[k])
			else:
				ax7.plot(ts2, np.mean(np.mean(v.Q_history[:,v.obj.eff_states,:], axis=-1), axis=-1), marker=markers[k], color=colors[k], markeredgecolor=colors[k])

		ax5.set_xlabel('Learning Steps')
		ax5.set_ylabel('RMSE')
		ax5.legend(model_name)
		ax5.set_title(' '.join([scene, str(slip_p), str(discount), 'With noise']))
		f5.savefig(scene+'_'+str(int(slip_p*10))+'_noise_qerr.png')

		ax6.set_xlabel('Learning Steps')
		ax6.set_ylabel('Cumulative Rewards')
		ax6.legend(model_name, loc='lower right')
		ax6.set_title(' '.join([scene, str(slip_p), str(discount),  'With noise']))
		if scene == 'grid5':
			ax6.set_ylim(0.0,1.1)
		if scene == 'minimaze':
			ax6.set_ylim(0.0,3.1)
		f6.savefig(scene+'_'+str(int(slip_p*10))+'_noise_eval.png')

		ax7.set_ylabel('Average Mean')
		ax7.legend(model_name)
		ax7.set_title(' '.join([scene, str(slip_p), str(discount), 'With noise ']))
		f7.savefig(scene+'_'+str(int(slip_p*10))+'_noise_mean.png')

		ax8.set_ylabel('Average Variance (Log)')
		ax8.legend(model_name[:-1])
		ax8.set_title(' '.join([scene, str(slip_p), str(discount),  'With noise ']))
		f8.savefig(scene+'_'+str(int(slip_p*10))+'_noise_variance.png')

pdb.set_trace()
"""""

	for asym in [False, True]:
		for i in range(Nrun):
			mean_rewards = np.zeros((100,))
			mean_counts = np.zeros((100,)) 
			mean_qerr = np.zeros((sc_v[1],))

			adfq = brl.adfq(sc, discount, init_mean, 100.0, init_policy=init_policy) 
			adfq.learning(updatePolicy='Approx', actionPolicy='offline', actionParam=actions[i], asymptotic=asym, eval_greedy=True)
			mean_rewards += np.array(adfq.test_rewards)
			mean_counts += np.array(adfq.test_counts)
			mean_qerr += np.array(adfq.Q_err)

		mean_rewards = mean_rewards/float(Nrun)
		mean_counts = mean_counts/float(Nrun)
		mean_qerr = mean_qerr/float(Nrun)

		results_rew[sc].append(mean_rewards)
		results_count[sc].append(mean_counts)
		results_qerr[sc].append(mean_qerr)

		ax1.plot(mean_rewards, marker=markers[i])
		ax2.plot(mean_counts, marker=markers[i])
		ax3.plot(mean_qerr)

	for  v in [-20.0, -60.0]:
		for i in range(Nrun):
			mean_rewards = np.zeros((100,)) 
			mean_counts = np.zeros((100,)) 
			mean_qerr = np.zeros((sc_v[1],))

			adfq = brl.adfq_log(sc, discount, init_mean, 100.0, init_policy=init_policy) 
			adfq.learning(updatePolicy='Approx', actionPolicy='offline', actionParam=actions[i], updateParam2=v, eval_greedy=True)
			mean_rewards += np.array(adfq.test_rewards)
			mean_counts += np.array(adfq.test_counts)
			mean_qerr += np.array(adfq.Q_err)

		mean_rewards = mean_rewards/float(Nrun)
		mean_counts = mean_counts/float(Nrun)
		mean_qerr = mean_qerr/float(Nrun)

		results_rew[sc].append(mean_rewards)
		results_count[sc].append(mean_counts)
		results_qerr[sc].append(mean_qerr)

		ax1.plot(mean_rewards, marker=markers[i])
		ax2.plot(mean_counts, marker=markers[i])
		ax3.plot(mean_qerr)

	for i in range(Nrun):
		mean_rewards = np.zeros((100,)) 
		mean_counts = np.zeros((100,)) 
		mean_qerr = np.zeros((sc_v[1],))

		adfq = brl.adfq(sc, discount, init_mean, 100.0, init_policy=init_policy) 
		adfq.learning(updatePolicy='SoftApprox', actionPolicy='offline', actionParam=actions[i], eval_greedy=True)
		mean_rewards += np.array(adfq.test_rewards)
		mean_counts += np.array(adfq.test_counts)
		mean_qerr += np.array(adfq.Q_err)

	mean_rewards = mean_rewards/float(Nrun)
	mean_counts = mean_counts/float(Nrun)
	mean_qerr = mean_qerr/float(Nrun)

	results_rew[sc].append(mean_rewards)
	results_count[sc].append(mean_counts)
	results_qerr[sc].append(mean_qerr)

	ax1.plot(mean_rewards, marker=markers[i])
	ax2.plot(mean_counts, marker=markers[i])
	ax3.plot(mean_qerr)

	ax1.legend(model_name)
	ax1.set_title(' '.join([sc, 'performance']))
	ax2.legend(model_name)
	ax2.set_title(' '.join([sc, 'counts']))
	ax3.legend(model_name)
	ax3.set_title(' '.join([sc, 'Q error']))

pickle.dump(results_qerr, open('results_qrr_maze.pkl','wb'))
pickle.dump(results_rew, open('results_rew_maze.pkl','wb'))

pdb.set_trace()

"""""


