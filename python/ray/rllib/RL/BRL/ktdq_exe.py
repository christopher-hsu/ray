
from ktd_q import *
import brl 
import models
import matplotlib.pyplot as plt
import seeding
import numpy as np
import gym
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--kappa', type=int, default=10, help='kappa')
parser.add_argument('--epsilon', type=float, default=0.0, help= 'epsilon for covariance')
parser.add_argument('--gym', type=bool, default=False)
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--iter', type=int, default=100, help='number of trials')

args = parser.parse_args()

def main():
	np_random,_  = seeding.np_random(None)
	np.random.seed()
	eta = 0.0
	reward_noise = 1.0
	P_init = 10.
	theta_noise = None
	env = models.Inv_Pendulum()
	test_env = models.Inv_Pendulum()
	x = KTD_Q(phi=env.phi[0], gamma=0.95, P_init=P_init, theta0 = np.zeros(30,),theta_noise=theta_noise, eta=eta,
	             reward_noise=reward_noise, anum = 3, kappa = 10.0)
	performance = []
	episode = 0
	step = 0
	state = env.reset(np_random)
	while(episode < 1000):
		action = np.random.choice(3,)
		reward, n_state, done = env.observe(state, action)
		x.update_V(state, action, n_state, reward)
		state = n_state
		step +=1
		if done or (step > 3000):
			if episode%50 == 0:
				performance.append(test(x, test_env))
				print("After %d steps, Episode %d: %d"%(step, episode, performance[-1]))
			episode +=1
			step = 0
			state = env.reset(np_random)

	plt.plot(performance)
	plt.show()

def main_brl():
	perf = []
	np.random.seed()
	for i in range(args.iter):
		print("Iteration %d"%i)
		x = brl.ktd_Q('inv_pendulum', 0.95)
		if args.gym:
			x.learning_cartpole_gym(args.kappa, epsilon = args.epsilon)
		else:
			x.learning_cartpole(args.kappa, epsilon = args.epsilon, obs_noise=1.0)
		if np.mean(x.test_counts) > 2000.0:
			pdb.set_trace()
		perf.append(x.test_counts)
	plt.plot(np.mean(perf, axis=0))
	plt.show()
	means = np.array(x.avgmeans)
	[plt.plot(means[:,i]) for i in range(30)]; plt.show()
	pdb.set_trace()

def main_gym():
	eta = 0.0
	reward_noise = 1.0
	P_init = 10.
	theta_noise = None
	env = gym.make('CartPole-v0')
	test_env = gym.make('CartPole-v0')
	x = KTD_Q(phi=rbf(10,2), gamma=1., P_init=P_init, theta0 = np.zeros(2*10,),theta_noise=theta_noise, eta=eta,
	             reward_noise=reward_noise, anum = 2)
	performance = []
	episode = 0
	step = 0
	while(episode < 1000):
		state = env.reset()
		action = np.random.choice(2,)
		n_state, reward, done, _ = env.step(action)
		x.update_V(state[-2:], action, n_state[-2:], reward)
		state = n_state
		step+=1
		if done or (step > 3000):
			step = 0
			
			if episode%50 == 0:
				performance.append(test_gym(x, test_env))
				print("Episode %d: %d"%(episode, performance[-1]))
			episode +=1
			state = env.reset()

	plt.plot(performance)
	plt.show()

def test(x, env):
	np_random,_  = seeding.np_random(None)
	state = env.reset(np_random)
	step = 0
	done = False
	while((not done) and (step < 3000)) :
		action =  np.argmax([np.dot(x.theta, x.phi(state,a)) for a in range(x.anum)])
		reward, n_state, done = env.observe(state, action)
		step+=1
		state = n_state
	return step

def test_gym(x, env):
	state = env.reset()
	step = 0
	done = False
	while((not done) and (step < 3000)) :
		action =  np.argmax([np.dot(x.theta, x.phi(state[-2:],a)) for a in range(x.anum)])
		print(action)
		env.render()
		n_state, reward, done, _ = env.step(action)
		step+=1
		state = n_state
	return step


if __name__ == "__main__":
	main_brl()