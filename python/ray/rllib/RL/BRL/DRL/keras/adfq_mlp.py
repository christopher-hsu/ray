from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import os, pdb

from keras.utils.generic_utils import get_custom_objects
from BRL.DRL.keras.adfq import ADFQAgent
from rl.memory import SequentialMemory
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.policy import BoltzmannQPolicy


def custom_initializer(shape, dtype=None):
    v = [args.init_mean]*int(shape[0]/2) + [args.init_std]*int(shape[0]/2)
    return K.constant(v, dtype=dtype)

class CustomInitializer:
    def __call__(self, shape, dtype=None):
        return custom_initializer(shape, dtype=dtype)

get_custom_objects().update({'custom_initializer': CustomInitializer})

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CartPole-v0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--eps_max', type=float, default=1.)
parser.add_argument('--eps_min', type=float, default=.1)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_std', type=float, default=50.)
parser.add_argument('--log_dir', type=str, default='.')
args = parser.parse_args()

# Get the environment and extract the number of actions.
ENV_NAME = args.env_name
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions*2, 
    bias_initializer =  custom_initializer)) # mean and SD
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
adfq = ADFQAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
adfq.compile(Adam(lr=1e-3), metrics=[])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
adfq.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
adfq.save_weights('adfq_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
adfq.test(env, nb_episodes=5, visualize=False)

