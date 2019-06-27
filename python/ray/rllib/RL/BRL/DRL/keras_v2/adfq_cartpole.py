import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

from adfq import ADFQAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'
init_mean = 50.0
init_sd = 50.0

def custom_initializer(shape, dtype=None):
    v = [init_mean]*int(shape[0]/2) + [init_sd]*int(shape[0]/2)
    return K.constant(v, dtype=dtype)

class CustomInitializer:
    def __call__(self, shape, dtype=None):
        return custom_initializer(shape, dtype=dtype)

get_custom_objects().update({'custom_initializer': CustomInitializer})

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dense(32))
#model.add(Activation('relu'))
model.add(Dense(2*nb_actions, bias_initializer =  custom_initializer))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
adfq = ADFQAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=10, policy=policy, varTH = 1e-5)
adfq.compile(Adam(lr=1e-3), metrics=[])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
adfq.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
adfq.save_weights('adfq_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
adfq.test(env, nb_episodes=5, visualize=True)
