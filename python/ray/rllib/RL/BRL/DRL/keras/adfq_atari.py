# *************************************************************************************************
# Modified from rl/example/dqn_atari.py in keras-rl (https://github.com/matthiasplappert/keras-rl)
# by Heejin Chloe Jeong ( University of Pennsylvania, chloe.hjeong@gmail.com)
# @ Jan 2018
# *************************************************************************************************

from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import os, pdb, pickle
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
import keras.initializers as KInit
from keras.utils.generic_utils import get_custom_objects

from BRL.DRL.keras.adfq import ADFQAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def custom_initializer(shape, dtype=None):
    v = [args.init_mean]*int(shape[0]/2) + [args.init_std]*int(shape[0]/2)
    return K.constant(v, dtype=dtype)

class CustomInitializer:
    def __call__(self, shape, dtype=None):
        return custom_initializer(shape, dtype=dtype)

get_custom_objects().update({'custom_initializer': CustomInitializer})

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env_name', type=str, default='Breakout-v0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--double_dqn', type=int, default=1)
parser.add_argument('--dueling', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=512)
parser.add_argument('--nb_steps_warmup', type=int, default = 50000)
parser.add_argument('--epoch_steps', type=int, default = 50000)
parser.add_argument('--nb_train_steps', type=int, default = 5000000)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--target_model_update', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--memory_size', type=int, default=1000000)
parser.add_argument('--eps_max', type=float, default=1.)
parser.add_argument('--eps_min', type=float, default=.1)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_std', type=float, default=50.)
parser.add_argument('--device', type=str, default='/cpu:0')
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
with tf.device(args.device):
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    for _ in range(args.num_layers):
        model.add(Dense(args.num_units))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions*2, bias_initializer =  custom_initializer)) # mean and SD
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=args.memory_size, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=args.eps_max, value_min=args.eps_min, value_test=.05,
                                  nb_steps=1000000)
    test_policy = EpsGreedyQPolicy(eps=0.05)

    if bool(args.double_dqn):
        print("DOUBLE DQN")
    if bool(args.dueling):
        print("DUELING NETWORK")

    adfq = ADFQAgent(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, memory=memory,
                   processor=processor, nb_steps_warmup=args.nb_steps_warmup, gamma=args.gamma, target_model_update=args.target_model_update,
                   train_interval=4, delta_clip=1.)
    adfq.compile(Adam(lr=args.learning_rate), metrics=[])

    if args.mode == 'train':
        
        weights_filename = os.path.join(args.log_dir,'adfq_{}_weights.h5f').format(args.env_name)
        checkpoint_weights_filename = os.path.join(args.log_dir,'adfq_' + args.env_name + '_weights_{step}.h5f')
        log_filename = os.path.join(args.log_dir,'adfq_{}_log.json').format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=args.nb_train_steps/5)]
        callbacks += [FileLogger(log_filename, interval=args.epoch_steps/10)] # interval is for episode intervals. # of points nb_train_steps/(interval*1000)
        _, epoch_test = adfq.fit(env, callbacks=callbacks, nb_steps=args.nb_train_steps, log_interval=args.epoch_steps, nb_steps_epoch = args.epoch_steps)

        # After training is done, we save the final weights one more time.
        pickle.dump(epoch_test,open(os.path.join(args.log_dir, "adfq_epoch_test.pkl"),"wb"))
        adfq.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        adfq.test(env, nb_episodes=10, visualize=False)
    elif args.mode == 'test':
        weights_filename = 'adfq_{}_weights.h5f'.format(args.env_name)
        if args.weights:
            weights_filename = args.weights
        adfq.load_weights(weights_filename)
        adfq.test(env, nb_episodes=10, visualize=True)
