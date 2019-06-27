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
from gym.wrappers import Monitor

import os, pdb, pickle, datetime, json
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import keras.initializers as KInit
from keras.utils.generic_utils import get_custom_objects
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

# Default are from Hasselt's Double DQN paper.
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env_name', type=str, default='Breakout-v0')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--double_dqn', type=int, default=0)
parser.add_argument('--dueling', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--init_step', type=int, default=0) # When you want to start from learned weights.
parser.add_argument('--nb_train_steps', type=int, default = 50000000)
parser.add_argument('--nb_steps_warmup', type=int, default = 50000)
parser.add_argument('--no_ops', type=int, default=0)
parser.add_argument('--nb_max_start_steps', type = int, default=0)
parser.add_argument('--epoch_steps', type=int, default = 50000)
parser.add_argument('--target_model_update', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--memory_size', type=int, default=500000)
parser.add_argument('--eps_max', type=float, default=1.)
parser.add_argument('--eps_min', type=float, default=.1)
parser.add_argument('--init_mean', type =float, default=1.)
parser.add_argument('--init_sd', type=float, default=50.)
parser.add_argument('--sd_max', type=float, default=10.)
parser.add_argument('--sd_min', type=float, default=0.01)
parser.add_argument('--sd_steps', type=int, default=50000000)
parser.add_argument('--gpu_memory', type=float, default=0.1)
parser.add_argument('--loss_type', type=str, default='kl')
parser.add_argument('--device', type=str, default='/cpu:0')
parser.add_argument('--alg', choices=['dqn','adfq'], default='dqn')
parser.add_argument('--record',type=int, default=0)

args = parser.parse_args()
    
# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
if args.record == 1:
    env = Monitor(env, directory=args.log_dir)
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

    if args.alg=='dqn':
        from BRL.DRL.keras.dqn import DQNAgent
        agent_class = DQNAgent
        model.add(Dense(nb_actions))
        
    elif args.alg== 'adfq':
        from BRL.DRL.keras.adfq import ADFQAgent
        agent_class = ADFQAgent
        def custom_initializer(shape, dtype=None):
            v = [args.init_mean]*int(shape[0]/2) + [args.init_sd]*int(shape[0]/2)
            return K.constant(v, dtype=dtype)
        class CustomInitializer:
            def __call__(self, shape, dtype=None):
                return custom_initializer(shape, dtype=dtype)
        get_custom_objects().update({'custom_initializer': CustomInitializer})
        model.add(Dense(nb_actions*2, bias_initializer =  custom_initializer)) # mean and SD

    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=args.memory_size, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, 
                            value_test=.05, nb_steps=1000000)
    test_policy = EpsGreedyQPolicy(eps=0.05)

    if bool(args.double_dqn):
        print("DOUBLE DQN")
    if bool(args.dueling):
        print("DUELING NETWORK")

    if 'GPU' in args.device.split(':'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory
        config.gpu_options.polling_inactive_delay_msecs = 25
        session = tf.Session(config=config)
        K.set_session(session)

    if args.alg== 'adfq': 
        agent = agent_class(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, 
                    memory=memory, processor=processor, nb_steps_warmup=args.nb_steps_warmup, 
                    sd_min = args.sd_min, sd_max = args.sd_max, sd_steps = args.sd_steps,
                    gamma=args.gamma, batch_size=args.batch_size, target_model_update=args.target_model_update, 
                    enable_double_dqn = bool(args.double_dqn), enable_dueling_network = bool(args.dueling), 
                    train_interval=4, delta_clip=1., no_ops = args.no_ops)
        agent.compile(Adam(lr=args.learning_rate, clipnorm=10.0), loss_type = args.loss_type, metrics=[])
    elif args.alg == 'dqn':
        agent = agent_class(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, 
                    memory=memory, processor=processor, nb_steps_warmup=args.nb_steps_warmup, 
                    gamma=args.gamma, batch_size=args.batch_size, target_model_update=args.target_model_update, 
                    enable_double_dqn = bool(args.double_dqn), enable_dueling_network = bool(args.dueling), 
                    train_interval=4, delta_clip=1., no_ops = args.no_ops)
        #agent.compile(Adam(lr=args.learning_rate, clipnorm = 10.0), metrics=['mae'])
        agent.compile(RMSprop(lr=args.learning_rate, clipnorm = 10.0, rho = 0.95), metrics=['mae'])
    directory = os.path.join(args.log_dir, datetime.datetime.now().strftime("%m%d%H%M"))
    if args.mode == 'train':
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            ValueError("The directory already exists...", directory)
        json.dump(vars(args), open(os.path.join(directory, 'learning_prop.json'), 'wb'))

        if args.weights: 
            agent.load_weights(args.weights)
            agent.nb_steps_warmup += args.init_step

        weights_filename = os.path.join(directory, (args.alg + '_{}_weights.h5f').format(args.env_name))
        checkpoint_weights_filename = os.path.join(directory, args.alg + '_' + args.env_name + '_weights_{step}.h5f')
        log_filename = os.path.join(directory, (args.alg + '_{}_log.json').format(args.env_name))
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)]
        callbacks += [FileLogger(log_filename, interval=args.epoch_steps/2)] # interval is for episode intervals. # of points nb_train_steps/(interval*1000)
        _ = agent.fit(env, callbacks=callbacks, nb_steps=args.nb_train_steps, 
            log_interval=args.epoch_steps, nb_steps_epoch = args.epoch_steps, 
            nb_max_start_steps=args.nb_max_start_steps, init_step = args.init_step, epoch_file_path = os.path.join(directory, args.alg + '_epoch_test.pkl'))

        # After training is done, we save the final weights one more time.
        agent.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        agent.test(env, nb_episodes=10, visualize=False)


    elif args.mode == 'test':
        weights_filename = os.path.join(directory,(args.alg + '_{}_weights.h5f').format(args.env_name))
        if args.weights:
            weights_filename = args.weights
        agent.load_weights(weights_filename)
        agent.test(env, nb_episodes=10, visualize=True)
