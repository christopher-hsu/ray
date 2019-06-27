import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

from BRL.DRL.keras.adfq import ADFQAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = 'CartPole-v0'
init_mean = 1.0
init_std = 30.0

def custom_initializer(shape, dtype=None):
    v = [init_mean]*int(shape[0]/2) + [init_std]*int(shape[0]/2)
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
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions*2, bias_initializer =  custom_initializer))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
test_policy = EpsGreedyQPolicy(eps=0.05)
adfq = ADFQAgent(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, memory=memory,
                  nb_steps_warmup=10, target_model_update=10)
adfq.compile(Adam(lr=1e-3), metrics=[])
    
#weights_filename = os.path.join(args.log_dir,'adfq_{}_weights.h5f').format(args.env_name)
#checkpoint_weights_filename = os.path.join(args.log_dir,'adfq_' + args.env_name + '_weights_{step}.h5f')
#log_filename = os.path.join(args.log_dir,'adfq_{}_log.json').format(args.env_name)
#callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=args.nb_train_steps/5)]
#callbacks += [FileLogger(log_filename, interval=args.epoch_steps/10)] # interval is for episode intervals. # of points nb_train_steps/(interval*1000)
_, epoch_test = adfq.fit(env, nb_steps=50000, log_interval=100, nb_steps_epoch = 100, verbose=2)

# After training is done, we save the final weights one more time.
#pickle.dump(epoch_test,open(os.path.join(args.log_dir, "adfq_epoch_test.pkl"),"wb"))
adfq.save_weights(weights_filename, overwrite=True)

adfq.test(env, nb_episodes=5, visualize=False)
pdb.set_trace()

