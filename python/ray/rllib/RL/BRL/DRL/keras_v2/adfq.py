from __future__ import division
import warnings

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.losses import kullback_leibler_divergence

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
import BRL.brl_util_new as util

import pdb
import tensorflow as tf

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class AbstractADFQAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractADFQAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_stats(self, state_batch):
        batch = self.process_state_batch(state_batch)
        stats = self.model.predict_on_batch(batch)
        assert stats.shape == (len(state_batch), self.nb_actions*2)
        return stats

    def compute_q_means(self, state):
        q_stats = self.compute_batch_q_stats([state]).flatten()
        assert q_stats.shape == (self.nb_actions*2,)
        return q_stats[:self.nb_actions]

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


class ADFQAgent(AbstractADFQAgent):
    """
    # Arguments 
        model__: A Keras model. 
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py). 
        test_policy__: A Keras-rl policy. 
    """
    def __init__(self, model, policy=None, test_policy=None, varTH=1e-5, *args, **kwargs):
        super(ADFQAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. ADFQ expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, 2*self.nb_actions):
            raise ValueError('Model output "{}" has invalid shape. ADFQ expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy
        self.varTH = np.float32(varTH)

        # State.
        self.reset_states()

    def get_config(self):
        config = super(ADFQAgent, self).get_config()
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config
    def mean_mu(self, y_true, y_pred):
        return K.mean(K.max(y_pred[:,:self.nb_actions], axis=-1))

    def mean_rho(self, y_true, y_pred):
        return K.mean(K.max(y_pred[:,self.nb_actions:], axis=-1))

    def compile(self, optimizer, metrics=[]):
        metrics += [self.mean_mu]  # register default metrics
        metrics += [self.mean_rho]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_update, y_pred, mask = args
            # y_update : [updated_mean, updated_sds]
            y_pred_mean = K.sum(mask*y_pred[:, :self.nb_actions], axis=-1, keepdims=True) # element-wise mask
            y_pred_sd  = K.clip(K.log(1.+K.exp(K.sum(mask*y_pred[:, self.nb_actions:], axis=-1, keepdims=True))), np.sqrt(self.varTH), 50.0)
            y_update_mean = y_update[:,0]
            y_update_sd = y_update[:,1]
            #loss = huber_loss(y_pred_mean, y_update_mean, 1e+3) + huber_loss(y_pred_sd, y_update_sd, 1e+3)
            #loss = tf.reduce_sum(tf.contrib.distributions.kl_divergence(
            #        tf.distributions.Normal(loc=y_update_mean, scale=y_update_sd),
            #         tf.distributions.Normal(loc=y_pred_mean, scale=y_pred_sd)), name = 'loss')
            
            loss = -0.5*K.mean(K.square(y_update_sd/y_pred_sd) - 2.*K.log(y_update_sd/y_pred_sd) \
                + K.square((y_update_mean - y_pred_mean)/y_pred_sd))
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_update = Input(name='y_update', shape=(2,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_update, y_pred, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_update, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_update, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_update, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_means(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(1. if e.terminal1 else 0.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            action_batch = np.array(action_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute the c_mean and c_vars given state0, and the n_means, and n_vars given state1.
            c_stats = self.target_model.predict_on_batch(state0_batch)
            n_stats = self.target_model.predict_on_batch(state1_batch)
            c_mean = c_stats[np.arange(self.batch_size), action_batch]
            c_var  = np.clip(np.square(np.log(1. + np.exp(c_stats[np.arange(self.batch_size), action_batch+self.nb_actions]))), self.varTH, 50.0**2)
            n_means = n_stats[:,:self.nb_actions]
            n_vars = np.clip(np.square(np.log(1. + np.exp(n_stats[:,self.nb_actions:]))), self.varTH, 50.0**2)
            # Compute new mean and new variance in ADFQ

            new_mean, new_var, _ = util.posterior_approx(n_means, n_vars, c_mean, c_var, reward_batch, self.gamma,
                                                        terminal1_batch, varTH = self.varTH, REW_VAR = 1e-3, batch=True )

            new_stats = np.concatenate((new_mean[:,np.newaxis], np.sqrt(new_var[:,np.newaxis])),axis=1)
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * c_mean
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, ( mask, R, action) in enumerate(zip(masks, Rs, action_batch)):
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            new_stats = np.array(new_stats).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [new_stats, masks], [dummy_targets, new_stats])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)

