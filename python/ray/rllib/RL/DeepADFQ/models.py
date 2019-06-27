"""This code was modified from a OpenAI baseline code - baselines0/baselines0/deepq/models.py
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def wrap_atari_dqn(env):
    from baselines0.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)

def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)

        bias_init = [init_mean for _ in range(int(num_actions/2))]
        bias_init.extend([-np.log(init_sd) for _ in range(int(num_actions/2))])
        q_out = layers.fully_connected(out, 
            num_outputs=num_actions, 
            activation_fn=None,
            weights_initializer=tf.zeros_initializer(),
            biases_initializer=tf.constant_initializer(bias_init))

        return q_out


def mlp(hiddens=[], layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, init_mean=init_mean, init_sd = init_sd, *args, **kwargs)


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs, # number of output filters
                                           kernel_size=kernel_size, # filter spatial dimension
                                           stride=stride, 
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            #action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
            bias_init = [init_mean for _ in range(int(num_actions/2))]
            bias_init.extend([-np.log(init_sd) for _ in range(int(num_actions/2))])
            action_scores = layers.fully_connected(action_out, 
                num_outputs=num_actions, 
                activation_fn=None,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=tf.constant_initializer(bias_init))

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, init_mean=init_mean, init_sd = init_sd, *args, **kwargs)

def _cnn_plus_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    
    with tf.variable_scope(scope, reuse=reuse):
        """
            inpt: vectorized image input + continuous input
        """
        im_size = 50
        out = tf.reshape(tf.slice(inpt, [0, 0],[-1, im_size*im_size]), [-1, im_size, im_size, 1])
        mlp_inpt = tf.slice(inpt, [0, im_size*im_size],[-1, int(inpt.shape[1])-im_size*im_size])

        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
                out = layers.avg_pool2d(out,
                                        kernel_size=2,
                                        stride=2,
                                        padding='VALID')
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = tf.concat([conv_out, mlp_inpt], 1)
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            #action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
            bias_init = [init_mean for _ in range(int(num_actions/2))]
            bias_init.extend([-np.log(init_sd) for _ in range(int(num_actions/2))])
            action_scores = layers.fully_connected(action_out, 
                num_outputs=num_actions, 
                activation_fn=None,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=tf.constant_initializer(bias_init))

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = tf.concat([conv_out, mlp_inpt], 1)
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_plus_mlp(convs, hiddens, dueling=False, layer_norm=False, init_mean = 1.0, init_sd = 20.0):
    """This model takes an image input and a 1D vector input 
     and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_plus_mlp(convs, hiddens, dueling, layer_norm=layer_norm, init_mean=init_mean, init_sd = init_sd, *args, **kwargs)

