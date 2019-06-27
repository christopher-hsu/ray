from baselines0.categorial_deepq import models  # noqa
from baselines0.categorial_deepq.build_graph import build_act, build_train  # noqa
from baselines0.categorial_deepq.simple import learn, load, iqr  # noqa
from baselines0.categorial_deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines0.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)

