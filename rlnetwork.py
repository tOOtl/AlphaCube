"""
Reinforcement learning network using slnetwork to initialise weights, and
keras-rl to run the training protocol.

PLEASE NOTE:
keras-rl does not support python 3, and so this file, unlike the rest of the
project, is in python 2.7. The interface between this program and the rest of
the project is done using keras models saved in .h5 files, and training data
stored in .txt files.
"""

from __future__ import division, print_function

from keras.models import load_model, Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import rl.policy

from gym.core import Env
from gym.spaces.discrete import Discrete

import numpy as np
import mincube as rubiks
from features import get_features

PATH_BASE = "/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/"
SL_PATH = PATH_BASE + "policy/rl_initialiser_aggregate_with_rand_move_scrambles.h5"
RL_PATH = PATH_BASE + "policy/rl_test.h5"
MODEL = None
FAILURE_REWARD = -1

class CubeEnv(Env):
    """
    Wrapper for Cube objects to match the OpenAI gym interface for problems.
    """

    def __init__(self, move_limit=30, scramble_len=25):
        self.move_limit = move_limit
        self.scramble_len = scramble_len
        self.cube = rubiks.Cube()
        self.cube.scramble(scramble_len=scramble_len)
        self.num_moves = 0

    action_space = Discrete(18)
    observation_space = None
    reward_range = (FAILURE_REWARD, 1)

    def _step(self, action):
        #print "Cube state is:"
        #print self.cube
        #print "Move chosen is: ", rubiks.MOVES[action]
        self.cube.apply_move(rubiks.MOVES_OBJS[action])
        self.num_moves += 1
        observation = get_features(self.cube)
        reward = 0 #FAILURE_REWARD * (self.num_moves // self.move_limit)
        done = False
        if self.cube.is_solved():
            done = True
            # Bonus is linearly proportional to the shortness of the solution
            # to reward shorter solutions more
            bonus = 0 # 0.5 * (1 - ((self.num_moves - 1) / self.move_limit))
            reward = 1 + bonus
        elif self.num_moves == self.move_limit:
            done = True
            reward = FAILURE_REWARD
        info = {}
        return (observation, reward, done, info)

    def _reset(self):
        self.cube = rubiks.Cube()
        self.cube.scramble(scramble_len=self.scramble_len)
        self.num_moves = 0
        observation = get_features(self.cube)
        return observation

    def _render(self, mode='human', close=False):
        """
        Used for visualisation. Not implemented here.
        """
        return

    def _seed(self, seed=None):
        # Docs don't make it clear what this is meant to do, so let's just
        # assume this is right
        if seed:
            np.random.seed(seed)

class WeightedEpsGreedyQPolicy(rl.policy.EpsGreedyQPolicy):

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            # Note, this can still sample the same move that argmax would.
            # This means that exploration is proportional to confidence: if one
            # move is overwhelmingly likely, it will still probably be chosen
            # even if it is selected by this random choice rather than argmax
            action = np.random.choice(range(len(q_values)), p=q_values)
        else:
            action = np.argmax(q_values)
        return action

def top_3(y_true, y_pred):
    """
    Classes a prediction as a success if the move tagged as correct is within
    the top 3 moves chosen by the net.
    Redefined here from slnetwork to avoid py2 vs py3 issues.
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def get_initial_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(SL_PATH, custom_objects={"top_3":top_3})
        """
        MODEL = Sequential()
        MODEL.add(Dense(32, activation="relu", input_shape=(1,420)))
        MODEL.add(Dense(100, activation="relu"))
        MODEL.add(Dense(100, activation="relu"))
        MODEL.add(Flatten())
        MODEL.add(Dense(18, activation="softmax"))

        for i in (1, 2):
            MODEL.layers[i].set_weights(w.layers[i].get_weights())
        MODEL.layers[-1].set_weights(w.layers[3].get_weights())
        """

    return MODEL

def train():
    model = get_initial_model()
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = rl.policy.EpsGreedyQPolicy(eps=0.1)
    dqn = DQNAgent(model=model,
                    nb_actions=18,
                    memory=memory,
                    nb_steps_warmup=1000,
                    target_model_update=1e-2,
                    policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    #dqn.load_weights(SL_PATH)


    for n in range(4, 5):
        print("\nStarting %s-move scrambles\n" % n)
        env = CubeEnv(scramble_len=n, move_limit=n*2)
        steps = 500000 #min(100 * (18**n), 50000)
        dqn.fit(env, nb_steps=steps, verbose=1)
        dqn.save_weights(RL_PATH, overwrite=True)

    return dqn
    #dqn.save_weights(PATH_BASE + "policy/rl_backup_" + str(time.time()) + ".h5")

if __name__ == "__main__":

    dqn = train()
    env = CubeEnv(scramble_len=5, move_limit=10)
    np.random.seed(17)
    dqn.test(env, nb_episodes=10, visualize=False)
    quit()




    print("Testing DQN with SL weights:")
    model_sl = load_model(SL_PATH, custom_objects={"top_3":top_3})
    memory_sl = SequentialMemory(limit=50000, window_length=1)
    policy_sl = WeightedEpsGreedyQPolicy(eps=0.1)
    dqn_sl = DQNAgent(model=model_sl,
                    nb_actions=18,
                    memory=memory_sl,
                    nb_steps_warmup=1000,
                    target_model_update=1e-2,
                    policy=policy_sl)
    dqn_sl.compile(Adam(lr=1e-3), metrics=['mae'])
    rng_state = np.random.get_state()
    dqn_sl.test(env, nb_episodes=10, visualize=False)

    print("\nTesting DQN with RL weights:")
    model_rl = load_model(SL_PATH, custom_objects={"top_3":top_3})
    memory_rl = SequentialMemory(limit=50000, window_length=1)
    policy_rl = WeightedEpsGreedyQPolicy(eps=0.1)
    dqn_rl = DQNAgent(model=model_rl,
                    nb_actions=18,
                    memory=memory_rl,
                    nb_steps_warmup=1000,
                    target_model_update=1e-2,
                    policy=policy_rl)
    dqn_rl.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn_rl.load_weights(RL_PATH)
    np.random.set_state(rng_state)
    dqn_rl.test(env, nb_episodes=10, visualize=False)
