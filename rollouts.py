"""
Different rollout functions for use with MCTS.
Rollouts take parameters via kwargs, and should return a reward value.
"""

import valuenetwork as valnet
import slnetwork as slnet
import numpy as np
import mincube as rubik

"""
Don't need to use kwargs here: the call to the function can unpack a dict
using the ** prefix into explicit keyword arguments here.

so `f(**{"val":1})` will work for `def f(val=None)`
"""

def strict_policy_rollout(state):
    # This always chooses the highest rated move. It's very liable to loop.
    move_probs = slnet.evaluate(state)
    index = np.argmax(move_probs)
    return state.legal_moves()[index]

def policy_rollout(state):
    moves = state.legal_moves()
    move_probs = slnet.evaluate(state)
    return np.random.choice(moves, p=move_probs)

def value_rollout(state):
    value = lambda m : valnet.evaluate(deepcopy(state).apply_move(m))
    return max(state.legal_moves(), key=value)

def mixed_rollout(state, mixing_factor=0.5):
    pass
