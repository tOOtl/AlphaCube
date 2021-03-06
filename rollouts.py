"""
Different rollout functions for use with MCTS.
Rollouts take parameters via kwargs, and should return a reward value.
"""

from montecarlo import Rollout
from copy import deepcopy

import valuenetwork as valnet
import slnetwork as slnet
import numpy as np
import mincube as rubik


class StrictPolicyRollout(Rollout):
    # Always chooses the highest rated move that won't lead to a cycle
    def rollout(self, state, history):
        # Get the highest rated move
        move_probs = slnet.evaluate(state)
        index = np.argmax(move_probs)
        # Check that it will not bring us back to a visited state
        state_copy = deepcopy(state)
        state_copy.apply_move(state.legal_moves()[index])
        if state_copy.hashable_repr() in history:
            # If it does, set that move's probability to zero
            move_probs[index] = 0
            # Rescale the array so that it sums to 1
            move_probs = move_probs/move_probs.sum()
            # and then run argmax again to get the second best move
            index = np.argmax(move_probs)
        # Return the chosen move
        return state.legal_moves()[index]

class ProbabilisticPolicyRollout(Rollout):
    # Does a random choice of moves weighted by the policy network
    def rollout(self, state, history):
        # Get the highest rated move
        move_probs = slnet.evaluate(state)
        moves = state.legal_moves()
        index = np.random.choice(range(len(moves)), p=move_probs)
        move = moves[index]
        # Check that it will not bring us back to a visited state
        state_copy = deepcopy(state)
        state_copy.apply_move(move)
        if state_copy.hashable_repr() in history:
            # If it does, set that move's probability to zero
            move_probs[index] = 0
            # Rescale the array so that it sums to 1
            move_probs = move_probs/move_probs.sum()
            # and then run random choice again
            move = np.random.choice(state.legal_moves(), p=move_probs)
        # Return the chosen move
        return move

class ValueRollout(Rollout):
    # Chooses the move that leads to the highest value state
    def rollout(self, state, history):
        # Create a list of each state reachable from this one, containing tuples
        # (state, value assessed by valnet, move needed to reach the state)
        states = []
        for move in state.legal_moves():
            state_copy = deepcopy(state)
            state_copy.apply_move(move)
            states.append(state_copy)
        values = [valnet.evaluate(s) for s in states]
        state_val_move = list(zip(states, values, state.legal_moves()))
        # Sort according to descending value
        state_val_move.sort(key=lambda x : x[1], reverse=True)
        # Choose the highest value move that hasn't been visited
        for svm in state_val_move:
            if svm[0].hashable_repr() not in history:
                return svm[2]
        # If, somehow, all states have been visited, just return the highest
        # value move
        return state_val_move[0][2]
