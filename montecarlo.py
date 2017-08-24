import numpy as np
from copy import deepcopy
from collections import OrderedDict
from time import time, sleep

class Rollout:

    """
    Abstract class for rollouts.
    """

    def rollout(self, state, history):
        """
        Takes the current state, and a history of states that have been visited
        during this rollout, and returns a move choice. History is available
        so that we can avoid cycles.
        """
        raise NotImplementedError

class RandomRollout(Rollout):
    """
    Generic light rollout that can be used for any state space.
    """
    def rollout(self, state, history=None):
        return np.random.choice(state.legal_moves())

# TODO: add an object that enforces the interface I need
# ie legal_moves, hashable_repr, is_solved, apply_move


class MonteCarlo:

    class Node:

        """
        Represents a particular state of the space being searched.
        Has an associated dictionary which maps all the legal moves from the
        state to an Edge object, which contains the necessary data to choose
        a move at any given point in a tree traversal.
        """

        def __init__(self, state, edges):
            self.state = state
            self.edges = edges

    class Edge:

        """
        Represents an edge in the search tree, which is effectively a
        (state, action) pair.
        All data required to make choices about taking this edge during tree
        traversal are stored: the number of times the edge has been traversed,
        the initial value assessment of the action, and the current value
        assessment of the action.
        """

        def __init__(self, action, initial_value):
            self.action = action
            self.num_visits = 0
            self.initial_value = initial_value
            self.value = initial_value

        def __str__(self):
            return "Edge({}, {} visits, {})".format(self.action,
                                                    self.num_visits,
                                                    self.value)


    def __init__(self,
                root,
                state_evaluation_function,
                moves_evaluation_function,
                search_time=100,
                rollout_type=RandomRollout(),
                rollout_depth=20):
        # The state we start our search from
        self.root = root
        # The function used to give states their initial value
        self.evaluate_state = state_evaluation_function
        # The function used to give moves their initial value
        self.evaluate_moves = moves_evaluation_function
        # The time limit for building the search tree, in seconds
        self.search_time = search_time
        # The function used to play the rollouts
        assert isinstance(rollout_type, Rollout)
        self.rollout = rollout_type.rollout
        # The max number of moves the rollout will proceed for w/o winning
        self.rollout_depth = rollout_depth
        # Dict of nodes in the search tree
        self.nodes = {}
        # Initialise the search tree with the root state
        self._add_to_tree(root)
        # Stat to keep track of how many traversals happened while building the tree
        self.num_loops = 0
        # Stat to keep track of the best state found so far (as judged by the eval function)
        self.best_state_value = 0
        self.best_state_moves = []
        self.best_state_str = ""
        # List of any solutions found while searching
        self.solutions = []

    def search(self):
        """
        This is the main use case of the MonteCarlo class. It runs the MCTS
        algorithm for a particular starting state (`self.root`). It will return
        a list of solutions it finds (if any).
        """
        # First, generate the search tree and all its stats
        timeout = time() + self.search_time
        while time() < timeout:
            print("Running search loop {}".format(self.num_loops + 1), end="\r")
            leaf_state, visited = self.select()
            self.expand(leaf_state)
            score, solution = self.simulate(leaf_state, visited)
            # If the rollout found a solution, add it to the list
            if solution:
                self.solutions.append(" ".join([str(v[1]) for v in visited] + solution))
                print("Solution found:")
                print(self.solutions[-1])
            self.update(visited, score)
            self.num_loops += 1
        print("Finished searching. Number of loops was {}".format(self.num_loops))
        # Then choose a move based on the information in the tree
        #key = self.root.hashable_repr()
        #moves = self.nodes[key].edges
        #best_move = max(moves, key=lambda m : moves[m].value)
        #for e in moves.values():
            #print(e)
        #print("Best move's value was {}".format(moves[best_move].value))
        #return best_move
        return self.solutions

    def select(self):
        """
        Traverses the part of the tree for which we have statistics.
        This is guided by the value of the node, divided by visit count, so as
        to balance exploitation and exploration.
        It stops once a state is reached that is not part of the search tree.
        """
        # List of states
        visited = []
        # Make a copy of the starting state so we can apply moves to it
        current_state = deepcopy(self.root)
        # Traverse the tree until we reach a leaf node
        count = 1
        invalid = set()
        while current_state.hashable_repr() in self.nodes:
            #print("Making selection move number {}".format(count))
            # Get valid moves
            moves = [m for m in current_state.legal_moves() if str(m) not in invalid]
            edges = self.nodes[current_state.hashable_repr()].edges
            # Choose one to make
            chosen_move = max(moves, key=lambda m : edges[m].value / (edges[m].num_visits + 1))
            # Add the chosen move's edge to the list of visited edges
            v = (current_state.hashable_repr(), chosen_move)
            if v in visited:
                invalid.add(str(chosen_move))
                continue
            else:
                visited.append((current_state.hashable_repr(), chosen_move))
                invalid.clear()
            # Apply the move (effectively a step down the tree)
            current_state.apply_move(chosen_move)
            #print(current_state.hashable_repr())
            count += 1

        return current_state, visited

    def expand(self, state):
        """
        Adds a new node to the search tree.
        This is called once the selection stage has finished.
        Delegating to _add_to_tree allows the option to only expand when certain
        conditions are met, e.g. only add a node to the tree once it has been
        visited n times.
        """
        self._add_to_tree(state)

    def simulate(self, state, visited):
        """
        Plays out the game from a newly expanded node.
        """
        # TODO: Prevent this from looping
        # might be best to switch rollout_moves to a dict keyed by the state
        # although then the moves wouldn't be in order... Could use collections.ordered_dict
        rollout_moves = OrderedDict()
        max_value = self.evaluate_state(state)
        for step in range(self.rollout_depth):
            if state.is_solved():
                return max_value, rollout_moves.values()
            next_move = self.rollout(state=state)
            state.apply_move(next_move)
            rollout_moves[state.hashable_repr()] = str(next_move)
            max_value = max(max_value, self.evaluate_state(state))
            if max_value > self.best_state_value:
                self.best_state_value = max_value
                self.best_state_moves = " ".join([str(v[1]) for v in visited] + ["*"] + rollout_moves.values())
                self.best_state_str = str(state)
        return max_value, None

    def update(self, visited, outcome):
        """
        Updates all nodes visited in this game with statistics representing
        the outcome, e.g. the visit count for all nodes will be incremented.

        Visited will be in order, from the root to the leaf node (and the newly
        added node).
        Should probably traverse backwards to allow for discounting value
        as we move away from the solution.
        """
        for state, move in visited[::-1]:
            edge = self.nodes[state].edges[move]
            # Increment visit count
            edge.num_visits += 1
            # Update the value of the edge
            edge.value = max(edge.value, outcome)


    def _add_to_tree(self, state):
        # Get a (short) string that can be used to reference the node
        key = state.hashable_repr()
        if key in self.nodes:
            # This node is already in the tree
            return
        edges = {}
        # Get prior probabilities for each move
        move_values = self.evaluate_moves(state)
        # Create an Edge object for each move, with a value
        for i, move in enumerate(state.legal_moves()):
            initial_value = move_values[i]
            edges[move] = MonteCarlo.Edge(move, initial_value)
        # Create a Node object for this state
        node = MonteCarlo.Node(state, edges)
        # Add it to the tree
        self.nodes[key] = node




if __name__ == "__main__":

    import mincube as rubiks
    import valuenetwork as valnet
    import slnetwork as polnet
    from rollouts import policy_rollout

    state_value = lambda s : 1 / valnet.evaluate(s)

    # Scramble taken from qqtimer
    #scramble = "R' B U B U B' D' F L B2 L2 B2 U2 D F2 B2 U' L2 D L2"
    scramble = "R' B U B U B' D'"

    cube = rubiks.Cube(alg=scramble)

    mc = MonteCarlo(cube, state_value, polnet.evaluate, rollout_type=policy_rollout)
    solutions = mc.search()
    if solutions:
        for sol in sorted(solutions, key=lambda x : len(x))[:5]:
            print("{} ({})".format(sol, sol.count(" ") + 1))
    else:
        print("No solutions found.")
    print("Best position found was valued at {}".format(
                    format(mc.best_state_value, ".3f"), end=" "))
    print("after the moves {}".format(mc.best_state_moves))

    print("No of states in the tree was: {}".format(len(mc.nodes)))
