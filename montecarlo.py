import numpy as np
from copy import deepcopy
from time import time


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


    def __init__(self,
                root,
                all_possible_moves,
                search_time=10,
                rollout_type=random_rollout,
                rollout_depth=20):
        self.root = root
        self.nodes = {} # Need to add root to this here with appropriate values
        self.possible_moves = all_possible_moves
        self.search_time = search_time
        self.rollout = rollout_type
        self.rollout_depth = rollout_depth

    def choose_move(self):
        """
        This is the main use case of the MonteCarlo class. It runs the MCTS
        algorithm for a particular starting state (`self.root`) and returns a
        move choice based on this.
        """
        # First, generate the search tree and all its stats
        start_time = time()
        while (time() - start_time) < self.search_time:
            # Selection
            leaf_state, visited = self.selection()
            # Expansion
            self.expand(leaf_state)
            # Simulation
            # Update
        # Then choose a move based on the information in the tree
        key = self.root.hashable_repr()
        moves = self.nodes[key].edges
        best_move = max(moves, key=lambda edge : edge.value)
        return best_move.action

    def selection(self):
        """
        Traverses the part of the tree for which we have statistics.
        This is guided by the value of the node, divided by visit count, so as
        to balance exploitation and exploration.
        It stops once a state is reached that is not part of the search tree.
        """
        # List of states
        visited = []
        current_state = deepcopy(self.root)
        while current_state.hashable_repr() in self.nodes:
            moves = self.nodes[current_state].edges
            chosen_move = _select(moves)
            visited.append((current_state.hashable_repr(), chosen_move))
            current_state = current_state.apply_move(chosen_move)
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

    def simulate(self, start_state):
        """
        Plays out the game from a newly expanded node.
        """
        self.rollout(state=start_state)

    def update(self, visited, outcome):
        """
        Updates all nodes visited in this game with statistics representing
        the outcome, e.g. the visit count for all nodes will be incremented.

        Visited will be in order, from the root to the leaf node (and the newly
        added node).
        Should probably traverse backwards to allow for discounting value
        as we move away from the solution.
        """
        for state, move in visited:
            # Increment visit count
            self.nodes[state].edges[move].num_visits += 1


    def _add_to_tree(self, state):
        edges = {}
        for move in self.possible_moves:
            initial_value = 0 # Need to make this call the networks
            edges[move] = Edge(move, initial_value)
        node = Node(state, edges)
        self.nodes[node.hashable_repr()] = node



    def random_rollout(self, **kwargs):
        state = kwargs["state"]
        for _ in range(self.rollout_depth):
            state.apply_move(np.random.choice(state.legal_moves()))
            if state.is_solved():
                return 1
        return 0


def _select(moves):
    """
    Chooses a move for the selection phase.
    `moves` is the node.edges dictionary for the current node.
    """
    # Random for now
    return np.random.choice(moves)
