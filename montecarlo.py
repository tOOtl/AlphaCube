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
            return "Edge(action={}, num_visits={}, value={})".format(
                                                                self.action,
                                                                self.num_visits,
                                                                self.value
                                                                )


    def __init__(self,
                root,
                state_evaluation_function,
                moves_evaluation_function,
                rollout_type=RandomRollout(),
                rollout_depth=20):
        # The state we start our search from
        self.root = root
        # The function used to give states their initial value
        self.evaluate_state = state_evaluation_function
        # The function used to give moves their initial value
        self.evaluate_moves = moves_evaluation_function
        # The function used to play the rollouts
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

    def single_search(self, search_time=100):
        """
        This is the main use case of the MonteCarlo class. It runs the MCTS
        algorithm for a particular starting state (`self.root`). It will return
        a list of solutions it finds (if any).
        """
        timeout = time() + search_time
        while time() < timeout:
            print("Running search loop {}".format(self.num_loops + 1), end="\r")
            leaf_state, visited = self.select()
            self.expand(leaf_state)
            score, solution = self.simulate(leaf_state, visited)
            # If the rollout found a solution, add it to the list
            if solution:
                self.solutions.append(" ".join([str(v[1]) for v in visited] + list(solution)))
            self.update(visited, score)
            self.num_loops += 1
        print("Finished searching. Number of loops was {}".format(self.num_loops))
        return self.solutions

    def choose_move(self, search_time=2, search_depth=10):
        timeout = time() + search_time
        num_loops = 0
        while time() < timeout:
            leaf_state, visited = self.select()
            self.expand(leaf_state)
            sim_depth = search_depth - len(visited)
            score, solution = self.simulate(leaf_state, visited, sim_depth)
            if solution:
                self.solutions.append(" ".join([str(v[1]) for v in visited] + list(solution)))
            self.update(visited, score)
            num_loops += 1
        key = self.root.hashable_repr()
        moves = self.nodes[key].edges
        best_move = max(moves, key=lambda m : moves[m].value)
        return best_move


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

    def simulate(self, state, visited, depth=None):
        """
        Plays out the game from a newly expanded node.
        """
        if not depth: depth = self.rollout_depth
        rollout_moves = OrderedDict()
        max_value = self.evaluate_state(state)
        for step in range(depth):
            if state.is_solved():
                return max_value, rollout_moves.values()
            next_move = self.rollout(state=state, history=rollout_moves)
            state.apply_move(next_move)
            rollout_moves[state.hashable_repr()] = str(next_move)
            max_value = max(max_value, self.evaluate_state(state))
            if max_value > self.best_state_value:
                self.best_state_value = max_value
                self.best_state_moves = " ".join([str(v[1]) for v in visited]
                                                + ["*"]
                                                + list(rollout_moves.values())
                                                )
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
    import rollouts

    r = rollouts.ProbabilisticPolicyRollout()

    nb_trials = 50
    for scramble_len in range(11, 15):
        nb_successes = 0
        for trial in range(nb_trials):
            print("Running trial {}".format(trial + 1), end="\r")
            scramble = rubiks.get_scramble(scramble_len)
            cube = rubiks.Cube(alg=scramble)
            moves_made = []
            while len(moves_made) < (scramble_len * 2):
                mc = MonteCarlo(cube, valnet.evaluate, polnet.evaluate, rollout_type=r)
                move = mc.choose_move(search_time=1)
                cube.apply_move(move)
                moves_made.append(move)
                if cube.is_solved():
                    nb_successes += 1
                    break
            #print("Scramble: {}".format(scramble))
            #print("Moves made: {}".format(" ".join([str(m) for m in moves_made])))
        print("Depth {}:".format(scramble_len).ljust(10), end="")
        print("{}%".format((nb_successes / nb_trials) * 100).ljust(10))
    quit()




    solutions = mc.single_search()
    print("Scramble was: {}".format(scramble))
    if solutions:
        for sol in sorted(solutions, key=len)[:5]:
            print("{} ({})".format(sol, sol.count(" ") + 1))
    else:
        print("No solutions found.")
    print("Best position found was valued at {}".format(
                    format(mc.best_state_value, ".3f"), end=" "))
    print("after the moves {}".format(mc.best_state_moves))

    print("No of states in the tree was: {}".format(len(mc.nodes)))
