import numpy as np
from copy import deepcopy
from time import time, sleep

def random_rollout(state):
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
            return "Edge({}, {} visits, {})".format(self.action,
                                                    self.num_visits,
                                                    self.value)


    def __init__(self,
                root,
                state_evaluation_function,
                moves_evaluation_function,
                search_time=10,
                rollout_type=random_rollout,
                rollout_depth=20):
        # The state we start our search from
        self.root = root
        # The function used to give states their initial value
        self.evaluate_state = state_evaluation_function
        # The function used to give moves their initial value
        self.evaluate_moves = moves_evaluation_function
        # The time limit for building the search tree, in seconds
        self.search_time = search_time
        # The function used to play the rollouts, should return a value in [0, 1]
        self.rollout = rollout_type
        # The max number of moves the rollout will proceed for w/o winning
        self.rollout_depth = rollout_depth
        # Dict of nodes in the search tree
        self.nodes = {}
        # Initialise the search tree with the root state
        self._add_to_tree(root)
        # Stat to keep trach of how many traversals happened while building the tree
        self.num_loops = 0
        # List of any solutions found while searching
        self.solutions = []

    def choose_move(self):
        """
        This is the main use case of the MonteCarlo class. It runs the MCTS
        algorithm for a particular starting state (`self.root`) and returns a
        move choice based on this.
        """
        # First, generate the search tree and all its stats
        timeout = time() + self.search_time
        while time() < timeout:
            print("Starting search loop {}".format(self.num_loops + 1), end="\r")
            leaf_state, visited = self.select()
            self.expand(leaf_state)
            score, solution = self.simulate(leaf_state)
            # If the rollout found a solution, add it to the list
            if solution:
                self.solutions.append(" ".join([str(v[1]) for v in visited] + solution))
            self.update(visited, score)
            self.num_loops += 1
        print("Finished searching. Number of loops was {}".format(self.num_loops))
        # Then choose a move based on the information in the tree
        key = self.root.hashable_repr()
        moves = self.nodes[key].edges
        best_move = max(moves, key=lambda m : moves[m].value)
        #for e in moves.values():
            #print(e)
        print("Best move's value was {}".format(moves[best_move].value))
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
        while current_state.hashable_repr() in self.nodes:
            #print("Making selection move number {}".format(count))
            # Get valid moves
            moves = self.nodes[current_state.hashable_repr()].edges
            # Choose one to make
            chosen_move = max(moves, key=lambda m : moves[m].value / (moves[m].num_visits + 1))
            # Add the chosen move's edge to the list of visited edges
            visited.append((current_state.hashable_repr(), chosen_move))
            # Apply the move (effectively a step down the tree)
            current_state.apply_move(chosen_move)
            #print(current_state.hashable_repr())
            count += 1
            if count > 10:

                #print("Making selection move number {}".format(count))
                #print("Move made: {}".format(chosen_move))
                if count > 20:
                    break

            #sleep(0.5)
        return current_state, visited

    def _get_selection_move(self, moves):
        return 0

    def expand(self, state):
        """
        Adds a new node to the search tree.
        This is called once the selection stage has finished.
        Delegating to _add_to_tree allows the option to only expand when certain
        conditions are met, e.g. only add a node to the tree once it has been
        visited n times.
        """
        self._add_to_tree(state)

    def simulate(self, state):
        """
        Plays out the game from a newly expanded node.
        """
        rollout_moves = []
        max_value = self.evaluate_state(state)
        for step in range(self.rollout_depth):
            if state.is_solved():
                # Need to handle solved stuff here...
                return max_value, rollout_moves
            next_move = self.rollout(state=state)
            state.apply_move(next_move)
            rollout_moves.append(str(next_move))
            max_value = max(max_value, self.evaluate_state(state))
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
    scramble = "R' B U B U B' D' F L B2"

    cube = rubiks.Cube(alg=scramble)

    movecount = 0
    move_history = []
    while movecount < 30:

        if cube.is_solved():
            print("Solved!")
            move_history.append(str(move))
            break

        print("Move {}".format(movecount + 1))

        mc = MonteCarlo(cube, state_value, polnet.evaluate, rollout_type=policy_rollout)
        move = mc.choose_move()

        print("Move chosen by MonteCarlo: {}".format(move))

        move_history.append(str(move))
        cube.apply_move(move)

        print("Depth estimated by valnet: {}".format(valnet.evaluate(cube)))
        if mc.solutions:
            print("Solutions found:")
            print(mc.solutions)
        movecount += 1

        print()

    print("Scramble:   {}".format(scramble))
    print("Moves made: {}".format(" ".join(move_history)))
