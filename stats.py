import numpy as np
import time
import mincube as rubiks

from slnetwork import top_3
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.metrics import top_k_categorical_accuracy as top_5
from training import load_data


def show_move_distributions():

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"

    counts = dict()
    from training import MOVE_ENCODING

    for scramble_len in range(1, 21):
        t = time.time()
        path = path_base + str(scramble_len) + "_move_scrambles.txt"

        counts[scramble_len] = [0 for _ in range(len(MOVE_ENCODING))]
        with open(path, "r") as f:
            for scramble in f:
                try:
                    i = 0
                    depth = 1
                    while i < len(scramble) - 1:
                        move = scramble[i:i+3]
                        index = MOVE_ENCODING.index(move.strip())
                        counts[depth][index] += 1
                        i += 3
                        depth += 1
                except ValueError:
                    print(scramble)
                    print("Error occurred at i = {}".format(i))
                    quit()
        print("Finished length {} in {}s".format(scramble_len, format(time.time() - t, ".2f")))

    with open(path_base + "mixed_length_scrambles.txt", "r") as f:
        t = time.time()
        counts[21] = [0 for _ in range(len(MOVE_ENCODING))]
        for scramble in f:
            i = 0
            depth = 1
            while i < len(scramble) - 1:
                move = scramble[i:i+2]
                index = MOVE_ENCODING.index(move.strip())
                counts[21][index] += 1
                i += 3
                depth += 1
        print("Finished mixed scrambles in {}s".format(format(time.time() - t, ".2f")))
        print(counts[21])


    print("Depth ", end="")
    for move in MOVE_ENCODING:
        print(str(move).ljust(6), end="")
    print()
    for depth in counts:
        print(str(depth).ljust(6), end="")
        total = sum(counts[depth])
        for move in counts[depth]:
            print(format((move / total), ".3f"), end=" ")
        print()

    all_scrambles = [0 for _ in range(len(MOVE_ENCODING))]
    for c in counts:
        for i in range(len(all_scrambles)):
            all_scrambles[i] += counts[c][i]
    print("All   ", end="")
    total = sum(all_scrambles)
    for move in all_scrambles:
        print(format((move / total), ".3f"), end=" ")
    print()

def eval_slnetwork_top_k():

    model = load_model("/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/SLModel_test_32_100_100.h5")

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy", top_3, top_5])

    path = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/mixed_length_scrambles.txt"
    t = time.time()
    (x_train, x_test), (y_train, y_test) = load_data(path, "policy", training_set_size=0)
    print("Time taken to load data: {}".format(time.time() - t))

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\nEvaluation on mixed length scrambles:")
    for name, value in zip(model.metrics_names, score):
        print("{}: {}".format(name, value))
    print()

def compare_policy_network_architectures(architectures):

    import slnetwork as polnet
    import mincube as rubiks
    import training
    from features import get_features
    np.random.seed(17)

    # Build list of paths to training data
    paths = [polnet.DATA_PATH_BASE
            + str(depth)
            + "_move_scrambles.txt" for depth in range(1, 21)]
    paths.append(polnet.DATA_PATH_BASE + "random_move_scrambles_less_than_15.txt")

    sample_input = get_features(rubiks.Cube())

    # Load training data

    # Collect all data into one set and train it together
    grouped_x_train = np.empty((0, 1, len(sample_input)))
    grouped_y_train = np.empty((0, 18))
    for path in paths:
        print("Loading dataset {}".format(path[path.rfind("/")+1:]))
        t = time.time()
        (x_train, x_test_empty), (y_train, y_test_empty) = training.load_data(path, "policy", training_set_size=1)
        # Reshape for new RL network architecture
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        grouped_x_train = np.concatenate((grouped_x_train, x_train))
        grouped_y_train = np.concatenate((grouped_y_train, y_train))
        print("- took {}s".format(format((time.time() - t), ".3f")))
    # Shuffle while maintaining the pairings of training and test items
    polnet.shuffle_in_unison(grouped_x_train, grouped_y_train)

    # Load test data
    datapath = polnet.DATA_PATH_BASE + "mixed_length_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, "policy", training_set_size=0, limit=50000)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # Loop through architectures
    for arc in architectures:
        # Reset the random seed for each one
        np.random.seed(123)

        # Build the model
        model = Sequential()
        model.add(Dense(arc[0], activation="relu", input_shape=(1,len(sample_input))))
        for layer in arc[1:]:
            model.add(Dense(layer, activation="relu"))
        model.add(Flatten())
        model.add(Dense(18, activation="softmax"))

        model.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=["accuracy", polnet.top_3])

        training_time_start = time.time()
        model.fit(grouped_x_train, grouped_y_train,
                    batch_size=32, epochs=5, verbose=1)
        training_time = time.time() - training_time_start

        score = model.evaluate(x_test, y_test, verbose=1)

        print()
        print("Architecture: {}".format(arc))
        print("Training time: {}".format(training_time))
        for name, value in zip(model.metrics_names, score):
            print("{}: {}".format(name, value))
        print()

def compare_value_network_architectures(architectures):

    import valuenetwork as valnet
    import slnetwork as polnet
    import mincube as rubiks
    import training
    from features import get_features
    np.random.seed(17)

    # Build list of paths to training data
    paths = [valnet.DATA_PATH_BASE
            + str(depth)
            + "_move_scrambles.txt" for depth in range(1, 21)]
    paths.append(valnet.DATA_PATH_BASE + "random_move_scrambles_less_than_15.txt")

    sample_input = get_features(rubiks.Cube())

    # Load training data

    # Collect all data into one set and train it together
    grouped_x_train = np.empty((0, 1, len(sample_input)))
    grouped_y_train = np.empty((0,))
    for path in paths:
        print("Loading dataset {}".format(path[path.rfind("/")+1:]))
        t = time.time()
        (x_train, x_test_empty), (y_train, y_test_empty) = training.load_data(path, "value", training_set_size=1)
        # Reshape for new RL network architecture
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        grouped_x_train = np.concatenate((grouped_x_train, x_train))
        grouped_y_train = np.concatenate((grouped_y_train, y_train))
        print("- took {}s".format(format((time.time() - t), ".3f")))
    # Shuffle while maintaining the pairings of training and test items
    polnet.shuffle_in_unison(grouped_x_train, grouped_y_train)

    # Load test data
    datapath = valnet.DATA_PATH_BASE + "mixed_length_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, "value", training_set_size=0, limit=50000)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    results = []
    # Loop through architectures
    for arc in architectures:
        # Reset the random seed for each one
        np.random.seed(123)

        # Build the model
        model = Sequential()
        model.add(Dense(arc[0], activation="relu", input_shape=(1,len(sample_input))))
        for layer in arc[1:]:
            model.add(Dense(layer, activation="relu"))
        model.add(Flatten())
        model.add(Dense(1, activation="linear"))

        model.compile(loss="mse",
                        optimizer="adam",
                        metrics=["accuracy"])

        training_time_start = time.time()
        model.fit(grouped_x_train, grouped_y_train,
                    batch_size=32, epochs=5, verbose=1)
        training_time = time.time() - training_time_start

        score = model.evaluate(x_test, y_test, verbose=1)

        results.append((str(arc),
                        format(training_time, "1.0f"),
                        format(score[0], "1.3f"),
                        format(score[1], "1.3f"),
                        ))

        print()
        print("Architecture: {}".format(arc))
        print("Training time: {}".format(training_time))
        for name, value in zip(model.metrics_names, score):
            print("{}: {}".format(name, value))
        print()

    return results

def valnet_performance(nb_trials, min_depth=1, max_depth=20):

    import valuenetwork as valnet
    import mincube as rubiks

    def random_eval(depth):
        c = rubiks.Cube()
        c.scramble(depth)
        return valnet.evaluate(c)

    results = dict()

    for depth in range(min_depth, max_depth + 1):
        results[str(depth)] = sum(
                [1 / random_eval(depth) for _ in range(nb_trials)]
            ) / nb_trials

    return results

def network_only_solver(nb_trials, depth, choose_move, use_history=False):
    """choose_move should be a function that return a move object. The easiest
    way to do this is to pass the rollout method of one of the Rollout objects
    in rollouts.py as choose_move. This can be used to test the policy network
    by using the policy rollout, or the value network by using the value
    rollout.
    """
    trials = []
    for _ in range(nb_trials):
        solution = []
        solved = False
        scramble = rubiks.get_scramble(depth)
        c = rubiks.Cube(alg=scramble)
        history = [c.hashable_repr()] if use_history else []
        for step in range(depth * 2):
            move = choose_move(c, history=history)
            solution.append(str(move))
            c.apply_move(move)
            if use_history: history.append(c.hashable_repr())
            if c.is_solved():
                solved = True
                break
        trials.append((scramble, " ".join(solution), solved))
    return trials

def network_only_solver_printout(nb_trials, min_depth, max_depth, rollout, use_history):
    res = dict()
    for depth in range(min_depth, max_depth+1):
        print("\n~~~ Depth {} ~~~".format(depth))
        results = network_only_solver(nb_trials, depth, rollout, use_history)
        print("Number of successes: {}".format(sum([r[2] for r in results])))
        print("Interesting solutions:")
        print("Scramble".ljust(depth*3), end=" Solution\n")
        successes = sorted(filter(lambda r : r[2], results), key=lambda r : r[1].count(" "))
        interesting_count = 0
        for r in successes[:10]:
            print("{}".format(r[0]).ljust(depth*3), end=" {} ({})\n".format(r[1], r[1].count(" ")+1))
        res[str(depth)] = successes
    return res

def stepped_tree_search_solver(nb_trials, min_depth, max_depth, rollout, show_solutions=False):

    import valuenetwork as valnet
    import slnetwork as polnet
    import montecarlo

    r = rollout
    nb_trials = nb_trials

    for scramble_len in range(min_depth, max_depth+1):
        nb_successes = 0
        solutions_to_print = []
        for trial in range(nb_trials):
            print("Running trial {}".format(trial + 1), end="\r")
            scramble = rubiks.get_scramble(scramble_len)
            cube = rubiks.Cube(alg=scramble)
            moves_made = []
            while len(moves_made) < (scramble_len * 2):
                mc = montecarlo.MonteCarlo(cube, valnet.evaluate, polnet.evaluate, rollout_type=r)
                move = mc.choose_move(search_time=1)
                cube.apply_move(move)
                moves_made.append(move)
                if cube.is_solved():
                    nb_successes += 1
                    if show_solutions:
                        solutions_to_print.append((scramble, " ".join([str(m) for m in moves_made])))
                    break
        print("Depth {}:".format(scramble_len).ljust(10), end="")
        print("{}%".format((nb_successes / nb_trials) * 100).ljust(10))
        if show_solutions:
            print("Scramble".ljust(scramble_len*3), end=" Solution\n")
            solutions_to_print.sort(key=lambda s : len(s[1]))
            for s in solutions_to_print[:10]:
                print("{}".format(s[0]).ljust(scramble_len*3),
                        end=" {} ({})\n".format(s[1], s[1].count(" ")+1))

def single_tree_search_solver(nb_trials, min_depth, max_depth, rollout, show_solutions=False):

    import valuenetwork as valnet
    import slnetwork as polnet
    import montecarlo

    r = rollout
    nb_trials = nb_trials

    for scramble_len in range(min_depth, max_depth+1):
        nb_successes = 0
        solutions_to_print = []
        for trial in range(nb_trials):
            print("Running trial {}".format(trial + 1), end="\r")
            scramble = rubiks.get_scramble(scramble_len)
            cube = rubiks.Cube(alg=scramble)
            mc = montecarlo.MonteCarlo(cube, valnet.evaluate, polnet.evaluate, rollout_type=r)
            solutions = mc.single_search(search_time=min((scramble_len*2, 10)))
            if solutions:
                nb_successes += 1
                if show_solutions:
                    solutions_to_print.append((scramble, min(solutions)))
        print("Depth {}:".format(scramble_len).ljust(10), end="")
        print("{}%".format((nb_successes / nb_trials) * 100).ljust(10))
        if show_solutions:
            print("Scramble".ljust(scramble_len*3), end=" Solution\n")
            solutions_to_print.sort(key=lambda s : len(s[1]))
            for s in solutions_to_print[:10]:
                print("{}".format(s[0]).ljust(scramble_len*3),
                        end=" {} ({})\n".format(s[1], s[1].count(" ")+1))


def polnet_performance_by_depth(nb_items_per_depth):
    import slnetwork as polnet
    model = load_model(polnet.MODEL_PATH, custom_objects={"top_3":top_3})
    results = []
    for depth in range(1, 21):
        print("Depth {}".format(depth))
        path = polnet.DATA_PATH_BASE + str(depth) + "_move_scrambles.txt"
        (x, _x), (y, _y) = load_data(path, "policy", training_set_size=1.0, limit=nb_items_per_depth)
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        score = model.evaluate(x, y)
        for name, value in zip(model.metrics_names, score):
            print("{}: {}".format(name, value))
        results.append(score)
    return results

def polnet_performance_by_depth_random_move(nb_items_per_depth):
    import slnetwork as polnet
    import features
    from keras.utils import np_utils
    model = load_model(polnet.MODEL_PATH, custom_objects={"top_3":top_3})
    results = []
    for depth in range(1, 21):
        print("Depth {}".format(depth))
        scrambles = [rubiks.get_scramble(scramble_len=depth) for _ in range(nb_items_per_depth)]
        x = np.array([features.get_features(rubiks.Cube(alg=s)) for s in scrambles], dtype=np.bool)
        y = np.array([rubiks.MOVES.index(str(rubiks.Move(s.split()[-1]).inverse())) for s in scrambles])
        y = np_utils.to_categorical(y, 18)
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        score = model.evaluate(x, y, verbose=0)
        results.append(score)
    return results


if __name__ == "__main__":
    import rollouts
    r = rollouts.ProbabilisticPolicyRollout()
    stepped_tree_search_solver(5, 5, 12, r, show_solutions=True)
