import numpy as np
import mincube as rubiks
from features import get_features
import training
import time

from keras.models import Sequential, load_model
from keras.layers import Dense

MODEL = None
PATH_BASE = "/Users/Zak/Desktop/MScCS/Project/"
MODEL_PATH = PATH_BASE + "AlphaCube/saved_models/value/160k_2phase_items.h5"
DATA_PATH_BASE =  PATH_BASE + "DataGenerator/data/"

def main():

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"
    sample_input = training.get_features(rubiks.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation="relu", input_dim=len(sample_input)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error",
                    optimizer="adam",
                    metrics=["accuracy"])

    # Load training data
    t = time.time()
    path = DATA_PATH_BASE + "random_move_scrambles.txt"
    (x_train, x_test), (y_train, y_test) = training.load_data(path, "value")
    print("Loading data took {}s".format(format((time.time() - t), ".3f")))

    # Train model
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    model.save(MODEL_PATH)

    # Test model
    score = model.evaluate(x_test, y_test, verbose=1)
    print()
    for name, value in zip(model.metrics_names, score):
        print("{}:\t{}".format(name, value))
    print()


def evaluate(cube):
    global MODEL
    # Load model if it hasn't been loaded already
    if MODEL == None:
        MODEL = load_model(MODEL_PATH)
    prediction = MODEL.predict(np.array([get_features(cube)]))
    # The actual value is nested in arrays because model.predict() is set up
    # to take multiple inputs at once (first level of nesting), and produce
    # outputs for more than one neuron (second level of nesting)
    return prediction[0][0]





if __name__ == "__main__":


    #main()



    quit()
    # For testing
    #np.random.seed(17)

    #main()

    """
    saved_models_dir = "/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/"
    model = load_model(saved_models_dir + "value_network_90k_one_layer_test.h5")

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"
    path = path_base + "mixed_length_scrambles.txt"

    import mincube
    from copy import deepcopy

    with open(path, "r") as f:
        scrambles = f.readlines()

    i = 0
    while True:
        #print("Test {}".format(i+1))
        #i += 1

        scramble = np.random.choice(scrambles)

        t = time.time()
        c = rubiks.Cube(alg=rubiks.Algorithm(scramble))
        num_moves = 0
        moves_so_far = []
        min_estimated_depth = 100
        while num_moves < 30:
            cube_copies = [deepcopy(c) for _ in range(len(mincube.MOVES))]
            for i, copy in enumerate(cube_copies):
                copy.apply_move(rubiks.Move(mincube.MOVES[i]))
            cubes_to_eval = [get_features(x) for x in cube_copies]
            scores = model.predict(np.array(cubes_to_eval))
            move = rubiks.Move(mincube.MOVES[np.argmin(scores)])
            if min(scores) < min_estimated_depth:
                min_estimated_depth = min(scores)
            c.apply_move(move)
            moves_so_far.append(move)
            num_moves += 1
            if c.solved():
                soln = " ".join([str(m) for m in moves_so_far])
                if soln != str(rubiks.Algorithm(scramble).invert()):
                    print("Scramble: {} ({})".format(scramble.strip(), len(scramble) // 3))
                    print("Solution: {} ({})".format(soln, len(moves_so_far)))
                    break
        else:
            pass
            #print("Min estimated depth found: {}".format(min_estimated_depth))
            #print("Current estimated depth: {}".format(model.predict(np.array([get_features(c)]))))
        #print("Time taken for trial: {}s".format(format(time.time() - t, ".2f")))
        #print()
        """








    """
    (x_train, x_test), (y_train, y_test) = training.load_data(path, "value", training_set_size=0, limit=20)


    results = model.predict(x_test)

    print("\tPrediction\tExpected\tDifference")
    for i, (prediction, expected) in enumerate(zip(results, y_test)):
        print("{}:\t{}\t{}\t\t{}".format(i+1, format(prediction[0], "0.6f"), expected, format(expected - prediction[0], "0.6f")))
    """
