import numpy as np
import rubik.rubik as rubik
from features import get_features
import training
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

def main():

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"
    sample_input = training.get_features(rubik.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(len(sample_input),)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error",
                    optimizer="adam",
                    metrics=["accuracy"])


    # Training
    t = time.time()
    path = path_base + "mixed_length_scrambles.txt"
    (x_train, x_test), (y_train, y_test) = training.load_data(path, "value")
    print("Loading data took {}s".format(format((time.time() - t), ".3f")))

    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=1)

    saved_models_dir = "/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/"
    model.save(saved_models_dir + "value_network_90k_one_layer_test.h5")

    print()
    for name, value in zip(model.metrics_names, score):
        print("{}:\t{}".format(name, value))
    print()





if __name__ == "__main__":

    # For testing
    #np.random.seed(17)

    #main()

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
        c = rubik.Cube(alg=rubik.Algorithm(scramble))
        num_moves = 0
        moves_so_far = []
        min_estimated_depth = 100
        while num_moves < 30:
            cube_copies = [deepcopy(c) for _ in range(len(mincube.MOVES))]
            for i, copy in enumerate(cube_copies):
                copy.apply_move(rubik.Move(mincube.MOVES[i]))
            cubes_to_eval = [get_features(x) for x in cube_copies]
            scores = model.predict(np.array(cubes_to_eval))
            move = rubik.Move(mincube.MOVES[np.argmin(scores)])
            if min(scores) < min_estimated_depth:
                min_estimated_depth = min(scores)
            c.apply_move(move)
            moves_so_far.append(move)
            num_moves += 1
            if c.solved():
                soln = " ".join([str(m) for m in moves_so_far])
                if soln != str(rubik.Algorithm(scramble).invert()):
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
    (x_train, x_test), (y_train, y_test) = training.load_data(path, "value", training_set_size=0, limit=20)


    results = model.predict(x_test)

    print("\tPrediction\tExpected\tDifference")
    for i, (prediction, expected) in enumerate(zip(results, y_test)):
        print("{}:\t{}\t{}\t\t{}".format(i+1, format(prediction[0], "0.6f"), expected, format(expected - prediction[0], "0.6f")))
    """
