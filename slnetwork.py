import numpy as np
import mincube as rubiks
from features import get_features
import training
import time

import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.metrics import top_k_categorical_accuracy

MODEL = None
PATH_BASE = "/Users/Zak/Desktop/MScCS/Project/"
MODEL_PATH = PATH_BASE + "AlphaCube/saved_models/policy/rl_initialiser_aggregate_with_rand_move_scrambles.h5"
DATA_PATH_BASE =  PATH_BASE + "DataGenerator/data/"

# Define custom metrics

def top_3(y_true, y_pred):
    """
    Classes a prediction as a success if the move tagged as correct is within
    the top 3 moves chosen by the net.
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def train(datapaths, aggregate_data=True):

    sample_input = get_features(rubiks.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(1,len(sample_input))))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Flatten())
    model.add(Dense(18, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy", top_3])

    if aggregate_data:
        # Collect all data into one set and train it together
        grouped_x_train = np.empty((0, 1, len(sample_input)))
        grouped_y_train = np.empty((0, 18))
        grouped_x_test = np.empty((0, 1, len(sample_input)))
        grouped_y_test = np.empty((0, 18))
        for path in paths:
            print("Loading dataset {}".format(path[path.rfind("/")+1:]))
            t = time.time()
            (x_train, x_test), (y_train, y_test) = training.load_data(path, "policy")
            # Reshape for new RL network architecture
            x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
            grouped_x_train = np.concatenate((grouped_x_train, x_train))
            grouped_x_test = np.concatenate((grouped_x_test, x_test))
            grouped_y_train = np.concatenate((grouped_y_train, y_train))
            grouped_y_test = np.concatenate((grouped_y_test, y_test))
            print("- took {}s".format(format((time.time() - t), ".3f")))
        # Shuffle while maintaining the pairings of training and test items
        shuffle_in_unison(grouped_x_train, grouped_y_train)
        shuffle_in_unison(grouped_x_test, grouped_y_test)

        model.fit(grouped_x_train, grouped_y_train,
                batch_size=32, epochs=5, verbose=1)

        score = model.evaluate(grouped_x_test, grouped_y_test, verbose=1)

        print()
        for name, value in zip(model.metrics_names, score):
            print("{}: {}".format(name, value))
        print()

        model.save(MODEL_PATH)
    else:
        # Train on each data file separately, in order
        for path in datapaths:
            print("Starting dataset {}".format(path))
            t = time.time()
            (x_train, x_test), (y_train, y_test) = training.load_data(path, "policy")
            # Reshape for new RL network architecture
            x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
            print("Loading data took {}s".format(format((time.time() - t), ".3f")))

            model.fit(x_train, y_train,
                    batch_size=32, epochs=5, verbose=1)

            score = model.evaluate(x_test, y_test, verbose=1)

            print()
            for name, value in zip(model.metrics_names, score):
                print("{}: {}".format(name, value))
            print()

            model.save(MODEL_PATH)

def evaluate(state):
    global MODEL
    if MODEL == None:
        MODEL = load_model(MODEL_PATH, custom_objects={"top_3":top_3})
    prediction = MODEL.predict(np.array([get_features(state)]))
    return prediction[0]

def one_hot_to_move(arr):
    return rubiks.MOVES[np.argmax(arr)]

def shuffle_in_unison(a, b):
    # ref: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

if __name__ == "__main__":

    
    paths = [DATA_PATH_BASE
            + str(depth)
            + "_move_scrambles.txt" for depth in range(1, 21)]
    paths.append(DATA_PATH_BASE + "random_move_scrambles_less_than_15.txt")
    train(paths, aggregate_data=True)

    model = load_model(MODEL_PATH, custom_objects={"top_3":top_3})
    datapath = DATA_PATH_BASE + "mixed_length_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, "policy", training_set_size=0, limit=10000)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\nEvaluation on mixed length scrambles:")
    for name, value in zip(model.metrics_names, score):
        print("{}: {}".format(name, value))
    print()

    quit()

    print("Sample predictions:\n")
    results = model.predict(x_test)
    moves = rubiks.MOVES
    print("\tPrediction\tConfidence\tExpected\tConfidence\tDifference")
    for i, (prediction, expected) in enumerate(zip(results, y_test)):
        prediction_confidence = max(prediction)
        expected_confidence = prediction[np.argmax(expected)]
        diff = prediction_confidence - expected_confidence
        print("{}:\t{}\t\t{}\t{}\t\t{}\t{}".format(
                                    i+1,
                                    one_hot_to_move(prediction),
                                    format(prediction_confidence, "0.6f"),
                                    one_hot_to_move(expected),
                                    format(expected_confidence, "0.6f"),
                                    format(diff, "0.6f")
                                    ))
        if i == 10: break
