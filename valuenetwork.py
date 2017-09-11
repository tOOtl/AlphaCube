import numpy as np
import mincube as rubiks
from features import get_features
import training
import time

from keras.models import Sequential, load_model
from keras.layers import Dense

MODEL = None
PATH_BASE = "/Users/Zak/Desktop/MScCS/Project/"
MODEL_PATH = PATH_BASE + "AlphaCube/saved_models/value/final.h5"
DATA_PATH_BASE =  PATH_BASE + "DataGenerator/data/"

def train(save_path, data_paths):

    sample_input = training.get_features(rubiks.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(100, activation="relu", input_dim=len(sample_input)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error",
                    optimizer="adam",
                    metrics=["accuracy"])

    grouped_x_train = np.empty((0, len(sample_input)))
    grouped_y_train = np.empty((0))
    grouped_x_test = np.empty((0, len(sample_input)))
    grouped_y_test = np.empty((0))

    for path in data_paths:
        # Load training data
        print("Loading dataset {}".format(path[path.rfind("/")+1:]))
        t = time.time()
        (x_train, x_test), (y_train, y_test) = training.load_data(path, "value", limit=-1)
        print("Loading data took {}s".format(format((time.time() - t), ".3f")))

        grouped_x_train = np.concatenate((grouped_x_train, x_train))
        grouped_y_train = np.concatenate((grouped_y_train, y_train))
        grouped_x_test = np.concatenate((grouped_x_test, x_test))
        grouped_y_test = np.concatenate((grouped_y_test, y_test))

    # Train model
    model.fit(grouped_x_train, grouped_y_train,
                batch_size=32, epochs=10, verbose=1)
    model.save(save_path)
    # Test model
    score = model.evaluate(grouped_x_test, grouped_y_test, verbose=1)
    print()
    for name, value in zip(model.metrics_names, score):
        print("{}:\t{}".format(name, value))
    print()

    global MODEL
    MODEL = model


def evaluate(cube):
    global MODEL
    # Load model if it hasn't been loaded already
    if MODEL == None:
        MODEL = load_model(MODEL_PATH)
    prediction = MODEL.predict(np.array([get_features(cube)]))
    # The actual value is nested in arrays because model.predict() is set up
    # to take multiple inputs at once (first level of nesting), and produce
    # outputs for more than one neuron (second level of nesting).
    # The reciprocal is taken to scale the value.
    value = 1 / prediction[0][0]
    return value




if __name__ == "__main__":

    paths = [DATA_PATH_BASE + str(depth) + "_move_scrambles.txt"
                for depth in range(1, 21)]
    paths.extend([
        DATA_PATH_BASE + "mixed_length_scrambles.txt",
        DATA_PATH_BASE + "random_move_scrambles_less_than_15.txt"
        ])

    train(MODEL_PATH, paths)
