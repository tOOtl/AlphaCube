import numpy as np
import mincube as rubiks
from features import get_features
from keras.utils import np_utils


def load_data(path, target_type, training_set_size=0.9, limit=-1):

    assert target_type in {"value", "policy"}, "Invalid type for training data: {}".format(target_type)

    with open(path, "r") as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        if limit > 0:
            lines = lines[:limit]
            num_items = limit
        else:
            num_items = len(lines)

    print("Number of items in data set is {}".format(num_items))

    num_training_items = int(num_items * training_set_size)
    print("Creating data...".ljust(25), end="\r")
    x_train = np.array([get_features(rubiks.Cube(alg=rubiks.Algorithm(line)))
                            for line in lines[:num_training_items]],
                            dtype=np.bool)
    print("Done creating x_train.".ljust(25), end="\r")
    x_test = np.array([get_features(rubiks.Cube(alg=rubiks.Algorithm(line)))
                            for line in lines[num_training_items:]],
                            dtype=np.bool)
    print("Done creating x_test.".ljust(25), end="\r")

    # Generate target output values indicating the next move that should
    # be performed based on the input scramble.
    if target_type == "policy":
        y_train = np.array([rubiks.MOVES.index(str(rubiks.Algorithm(line).moves[-1].inverse()))
                                for line in lines[:num_training_items]])
        y_train = np_utils.to_categorical(y_train, 18)
        print("Done creating y_train.".ljust(25), end="\r")
        y_test = np.array([rubiks.MOVES.index(str(rubiks.Algorithm(line).moves[-1].inverse()))
                                for line in lines[num_training_items:]])
        y_test = np_utils.to_categorical(y_test, 18)
        print("Done creating y_test.".ljust(25), end="\r")

    # Generate target output values according to the length of the
    # input scramble
    elif target_type == "value":
        y_train = np.array([len(line) // 3 for line in lines[:num_training_items]])
        print("Done creating y_train.".ljust(25), end="\r")
        y_test = np.array([len(line) // 3 for line in lines[num_training_items:]])
        print("Done creating y_test.".ljust(25), end="\r")

    print("Done loading data.".ljust(25))

    return (x_train, x_test), (y_train, y_test)




if __name__ == "__main__":
    load_data("/Users/Zak/Desktop/MScCS/Project/DataGenerator/one_move_scrambles.txt")
