import numpy as np
import rubik.rubik as rubik
from features import get_features
from keras.utils import np_utils

MOVE_ENCODING = [face + magnitude
                    for face in rubik.FACE_MOVES
                    for magnitude in ("", "2", "'")]

def load_data(path):

    num_items = 0
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line)
            num_items += 1

    num_training_items = int(num_items * 0.9)
    #print("Creating data")
    x_train = np.array([get_features(rubik.Cube(alg=rubik.Algorithm(line)))
                            for line in lines[:num_training_items]],
                            dtype=np.bool)
    #print("Done creating x_train")
    x_test = np.array([get_features(rubik.Cube(alg=rubik.Algorithm(line)))
                            for line in lines[num_training_items:]],
                            dtype=np.bool)
    #print("Done creating x_test")

    y_train = np.array([MOVE_ENCODING.index(rubik.Algorithm(line).moves[-1].invert())
                            for line in lines[:num_training_items]])
    y_train = np_utils.to_categorical(y_train, 18)
    #print("Done creating y_train")
    y_test = np.array([MOVE_ENCODING.index(rubik.Algorithm(line).moves[-1].invert())
                            for line in lines[num_training_items:]])
    y_test = np_utils.to_categorical(y_test, 18)
    #print("Done creating y_test")

    return (x_train, x_test), (y_train, y_test)

if __name__ == "__main__":
    load_data("/Users/Zak/Desktop/MScCS/Project/DataGenerator/one_move_scrambles.txt")
