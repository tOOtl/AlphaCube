import numpy as np
import rubik.rubik as rubik
from features import get_features
import training
import time

from keras.models import Sequential, load_model
from keras.layers import Dense

def main():

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"

    sample_input = training.get_features(rubik.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(len(sample_input),)))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(18, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    for depth in range(1, 21):
        print("Starting depth {}".format(depth))
        t = time.time()
        path = path_base + str(depth) + "_move_scrambles.txt"
        (x_train, x_test), (y_train, y_test) = training.load_data(path, "policy")
        print("Loading data took {}s".format(format((time.time() - t), ".3f")))

        model.fit(x_train, y_train,
                batch_size=32, epochs=5, verbose=1)

        score = model.evaluate(x_test, y_test, verbose=1)

        print()
        for name, value in zip(model.metrics_names, score):
            print("{}: {}".format(name, value))
        print()

        model.save("/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/SLModel_curriculum_20k_1.h5")



if __name__ == "__main__":

    model = load_model("/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/SLModel_curriculum_20k.h5")
    datapath = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/1_move_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, training_set_size=0)
    score = model.evaluate(x_test, y_test, verbose=1)
    print()
    for name, value in zip(model.metrics_names, score):
        print("{}: {}".format(name, value))
    print()
