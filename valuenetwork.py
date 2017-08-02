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
    model.save(saved_models_dir + "value_network_60k_all_linear_activations.h5")

    print()
    for name, value in zip(model.metrics_names, score):
        print("{}:\t{}".format(name, value))
    print()





if __name__ == "__main__":
    main()

    saved_models_dir = "/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/"
    model = load_model(saved_models_dir + "value_network_60k_all_linear_activations.h5")

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"
    path = path_base + "mixed_length_scrambles.txt"
    (x_train, x_test), (y_train, y_test) = training.load_data(path, "value", training_set_size=0, limit=20)

    results = model.predict(x_test)

    print("\tPrediction\tExpected\tDifference")
    for i, (prediction, expected) in enumerate(zip(results, y_test)):
        print("{}:\t{}\t{}\t\t{}".format(i+1, format(prediction[0], "0.6f"), expected, format(expected - prediction[0], "0.6f")))
