import numpy as np
import training

from keras.models import Sequential
from keras.layers import Dense

def main():

    path = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/one_move_scrambles.txt"
    (x_train, x_test), (y_train, y_test) = training.load_data(path)

    # Model architecture
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(len(x_train[0]),)))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(18, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])

    model.fit(x_train, y_train,
                batch_size=32, epochs=5, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=1)

    print(score)

if __name__ == "__main__":
    main()
