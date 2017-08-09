import numpy as np
import rubik.rubik as rubik
from features import get_features
import training
import time

import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.metrics import top_k_categorical_accuracy

# Define custom metrics

def top_3(y_true, y_pred):
    """
    Classes a prediction as a success if the move tagged as correct is within
    the top 3 moves chosen by the net.
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def close_to_winner(y_true, y_pred):
    """
    Classes a prediction as a success if the move tagged as correct has a value
    within 10 per cent of the move chosen by the net as most likely.
    """
    prediction_confidence = K.max(y_pred)
    true_confidence = y_pred[K.argmax(y_true)]
    return K.mean(K.greater(prediction_confidence,
                            tf.scalar_mul(0.9, true_confidence)))

def main():

    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"

    sample_input = training.get_features(rubik.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(len(sample_input),)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(18, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy", top_k_categorical_accuracy])

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

        model.save("/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/SLModel_test_32_100_100.h5")


def one_hot_to_move(arr):
    return training.MOVE_ENCODING[np.argmax(arr)]

if __name__ == "__main__":

    pred = K.constant([[0.39, 0.4, 0.21], [0.5, 0.02, 0.48]])
    true = K.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    print("Pred is {}".format(tf.Session().run(pred)))
    print("True is {}".format(tf.Session().run(true)))
    prediction_confidence = K.max(pred)
    true_confidence = pred[K.argmax(true)]
    print("Pred conf is {}".format(tf.Session().run(prediction_confidence)))
    print("True conf is {}".format(tf.Session().run(true_confidence)))

    out = close_to_winner(true, pred)

    res = tf.Session().run(out)

    print(res)

    quit()


    np.random.seed(17)

    #main()

    model = load_model("/Users/Zak/Desktop/MScCS/Project/AlphaCube/saved_models/SLModel_test_32_100_100.h5")
    datapath = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/mixed_length_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, "policy", training_set_size=0, limit=10000)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\nEvaluation on mixed length scrambles:")
    for name, value in zip(model.metrics_names, score):
        print("{}: {}".format(name, value))
    print()

    print("Sample predictions:\n")
    results = model.predict(x_test)
    moves = training.MOVE_ENCODING
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
        if i == 20: break
