import numpy as np
import mincube as rubiks # TODO: CHANGE TO MINCUBE
from features import get_features
import training
import time

import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.metrics import top_k_categorical_accuracy

MODEL = None
PATH_BASE = "/Users/Zak/Desktop/MScCS/Project/"
MODEL_PATH = PATH_BASE + "AlphaCube/saved_models/policy/500k_test.h5"
DATA_PATH_BASE =  PATH_BASE + "DataGenerator/data/"

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

    sample_input = training.get_features(rubiks.Cube())

    # Model architecture
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(len(sample_input),)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(18, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy", top_3])

    for depth in range(1, 21):
        print("Starting depth {}".format(depth))
        t = time.time()
        path = DATA_PATH_BASE + str(depth) + "_move_scrambles.txt"
        (x_train, x_test), (y_train, y_test) = training.load_data(path, "policy")
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

if __name__ == "__main__":

    """
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
    """


    np.random.seed(17)

    #main()

    model = load_model(MODEL_PATH, custom_objects={"top_3":top_3})
    datapath = DATA_PATH_BASE + "mixed_length_scrambles.txt"
    (x, x_test), (y, y_test) = training.load_data(datapath, "policy", training_set_size=0, limit=10000)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\nEvaluation on mixed length scrambles:")
    for name, value in zip(model.metrics_names, score):
        print("{}: {}".format(name, value))
    print()

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
