import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot
import seaborn as sns

# Set learning rate
LEARNING_RATE = 0.000007


def predict(X, Y, w):
    preds = []
    for x, y in zip(X, Y):
        preds.append(int(np.where(w[0] + np.dot(x, w[1:]) >= 0, 1, 0)))
    return preds


def h_loss(X, Y, w):
    loss = 0
    for x, y in zip(X, Y):
        t = -1 if y == 0 else 1
        percep = (np.dot(x, w[1:]) + w[0])
        loss = loss + max([0, 1 - (t * percep)])
    return loss/len(X)


def train(X, Y, w, la):
    i = 0
    while i < 800:  # epoch
        total_error = 0
        for x, y in zip(X, Y):
            new_y = 1 if w[0] + np.dot(x, w[1:]) > 0 else 0
            w[0] = w[0] + float((la * (y - new_y)))
            w[1:] = w[1:] + ((la * (y - new_y)) * x)
            total_error = total_error + int((y - new_y) != 0)
        # print("Round {}: Errors = {}".format(i, total_error))

        if total_error == 0:
            break
        else:
            i = i + 1
    return w


def accuracy(Y, preds):
    total_loss = 0
    for y, pred in zip(Y, preds):
        if y == pred:
            total_loss = total_loss + 1
    return total_loss / len(preds)


if __name__ == "__main__":
    # parse arguments

    DATA_TRAIN_FILENAME = sys.argv[1]
    LABEL_TRAIN_FILENAME = sys.argv[2]
    DATA_TEST_FILENAME = sys.argv[3]
    LABEL_TEST_FILENAME = sys.argv[4]

    # read in training data
    X = pd.read_csv(DATA_TRAIN_FILENAME, index_col=None)
    Y = pd.read_csv(LABEL_TRAIN_FILENAME, index_col=None)

    # fill them na up with the mode
    for col_name in X.columns:
        mode = float(X[col_name].mode()[0])
        X[col_name].fillna(mode, inplace=True)

    # initialize w to all 0
    w = [0 for x in range(len(X.columns) + 1)]
    w[0] = 1

    # convert to numpy array cause its faster than pandas
    X = np.array(X)
    Y = np.array(Y)

    # train the bad boy
    w = train(X, Y, w, LEARNING_RATE)

    # read in testing data
    X = pd.read_csv(DATA_TEST_FILENAME, index_col=None)
    Y = pd.read_csv(LABEL_TEST_FILENAME, index_col=None)

    for col_name in X.columns:
        mode = float(X[col_name].mode()[0])
        X[col_name].fillna(mode, inplace=True)

    # convert to numpy array cause its faster
    X = np.array(X)
    Y = np.array(Y)

    # make predictions and get hinge loss
    preds = predict(X, Y, w)

    # get hinge loss
    loss = h_loss(X, Y, w)

    # get accuracy
    acc = accuracy(Y, preds)

    print("HINGE LOSS={:.4f}".format(loss))
    print("Test Accuracy={:.4f}".format(acc))

