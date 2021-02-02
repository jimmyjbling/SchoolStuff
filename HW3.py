import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot
import seaborn as sns


def make_discrete(train_df):
    chunk_df = train_df[["Age", "Fare"]]
    prev_quant = chunk_df.quantile(q=0.0, axis=0)
    quants = []
    for i in np.arange(0.2, 1.1, 0.2):
        quant = chunk_df.quantile(q=i, axis=0)
        replace_df = np.logical_and(prev_quant < train_df, train_df <= quant)
        train_df.mask(replace_df, quant, axis=1, inplace=True)
        prev_quant = quant
        quants.append(quant)
    return quants


def make_discrete_test(test_data, quants):
    chunk_df = test_data[["Age", "Fare"]]
    prev_quant = chunk_df.quantile(q=0.0, axis=0)
    for quant in quants:
        replace_df = np.logical_and(prev_quant < test_data, test_data <= quant)
        test_data.mask(replace_df, quant, axis=1, inplace=True)


def split_dataset(df):
    true_df = df[df['survived'] == 1]
    false_df = df[df['survived'] == 0]
    return true_df, false_df


def get_conditionals(df, true_df, false_df):
    conditionals = {}
    for col in df.columns:
        classes = list(set(df[col]))
        zero_conds = {}
        one_conds = {}
        for c in classes:
            # +1 and +len(class) are the laplace smoothing
            zero_conds[c] = (len(false_df[false_df[col] == c]) + 1) / (len(false_df[col]) + len(classes))
            one_conds[c] = (len(true_df[true_df[col] == c]) + 1) / (len(true_df[col]) + len(classes))
        conditionals[col] = [zero_conds, one_conds]
    return conditionals


def predict(prior, conditionals, data):
    preds = []
    for row in data.iterrows():
        neg_prob = prior[0]
        pos_prob = prior[1]
        for col in list(row[1].index):
            neg_prob = neg_prob * conditionals[col][0][row[1][col]]
            pos_prob = pos_prob * conditionals[col][1][row[1][col]]
        preds.append([neg_prob / (pos_prob + neg_prob), pos_prob / (pos_prob + neg_prob), (pos_prob/neg_prob) > 1])
    return preds


def accuracy(Y, preds):
    total_loss = 0
    for y, pred in zip(Y.values, preds):
        if y == int(pred[2]):
            total_loss = total_loss + 1
    return total_loss / len(preds)


def zero_one_loss(Y, preds):
    total_loss = 0
    for y, pred in zip(Y.values, preds):
        if y != int(pred[2]):
            total_loss = total_loss + 1
    return total_loss / len(preds)


def squared_loss(Y, preds):
    total_loss = 0
    for y, pred in zip(Y.values, preds):
        total_loss = total_loss + (1 - pred[int(y)])**2
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

    train_df = pd.concat([Y, X], axis=1)

    # fill them na up with the mode
    for col_name in train_df.columns:
        mode = float(train_df[col_name].mode()[0])
        train_df[col_name].fillna(mode, inplace=True)

    # make this bad boy discrete
    quants = make_discrete(train_df)

    # split into pos and neg class groups
    true_df, false_df = split_dataset(train_df)

    # get that juciy prior probaility
    prior = [len(false_df)/len(train_df), len(true_df)/len(train_df)]

    # get that guchie conditional probabilitys
    conditionals = get_conditionals(train_df, true_df, false_df)

    # read in testing data
    X = pd.read_csv(DATA_TEST_FILENAME, index_col=None)
    Y = pd.read_csv(LABEL_TEST_FILENAME, index_col=None)

    # make the data discrete like with the training
    make_discrete_test(X, quants)

    # make predictions
    preds = predict(prior, conditionals, X)

    # get accuracy
    acc = accuracy(Y, preds)

    # get 0-1 loss
    loss = zero_one_loss(Y, preds)

    # get squared loss
    sqloss = squared_loss(Y, preds)

    print("ZERO-ONE LOSS={:.4f}".format(loss))
    print("SQUARED LOSS={:.4f}".format(sqloss))
    print("Test Accuracy={:.4f}".format(acc))
