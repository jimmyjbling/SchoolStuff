##############
# Name:
# email:
# Date:

import numpy as np
import pandas as pd
import sys
import os


def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    """
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy


def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(A) - SUM ( |Di| / |D| * entropy(Di) )
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


def get_freq(data):
    counts = data.iloc[:, 0].value_counts()
    counts = sorted(list(zip(counts.index, counts.values)), key=lambda x: x[0])
    counts = [x[1] for x in counts]
    if not counts:
        counts = [0, 0]
    return counts


def get_classification(counts):
    if counts[0] > counts[1]:
        return 0
    else:
        return 1


def accuracy(pred, true_y):
    true_y = true_y.array
    num_correct = 0
    total_pred = len(pred)
    for x in range(len(pred)):
        if pred[x] == true_y[x]:
            num_correct = num_correct + 1
    return num_correct/total_pred


class Node(object):
    def __init__(self, l, r, attr, thresh, classification):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.classification = classification


class Tree(object):
    def __init__(self, root):
        self.root = root

    def predict(self, X):
        predicts = []
        for index, row in X.iterrows():
            current_node= self.root
            current_class = None
            while current_node is not None:
                current_class = current_node.classification
                if current_node.attribute is None:
                    break
                if row[current_node.attribute] <= current_node.threshold:  # NOTE the tree leans left on ties
                    current_node = current_node.left_subtree
                else:
                    current_node = current_node.right_subtree
            predicts.append(current_class)
        return predicts


def count_node(cur_node, running_count):
    if cur_node is None:
        return running_count
    left_sub_count = count_node(cur_node.left_subtree, running_count)
    right_sub_count = count_node(cur_node.right_subtree, running_count)
    total_sub_count = left_sub_count + right_sub_count
    return total_sub_count + 1


def ID3(train_data, max_depth, depth, min_split):
    # 1. use a for loop to calculate the infor-gain of every attribute
    best_att = None
    highest_gain = -100000000
    best_s1 = None
    best_s2 = None
    best_thresh = None

    # early exit case check
    if len(get_freq(train_data)) == 1:
        return Node(None, None, None, None, train_data.iloc[:, 0].value_counts().index[0])
    if len(train_data) < min_split:
        return Node(None, None, None, None, train_data.iloc[:, 0].value_counts().index[0])
    if depth >= max_depth:
        return Node(None, None, None, None, train_data.iloc[:, 0].value_counts().index[0])

    # Find gain of each attribute
    skip = 0
    for att_name, att in train_data.iteritems():
        if skip == 0:
            skip = 1
            continue
        thresh = None

        mode = float(att.mode()[0])

        att.fillna(mode, inplace=True)

        # Find Threshold
        min_entropy = 1000000000
        att_set = sorted(list(set(att)))
        mids = [(att_set[i] + att_set[i+1]) / 2 for i in range(0, len(att_set) - 1)]
        if not mids:
            continue
        tmp = 0
        for mid in mids:
            s1 = train_data[train_data[att_name] <= mid]
            s2 = train_data[train_data[att_name] > mid]
            cur_d = len(train_data)
            cur_entropy = ((len(s1)/cur_d) * entropy(get_freq(s1))) + ((len(s2)/cur_d) * entropy(get_freq(s2)))
            if cur_entropy < min_entropy:
                thresh = mid
                min_entropy = cur_entropy

        # Split data on threshold
        s1 = train_data[train_data[att_name] < thresh]
        s2 = train_data[train_data[att_name] > thresh]

        # Calculate info gain
        cur_gain = infor_gain(get_freq(train_data), [get_freq(s1), get_freq(s2)])
        if cur_gain > highest_gain:
            highest_gain = cur_gain
            best_att = att_name
            best_s1 = s1.copy()
            best_s2 = s2.copy()
            best_thresh = thresh
    current_node = Node(None, None, best_att, best_thresh, get_classification(get_freq(train_data)))

    # when this is true it means identical rows have different labels. Default to calling it class majority
    if best_s1 is None and best_s2 is None:
        return current_node

    left_subtree = ID3(best_s1, max_depth, depth+1, min_split)
    right_subtree = ID3(best_s2, max_depth, depth+1, min_split)
    current_node.left_subtree = left_subtree
    current_node.right_subtree = right_subtree

    return current_node


def please_prune_da_tree_sir(dt, dt_node, val_data):

    if dt_node is None:
        return dt_node

    dt_node.left_subtree = please_prune_da_tree_sir(dt, dt_node.left_subtree, val_data)
    dt_node.right_subtree = please_prune_da_tree_sir(dt, dt_node.right_subtree, val_data)

    left_pointer = dt_node.left_subtree
    right_pointer = dt_node.right_subtree

    current_acc = accuracy(dt.predict(val_data), val_data.iloc[:, 0])
    dt_node.left_subtree = None
    dt_node.right_subtree = None
    prune_acc = accuracy(dt.predict(val_data), val_data.iloc[:, 0])

    if prune_acc > current_acc:
        return dt_node
    else:
        dt_node.left_subtree = left_pointer
        dt_node.right_subtree = right_pointer
        return dt_node


class PCA(object):
    def __init__(self, n_component):
        self.n_component = n_component
        self.mean = None
        self.sd = None
        self.vec = None

    def fit_transform(self, train_data):

        # check to make sure that n_component is less than or equal to number of features in data
        # if its not I will throw an error, as the PCA is not possible with that many

        if len(train_data.columns) < self.n_component:
            raise ValueError("n_component exceeds num of features in data")

        # scale data
        data = train_data.iloc[:, 1:]
        means = data.mean()
        sds = data.std()
        trans_data = (data - means) / sds

        self.mean = means
        self.sd = sds

        # PCA part
        cov = np.cov(trans_data.T)

        vals, vecs = np.linalg.eig(cov)

        self.vec = vecs

        # transform the data
        pca_train = pd.DataFrame(train_data.iloc[:, 0])
        for i in range(1, self.n_component + 1):
            col_name = "PC" + str(i)
            pca_train[col_name] = trans_data.dot(vecs.T[i-1])
        return pca_train


    def transform(self, test_data):
        # scale data
        data = test_data.iloc[:, 1:]
        trans_data = (data - self.mean) / self.sd

        # transform the data
        pca_test = pd.DataFrame(test_data.iloc[:, 0])
        for i in range(1, self.n_component + 1):
            col_name = "PC" + str(i)
            pca_test[col_name] = trans_data.dot(self.vec.T[i - 1])
        return pca_test


if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--crossValidK', type=int, default=5)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--minSplit', type=int)
    parser.add_argument('--PCANcomp', type=int)
    args = parser.parse_args()

    # set up model parameters
    if args.model == "depth":
        depth = args.depth
        minSplit = 0
        post_prune = False
    elif args.model == "minSplit":
        depth = 10000000000
        minSplit = args.minSplit
        post_prune = False
    elif args.model == "postPrune":
        depth = 10000000000
        minSplit = 0
        post_prune = True
    else:
        depth = 10000000000
        minSplit = 0
        post_prune = False

    """
    if args.PCANcomp is not None:
        df = pd.read_csv(str(args.trainFolder) + "/bonus_data.csv")
        df.drop(df.columns[[0]], axis=1, inplace=True)
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]
    # load in train data
    else:
    """
    X = pd.read_csv(str(args.trainFolder) + "/train-file.data")
    Y = pd.read_csv(str(args.trainFolder) + "/train-file.label")

    # merge labels and data into one dataframe
    df = pd.concat([Y, X], axis=1)

    # fill na with mode
    for col_name in df.columns:
        mode = float(df[col_name].mode()[0])
        median = float(df[col_name].median())
        df[col_name].fillna(mode, inplace=True)

    if args.PCANcomp is not None:
        my_pca = PCA(args.PCANcomp)
        df = my_pca.fit_transform(df)

    # make k fold cross validation splits
    crossval_splits = [0]
    jump = int(np.floor(len(df) / args.crossValidK))
    for k in range(1, args.crossValidK):
        crossval_splits.append(int(k*jump))
    crossval_splits.append(len(df)-1)

    chunks = [df.iloc[crossval_splits[x]:crossval_splits[x+1], :] for x in range(len(crossval_splits) - 1)]

    # do cross validation
    for i in range(args.crossValidK):
        testing = chunks[i]
        training = pd.concat([chunks[x] for x in range(args.crossValidK) if x != i], axis=0)
        # pull out a pruning set if we are post pruning. About 20% of the training data
        if post_prune:
            post_prune_df = training.sample(int(len(training)/5), replace=False)
            training.drop(post_prune_df.index, inplace=True)
        val_dt = Tree(ID3(training, max_depth=depth, depth=0, min_split=minSplit))
        # post prune tree if called for
        if post_prune:
            val_dt = Tree(please_prune_da_tree_sir(val_dt, val_dt.root, post_prune_df))
        # training acc:
        pred = val_dt.predict(training)
        train_acc = accuracy(pred, training.iloc[:, 0])
        # testing acc
        pred = val_dt.predict(testing)
        test_acc = accuracy(pred, testing.iloc[:, 0])
        output = "fold={}, train set accuracy={:.4f}, validation set accuracy={:.4f}, num nodes={}"
        print(output.format(i+1, train_acc, test_acc, count_node(val_dt.root, 0)))

    if post_prune:
        post_prune_df = df.sample(int(len(df) / 5), replace=False)
        df.drop(post_prune_df.index, inplace=True)

    # build the tree from the training data
    dt = Tree(ID3(df, max_depth=depth, depth=0, min_split=minSplit))

    if post_prune:
        dt = Tree(please_prune_da_tree_sir(dt, dt.root, post_prune_df))

    # load in the testing data
    X_test = pd.read_csv(str(args.testFolder) + "/test-file.data")
    Y_test = pd.read_csv(str(args.testFolder) + "/test-file.label")

    df = pd.concat([Y_test, X_test], axis=1)

    # for analysis only
    # df = df.dropna()

    for col_name in X_test.columns:
        mode = float(X_test[col_name].mode()[0])
        median = float(df[col_name].median())
        X_test[col_name].fillna(median, inplace=True)

    if args.PCANcomp is not None:
        df = my_pca.transform(df)

    # predict on testing set & evaluate the testing accuracy
    pred = dt.predict(df.iloc[:, 1:])
    acc = accuracy(pred, df.iloc[:, 0])
    print("Test set accuracy=" + str(acc))
