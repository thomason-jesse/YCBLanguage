#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs simple baselines and reports performance.

import argparse
import numpy as np
import json
from functools import reduce


# Get accuracy from a confusion matrix.
def get_acc(cm):
    return np.trace(cm) / np.sum(cm)


# Count the majority class label in the train set and use it as a classification decision on every instance
# in the test set.
# tr_l - training labels
# te_l - testing labels
def run_majority_class(tr_l, te_l):

    # Train: count the majority class across all labels.
    cl = {}
    for l in tr_l:
        if l not in cl:
            cl[l] = 0
        cl[l] += 1
    mc = sorted(cl, key=cl.get, reverse=True)[0]

    # Test: for every instance, guess the majority class.
    cm = np.zeros(shape=(len(cl), len(cl)))
    for l in te_l:
        cm[l][mc] += 1

    # Return accuracy and cm.
    return get_acc(cm), cm


# A naive bayes implementation that assumes known categorical feature values.
# fs - feature shape, list of size |F| with entries the range of categorical values per feature.
# tr_f - training features
# tr_l - training labels
# te_f - testing features
# te_l - testing labels
def run_cat_naive_bayes(fs, tr_f, tr_l, te_f, te_l):

    # Train: calculate the prior per class and the conditional likelihood of each feature given the class.
    cf = {}  # class frequency
    ff_c = {}  # categorical feature frequency given class
    for idx in range(len(tr_f)):
        c = tr_l[idx]
        x = tr_f[idx]
        if c not in cf:
            cf[c] = 0
            ff_c[c] = [{cat: 0 for cat in fs[jdx]} for jdx in range(len(x))]
        cf[c] += 1
        for jdx in range(len(x)):
            ff_c[c][jdx][x[jdx]] += 1
    cp = {c: cf[c] / float(len(tr_f)) for c in cf}  # class prior
    fp_c = {c: [{cat: ff_c[c][jdx][cat] / float(cf[c]) for cat in fs[jdx]} for jdx in range(len(tr_f[0]))]
            for c in ff_c}  # categorical feature probability given class

    # Test: calculate the joint likelihood of each class for each instance conditioned on its features.
    cm = np.zeros(shape=(len(cp), len(cp)))
    for idx in range(len(te_f)):
        tc = te_l[idx]
        x = te_f[idx]
        y_probs = [cp[c] * reduce(lambda _x, _y: _x * _y, [fp_c[c][jdx][x[jdx]] for jdx in range(len(x))])
                   for c in cf]
        cm[tc][y_probs.index(max(y_probs))] += 1

    # Return accuracy and cm.
    return get_acc(cm), cm


def main(args):

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.infile + "'...")
    with open(args.infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        lf = all_d["folds"]
        train = lf['train']
        test = lf['dev']  # only ever peak at the dev set.
        preps = train.keys()
    print("... done")
    bs = []
    rs = []

    # Majority class baseline.
    print("Running majority class baseline...")
    bs.append("Majority Class")
    rs.append({})
    for p in preps:
        rs[0][p] = run_majority_class(train[p]['label'], test[p]['label'])
    print("... done")

    # Majority class | object ids baseline.
    print("Running Naive Bayes baseline...")
    bs.append("Naive Bayes Obj One Hots")
    rs.append({})
    fs = [range(len(names)), range(len(names))]  # Two one-hot vectors of which object name was seen.
    for p in preps:
        tr_f = [[names.index(n) for n in lf['train'][p]['pair'][idx].split('+')] for idx in
                range(len(lf['train'][p]['pair']))]
        te_f = [[names.index(n) for n in lf['test'][p]['pair'][idx].split('+')] for idx in
                range(len(lf['test'][p]['pair']))]
        rs[1][p] = run_cat_naive_bayes(fs, tr_f, train[p]['label'], te_f, test[p]['label'])
    print("... done")

    # Show results.
    print("Results:")
    for idx in range(len(bs)):
        print(" " + bs[idx] + ":")
        for p in preps:
            print("  " + p + ":\t%0.3f" % rs[idx][p][0])
            print('\t(CM\t' + '\n\t\t'.join(['\t'.join([str(int(ct)) for ct in rs[idx][p][1][i]])
                                             for i in range(len(rs[idx][p][1]))]) + ")")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    main(parser.parse_args())
