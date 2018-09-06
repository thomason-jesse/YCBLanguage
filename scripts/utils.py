#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Functions used by other scripts.

import numpy as np
from collections import Counter


# Get accuracy from a confusion matrix with an arbitrary number of classes.
def get_acc(cm):
    return np.trace(cm) / np.sum(cm)


# Get f1 from a confusion matrix, averaging the "real" classes' f1 scores and ignoring the "maybe" class f1.
# cm - a 3x3 confusion matrix with class 1 representing "maybe"/"undecided"
# cm - a 2x2 confusion matrix with class 0 N and class 1 Y
def get_f1(cm):
    if cm.shape == (2, 2):
        tp = cm[1, 1]
        p = tp / np.sum(cm[:, 1])
        r = tp / np.sum(cm[1, :])
    elif cm.shape == (3, 3):
        tp = cm[0, 0] + cm[2, 2]  # true positives for Y/N labels
        decided_y_n = np.sum(cm[:, 0]) + np.sum(cm[:, 2])
        if decided_y_n > 0:
            p = tp / decided_y_n  # precision when we decided to say Y or N
        else:
            p = 0  # set precision to zero (truly it is "undefined" since we labeled everything "maybe")
        # We don't need this check for recall, since we know the true labels are not all clustered on "maybe"
        r = tp / (np.sum(cm[0, :]) + np.sum(cm[2, :]))  # recall when the true label was Y or N
    else:
        raise ValueError
    if p + r > 0:  # if tp = 0, this can happen
        f = (2 * p * r) / (p + r)  # harmonic mean of Y/N-based precision and recall
    else:
        f = 0
    return f


# Make the language structures needed for other models.
# tr_f - referring expression structure for training
# te_f - referring expresison structure for testing
# inc_test - whether to 'see' test words unseen during training (UNK when false)
def make_lang_structures(tr_f, te_f, inc_test=False):
    word_to_i = {"<?>": 0, "<s>": 1, "<e>": 2, "<_>": 4}
    i_to_word = {0: "<?>", 1: "<s>", 2: "<e>", 3: "<_>"}
    maxlen = 0
    tr_enc_exps = []
    for idx in range(len(tr_f)):
        pair_re_is = []
        for oidx in range(2):
            ob_re_is = []
            for re in tr_f[idx][oidx]:
                re_is = [word_to_i["<s>"]]
                for w in re:
                    if w not in word_to_i:
                        i = len(i_to_word)
                        word_to_i[w] = i
                        i_to_word[i] = w
                    re_is.append(word_to_i[w])
                re_is.append(word_to_i["<e>"])
                maxlen = max(maxlen, len(re_is))
                ob_re_is.append(re_is)
            pair_re_is.append(ob_re_is)
        tr_enc_exps.append(pair_re_is)

    te_enc_exps = []
    for idx in range(len(te_f)):
        pair_re_is = []
        for oidx in range(2):
            ob_re_is = []
            for re in te_f[idx][oidx]:
                re_is = [word_to_i["<s>"]]
                for w in re:
                    if w not in word_to_i:
                        if inc_test:
                            i = len(i_to_word)
                            word_to_i[w] = i
                            i_to_word[i] = w
                        else:
                            i = word_to_i["<?>"]
                    else:
                        i = word_to_i[w]
                    re_is.append(i)
                re_is.append(word_to_i["<e>"])
                maxlen = max(maxlen, len(re_is))
                ob_re_is.append(re_is)
            pair_re_is.append(ob_re_is)
        te_enc_exps.append(pair_re_is)

    return word_to_i, i_to_word, maxlen, tr_enc_exps, te_enc_exps


# Get GLoVe vectors from target input file for given vocabulary.
# fn - the filename where the GLoVe vectors live.
# ws - the set of words to look up.
def get_glove_vectors(fn, ws):
    g = {}
    unk_v = None
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            p = line.strip().split(' ')
            w = p[0]
            if w in ws:
                v = np.array([float(n) for n in p[1:]])
                g[w] = v
            if w == "unk":
                unk_v = np.array([float(n) for n in p[1:]])
    m = 0
    for w in ws:
        if w not in g:
            m += 1
            g[w] = unk_v
    assert set(g.keys()) == ws
    return g, m


# Get BoW one-hot vectors for the given vocabulary.
# ws - the set of words to assign one-hots.
def get_bow_vectors(ws):
    wv = {}
    wl = list(ws)
    for w in ws:
        wv[w] = np.zeros(len(ws))
        wv[w][wl.index(w)] = 1
    return wv


# Get class labels by a vote over multiple trials.
# tf - trial features, a list of lists of per-trial features
# p - a list of class predictions on a per-trial basis
# returns - a list of size len(tf) of majority class vote over trials
def get_classes_by_vote(tf, p):
    pidx = 0
    c = []
    for idx in range(len(tf)):
        class_votes = Counter(p[pidx:pidx + len(tf[idx])])
        c.append(max(class_votes.keys(), key=(lambda k: class_votes[k])))
        pidx += len(tf[idx])
    return c


# Get class softmax by a vote over multiple trials.
# Assumes classes are numbered as 0, 1, ..., N
# tf - trial features, a list of lists of per-trial features
# p - a list of class predictions on a per-trial basis
# num_classes - if 3, calculates as-is; if 2, assumes 0 -> 0, 1 -> 2, and introduces a new class 1 with no votes
# returns - a list of size len(tf) of list of class vote softmax over trials
def get_softmax_by_vote(tf, p, num_classes):
    pidx = 0
    c = []
    for idx in range(len(tf)):
        class_votes = Counter(p[pidx:pidx + len(tf[idx])])
        c.append(np.asarray([class_votes[cidx] / float(len(tf[idx])) for cidx in range(num_classes)]))
        if num_classes == 2:  # insert a 0 vote 'maybe' between no/yes counts
            c[-1] = np.insert(c[-1], 1, 0)
        pidx += len(tf[idx])
    return c
