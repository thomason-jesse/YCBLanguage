#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs simple baselines and reports performance.

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from functools import reduce
from torch.autograd import Variable


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


# Based on:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LSTMTagger(nn.Module):

    def __init__(self, width, vocab_size, num_classes):
        super(LSTMTagger, self).__init__()
        self.width = width
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.word_embeddings = nn.Embedding(vocab_size, width)
        self.lstm_src = nn.LSTM(width, width)
        self.lstm_des = nn.LSTM(width, width)
        self.hidden2class = nn.Linear(width*2, num_classes)

        self.hidden_src = self.init_hidden()
        self.hidden_des = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.width),
                torch.zeros(1, 1, self.width))

    def forward(self, res):
        embeds_src = self.word_embeddings(res[0])
        embeds_des = self.word_embeddings(res[0])
        lstm_src_out, self.hidden_src = self.lstm_src(embeds_src.view(len(res[0]), 1, -1), self.hidden_src)
        lstm_des_out, self.hidden_des = self.lstm_des(embeds_des.view(len(res[0]), 1, -1), self.hidden_des)
        hcat = torch.cat((lstm_src_out, lstm_des_out), 0)
        final_logits = self.hidden2class(hcat.view(len(res[0]), -1))[-1, :]
        final_logits = final_logits.view(1, len(final_logits))
        final_scores = F.softmax(final_logits, dim=1)
        return final_scores, final_logits


# Train an LSTM language encoder to predict the category label given language descriptions.
# tr_f - lists of descriptions for training instances
# tr_l - training labels
# te_f - lists of descriptions for testing instances
# te_l - testing labels
# verbose - whether to print epoch-wise progress
# epochs - number of epochs to train
# width - width of the encoder LSTM
def run_lang_2_label(tr_f, tr_l, te_f, te_l,
                     verbose=False, epochs=100, width=32):

    # Preprocess: turn words into one-hot vectors, count number of classes, construct model.
    word_to_i = {"<?>": 0, "<s>": 1, "<e>": 2, "<_>": 4}
    i_to_word = {0: "<?>", 1: "<s>", 2: "<e>", 3: "<_>"}
    classes = []
    maxlen = 0
    enc_exps = []
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
                ob_re_is.append(torch.tensor(re_is, dtype=torch.long))
            pair_re_is.append(ob_re_is)
        enc_exps.append(pair_re_is)
        if tr_l[idx] not in classes:
            classes.append(tr_l[idx])

    # Train on the cross product of every description of source and description of destination object.
    tr_inputs = []
    tr_outputs = []
    p = nn.ConstantPad1d((0, maxlen), word_to_i["<_>"])
    for idx in range(len(enc_exps)):
        for ridx in range(len(enc_exps[idx][0])):
            for rjdx in range(len(enc_exps[idx][1])):
                tr_inputs.append([p(enc_exps[idx][0][ridx]), p(enc_exps[idx][1][rjdx])])
                tr_outputs.append(tr_l[idx])
    tr_outputs = torch.tensor(tr_outputs, dtype=torch.long).view(len(tr_outputs), 1)

    # At test time, run the model on the cross product of descriptions for the pair and sum logits.
    te_enc_exps = []
    for idx in range(len(tr_f)):
        pair_re_is = []
        for oidx in range(2):
            ob_re_is = []
            for re in tr_f[idx][oidx]:
                re_is = [word_to_i["<s>"]]
                for w in re:
                    if w not in word_to_i:
                        i = word_to_i["<?>"]
                    else:
                        i = word_to_i[w]
                    re_is.append(i)
                re_is.append(word_to_i["<e>"])
                maxlen = max(maxlen, len(re_is))
                ob_re_is.append(torch.tensor(re_is, dtype=torch.long))
            pair_re_is.append(ob_re_is)
        te_enc_exps.append(pair_re_is)
    te_inputs = []
    te_outputs = []
    for idx in range(len(te_enc_exps)):
        pairs_in = []
        for ridx in range(len(te_enc_exps[idx][0])):
            for rjdx in range(len(te_enc_exps[idx][1])):
                pairs_in.append([p(te_enc_exps[idx][0][ridx]), p(te_enc_exps[idx][1][rjdx])])
        te_inputs.append(pairs_in)
        te_outputs.append(tr_l[idx])
    te_outputs = torch.tensor(te_outputs, dtype=torch.long).view(len(te_outputs), 1)

    # Train: train a neural model for a fixed number of epochs.
    m = LSTMTagger(width, len(word_to_i), len(classes))
    loss_function = nn.CrossEntropyLoss(ignore_index = word_to_i['<_>'])
    optimizer = optim.SGD(m.parameters(), lr=0.1)
    cm = acc = None
    for epoch in range(epochs):
        tloss = 0
        idxs = list(range(len(tr_inputs)))
        np.random.shuffle(idxs)
        tr_inputs = [tr_inputs[idx] for idx in idxs]
        tr_outputs = tr_outputs[idxs, :]
        for idx in range(len(tr_inputs)):
            m.zero_grad()
            m.hidden_src = m.init_hidden()
            m.hidden_des = m.init_hidden()
            _, logits = m(tr_inputs[idx])
            loss = loss_function(logits, tr_outputs[idx])
            tloss += loss.data.item()
            loss.backward()
            optimizer.step()
        tloss /= len(tr_inputs)

        # Test: report accuracy after every epoch, finally returning accuracy at the final one.
        cm = np.zeros(shape=(len(classes), len(classes)))
        with torch.no_grad():
            for idx in range(len(te_inputs)):
                slogits = torch.zeros(len(classes))
                for vidx in range(len(te_inputs[idx])):
                    _, logits = model(inputs)
                    slogits += logits
                pc = slogits.max(0)
                cm[te_l[idx]][pc] += 1
            acc = get_acc
            print("... epoch " + str(epoch) + " train loss " + str(tloss) + "; test accuracy " + str(acc))

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

    # Read in metadata.
    print("Reading in metadata from '" + args.metadata_infile + "'...")
    with open(args.metadata_infile, 'r') as f:
        d = json.load(f)
        res = d["res"]
    print("... done")

    bs = []
    rs = []

    # Majority class baseline.
    print("Running majority class baseline...")
    bs.append("Majority Class")
    rs.append({})
    for p in preps:
        rs[0][p] = run_majority_class(train[p]["label"], test[p]["label"])
    print("... done")

    # Majority class | object ids baseline.
    print("Running Naive Bayes baseline...")
    bs.append("Naive Bayes Obj One Hots")
    rs.append({})
    fs = [range(len(names)), range(len(names))]  # Two one-hot vectors of which object name was seen.
    for p in preps:
        tr_f = [[train[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                range(len(train[p]["ob1"]))]
        te_f = [[test[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                range(len(test[p]["ob1"]))]
        rs[1][p] = run_cat_naive_bayes(fs, tr_f, train[p]["label"], te_f, test[p]["label"])
    print("... done")

    # Language encoder (train from scratch)
    print("Running language encoder baseline...")
    bs.append("Language Encoder")
    rs.append({})
    for p in preps:
        tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                range(len(train[p]["ob1"]))]
        te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                range(len(test[p]["ob1"]))]
        rs[1][p] = run_lang_2_label(tr_f, train[p]["label"], te_f, test[p]["label"])
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
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    main(parser.parse_args())
