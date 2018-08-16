#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
from PIL import Image
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize


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
def run_cat_naive_bayes(fs, tr_f, tr_l, te_f, te_l,
                        verbose=False, smooth=0):

    # Train: calculate the prior per class and the conditional likelihood of each feature given the class.
    if verbose:
        print("NB: training on " + str(len(tr_f)) + " inputs...")
    cf = {}  # class frequency
    ff_c = {}  # categorical feature frequency given class
    for idx in range(len(tr_f)):
        c = tr_l[idx]
        x = tr_f[idx]
        if c not in cf:
            cf[c] = smooth
            ff_c[c] = [{cat: smooth for cat in fs[jdx]} for jdx in range(len(x))]
        cf[c] += 1
        for jdx in range(len(x)):
            ff_c[c][jdx][x[jdx]] += 1
    cp = {c: cf[c] / float(len(tr_f)) for c in cf}  # class prior
    fp_c = {c: [{cat: ff_c[c][jdx][cat] / float(cf[c]) for cat in fs[jdx]} for jdx in range(len(tr_f[0]))]
            for c in ff_c}  # categorical feature probability given class
    if verbose:
        print("NB: ... done")

    # Test: calculate the joint likelihood of each class for each instance conditioned on its features.
    if verbose:
        print("NB: testing on " + str(len(te_f)) + " outputs...")
    cm = np.zeros(shape=(len(cp), len(cp)))
    for idx in range(len(te_f)):
        tc = te_l[idx]
        x = te_f[idx]
        y_probs = [np.log(cp[c]) + reduce(lambda _x, _y: _x + _y, [np.log(fp_c[c][jdx][x[jdx]])
                                                                   for jdx in range(len(x))])
                   for c in cf]
        cm[tc][y_probs.index(max(y_probs))] += 1
    if verbose:
        print("NB: ... done")

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
# batch_size - number of training examples to run before gradient update
# epochs - number of epochs to run over data; if None, runs over all data once
def run_lang_2_label(maxlen, word_to_i,
                     tr_enc_exps, tr_l, te_enc_exps, te_l,
                     verbose=False, width=16, batch_size=100, epochs=None):

    # Train on the cross product of every description of source and description of destination object.
    if verbose:
        print("L2L: preparing training and testing data...")
    classes = []
    tr_inputs = []
    tr_outputs = []
    p = nn.ConstantPad1d((0, maxlen), word_to_i["<_>"])
    for idx in range(len(tr_enc_exps)):
        for ridx in range(len(tr_enc_exps[idx][0])):
            for rjdx in range(len(tr_enc_exps[idx][1])):
                tr_inputs.append([p(torch.tensor(tr_enc_exps[idx][0][ridx], dtype=torch.long)),
                                  p(torch.tensor(tr_enc_exps[idx][1][rjdx], dtype=torch.long))])
                tr_outputs.append(tr_l[idx])
                if tr_l[idx] not in classes:
                    classes.append(tr_l[idx])
    tr_outputs = torch.tensor(tr_outputs, dtype=torch.long).view(len(tr_outputs), 1)
    if epochs is None:  # iterate given the batch size to cover all data at least once
        epochs = int(np.ceil(len(tr_inputs) / float(batch_size)))

    # At test time, run the model on the cross product of descriptions for the pair and sum logits.
    te_inputs = []
    for idx in range(len(te_enc_exps)):
        pairs_in = []
        for ridx in range(len(te_enc_exps[idx][0])):
            for rjdx in range(len(te_enc_exps[idx][1])):
                pairs_in.append([p(torch.tensor(te_enc_exps[idx][0][ridx], dtype=torch.long)),
                                 p(torch.tensor(te_enc_exps[idx][1][rjdx], dtype=torch.long))])
        te_inputs.append(pairs_in)
    if verbose:
        print("L2L: ... done")

    # Train: train a neural model for a fixed number of epochs.
    if verbose:
        print("L2L: training on " + str(len(tr_inputs)) + " inputs with batch size " + str(batch_size) + " for " + str(epochs) + " epochs...")
    m = LSTMTagger(width, len(word_to_i), len(classes))
    loss_function = nn.CrossEntropyLoss(ignore_index = word_to_i['<_>'])
    optimizer = optim.SGD(m.parameters(), lr=0.001)  # TODO: this could be touched up.
    cm = acc = None
    idxs = list(range(len(tr_inputs)))
    np.random.shuffle(idxs)
    tr_inputs = [tr_inputs[idx] for idx in idxs]
    tr_outputs = tr_outputs[idxs, :]
    idx = 0
    c = 0
    for epoch in range(epochs):
        tloss = 0
        c = 0
        while c < batch_size:
            m.zero_grad()
            m.hidden_src = m.init_hidden()
            m.hidden_des = m.init_hidden()
            _, logits = m(tr_inputs[idx])
            loss = loss_function(logits, tr_outputs[idx])
            tloss += loss.data.item()
            loss.backward()
            optimizer.step()
            c += 1
            idx += 1
            if idx == len(tr_inputs):
                idx = 0
        tloss /= batch_size
        print("... epoch " + str(epoch) + " train loss " + str(tloss))
    if verbose:
        print("L2L: ... done")

    # Test: report accuracy after training.
    if verbose:
        print("L2L: calculating test-time accuracy...")
    with torch.no_grad():
        cm = np.zeros(shape=(len(classes), len(classes)))
        for jdx in range(len(te_inputs)):
            slogits = torch.zeros(1, len(classes))
            for vidx in range(len(te_inputs[jdx])):
                _, logits = m(te_inputs[jdx][vidx])
                slogits += logits
            pc = slogits.max(1)[1]
            cm[te_l[jdx]][pc] += 1
        acc = get_acc(cm)
    if verbose:
        print("L2L: ... done; " + str(acc))

    # Return accuracy and cm.
    return acc, cm


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
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            p = line.strip().split(' ')
            w = p[0]
            if w in ws:
                v = np.array([float(n) for n in p[1:]])
                g[w] = v
    m = 0
    for w in ws:
        if w not in g:
            m += 1
            v = np.zeros(len(g[list(g.keys())[0]]))
            g[w] = v
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


# Run a GLoVe-based model (either perceptron or FF network with relu activations)
# wv - list (by modality) of dictionaries mapping all vocabulary words to their word vectors
# tr_f - list of feature modalities, indexed then by oidx, each has a list of input lists whose members can be looked
#        up in the vocabulary to vector dictionary wv
# tr_l - training labels
# te_f - same as tr_f but for testing
# te_l - testing labels
# layers - a list of hidden widths for a FF network or None to create a linear perceptron
# verbose - whether to print epoch-wise progress
# Reference: https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
def run_ff_model(wv, tr_f, tr_l, te_f, te_l,
                 layers=None, batch_size=None, epochs=None,
                 verbose=False):
    assert batch_size is not None or epochs is not None

    # Prepare the data.
    print("FF: preparing training and testing data...")
    tr_inputs = []
    te_inputs = []
    classes = []
    inwidth = None
    for model_in, orig_in, orig_out in [[tr_inputs, tr_f, tr_l], [te_inputs, te_f, te_l]]:
        for idx in range(len(orig_in[0])):
            cat_v = []
            for midx in range(len(orig_in)):
                v = wv[midx]
                modality = orig_in[midx]
                ob1_ws = [w for ws in modality[idx][0] for w in ws]
                avg_ob1_v = np.sum([v[w] for w in ob1_ws], axis=0) / len(ob1_ws)
                ob2_ws = [w for ws in modality[idx][1] for w in ws]
                avg_ob2_v = np.sum([v[w] for w in ob2_ws], axis=0) / len(ob2_ws)
                incat = np.concatenate((avg_ob1_v, avg_ob2_v))
                cat_v = np.concatenate((cat_v, incat))
            model_in.append(torch.tensor(cat_v, dtype=torch.float))
            inwidth = len(model_in[-1])
            if orig_out[idx] not in classes:
                classes.append(orig_out[idx])
    outwidth = len(classes)
    tr_outputs = torch.tensor(tr_l, dtype=torch.long).view(len(tr_l), 1)
    if epochs is None:  # iterate given the batch size to cover all data at least once
        epochs = int(np.ceil(len(tr_inputs) / float(batch_size)))
    if batch_size is None:
        batch_size = len(tr_inputs)
    print("FF: ... done")

    # Construct the model with specified number of hidden layers / dimensions (or none) and relu activations between.
    print("FF: constructing model...")
    if layers is not None:
        lr = [nn.Linear(inwidth, layers[0])]
        for idx in range(1, len(layers)):
            lr.append(torch.nn.ReLU())
            lr.append(nn.Linear(layers[idx - 1], layers[idx]))
        lr.append(torch.nn.ReLU())
        lr.append(nn.Linear(layers[-1], outwidth))
    else:
        lr = [nn.Linear(inwidth, outwidth)]
    model = nn.Sequential(*lr)
    print("FF: ... done")

    # Train: train a neural model for a fixed number of epochs.
    if verbose:
        print("FF: training on " + str(len(tr_inputs)) + " inputs with batch size " + str(batch_size) +
              " for " + str(epochs) + " epochs...")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # TODO: this could be touched up.
    best_acc = best_cm = None
    idxs = list(range(len(tr_inputs)))
    np.random.shuffle(idxs)
    tr_inputs = [tr_inputs[idx] for idx in idxs]
    tr_outputs = tr_outputs[idxs, :]
    idx = 0
    for epoch in range(epochs):
        tloss = 0
        c = 0
        trcm = np.zeros(shape=(len(classes), len(classes)))
        while c < batch_size:
            model.zero_grad()
            logits = model(tr_inputs[idx])
            loss = loss_function(logits.view(1, len(logits)), tr_outputs[idx])
            tloss += loss.data.item()
            trcm[tr_l[c]][logits.max(0)[1]] += 1
            loss.backward()
            optimizer.step()
            c += 1
            idx += 1
            if idx == len(tr_inputs):
                idx = 0
        tloss /= batch_size

        with torch.no_grad():
            cm = np.zeros(shape=(len(classes), len(classes)))
            for jdx in range(len(te_inputs)):
                logits = model(te_inputs[jdx])
                pc = logits.max(0)[1]
                cm[te_l[jdx]][pc] += 1
            acc = get_acc(cm)
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_cm = cm

        print("... epoch " + str(epoch) + " train loss " + str(tloss) + "; train accuracy " + str(get_acc(trcm)) +
              "; test accuracy " + str(acc))
    if verbose:
        print("FF: ... done")

    return best_acc, best_cm


def main(args):
    assert args.baseline is None or args.baseline in ['majority', 'nb_names', 'nb_bow',
                                                      'lstm', 'glove', 'nn_bow', 'resnet', 'glove+resnet']
    assert args.glove_infile is not None or (args.baseline is not None and args.baseline != 'glove')
    verbose = True if args.verbose == 1 else False

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
        imgs = d["imgs"]
    print("... done")

    bs = []
    rs = []

    # Majority class baseline.
    if args.baseline is None or args.baseline == 'majority':
        print("Running majority class baseline...")
        bs.append("Majority Class")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_majority_class(train[p]["label"], test[p]["label"])
        print("... done")

    # Majority class | object ids baseline.
    if args.baseline is None or args.baseline == 'nb_names':
        print("Running Naive Bayes baseline...")
        bs.append("Naive Bayes Obj One Hots")
        rs.append({})
        fs = [range(len(names)), range(len(names))]  # Two one-hot vectors of which object name was seen.
        for p in preps:
            tr_f = [[train[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[test[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            rs[-1][p] = run_cat_naive_bayes(fs, tr_f, train[p]["label"], te_f, test[p]["label"], verbose=verbose)
        print("... done")

    # Prep language dictionary.
    maxlen = tr_enc_exps = te_enc_exps = word_to_i_all = word_to_i = None
    if args.baseline is None or args.baseline in ['nb_bow', 'lstm', 'glove', 'nn_bow', 'glove+resnet']:
        print("Preparing infrastructure to run language models...")
        word_to_i = {}
        word_to_i_all = {}
        i_to_word = {}
        maxlen = {}
        tr_enc_exps = {}
        te_enc_exps = {}
        for p in preps:
            tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            word_to_i[p], i_to_word[p], maxlen[p], tr_enc_exps[p], te_enc_exps[p] = make_lang_structures(tr_f, te_f)
            word_to_i_all[p], _, _, _, _ = make_lang_structures(tr_f, te_f, inc_test=True)
        print("... done")

    # Language naive bayes (bag of words)
    if args.baseline is None or args.baseline == 'nb_bow':
        print("Running BoW Naive Bayes...")
        bs.append("BoW Naive Bayes")
        rs.append({})
        for p in preps:
            print("... prepping model input and output data...")
            fs = [[0, 1] for _ in range(len(word_to_i[p]) * 2)]  # Two BoW vectors; one for each object.
            tr_inputs = []
            tr_outputs = []
            for idx in range(len(tr_enc_exps[p])):
                w_in_re_src = set()
                for ridx in range(len(tr_enc_exps[p][idx][0])):
                    w_in_re_src.update(tr_enc_exps[p][idx][0][ridx])
                w_in_re_tar = set()
                for rjdx in range(len(tr_enc_exps[p][idx][1])):
                    w_in_re_tar.update(tr_enc_exps[p][idx][1][rjdx])
                tr_inputs.append([1 if ((i < len(word_to_i[p]) and i in w_in_re_src) or
                                        (i >= len(word_to_i[p]) and i - len(word_to_i[p]) in w_in_re_tar)) else 0
                                  for i in range(len(word_to_i[p]) * 2)])
                tr_outputs.append(train[p]["label"][idx])
            te_inputs = []
            for idx in range(len(te_enc_exps[p])):
                w_in_re_src = set()
                for ridx in range(len(te_enc_exps[p][idx][0])):
                    w_in_re_src.update(te_enc_exps[p][idx][0][ridx])
                w_in_re_tar = set()
                for rjdx in range(len(te_enc_exps[p][idx][1])):
                    w_in_re_tar.update(te_enc_exps[p][idx][1][rjdx])
                te_inputs.append([1 if ((i < len(word_to_i[p]) and i in w_in_re_src) or
                                       (i >= len(word_to_i[p]) and i - len(word_to_i[p]) in w_in_re_tar)) else 0
                                 for i in range(len(word_to_i[p]) * 2)])
            print("...... done")
            rs[-1][p] = run_cat_naive_bayes(fs, tr_inputs, train[p]["label"], te_inputs, test[p]["label"],
                                            smooth=1, verbose=verbose)
        print ("... done")

    # Language encoder (train from scratch)
    if args.baseline is None or args.baseline == 'lstm':
        print("Running language encoder lstms...")
        bs.append("Language Encoder")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_lang_2_label(maxlen[p], word_to_i[p],
                                         tr_enc_exps[p], train[p]["label"], te_enc_exps[p], test[p]["label"],
                                         verbose=verbose, batch_size=10000)
        print("... done")

    # Average GLoVe embeddings concatenated and used to predict class.
    g = None
    emb_dim_l = None
    if args.baseline is None or args.baseline == 'glove' or args.baseline == 'glove+resnet':
        print("Preparing infrastructure to run GLoVe-based feed forward models...")
        ws = set()
        for p in preps:
            ws.update(word_to_i_all[p].keys())
        g, missing = get_glove_vectors(args.glove_infile, ws)
        emb_dim_l = len(g[list(g.keys())[0]])
        print("... done; missing " + str(missing) + " vectors out of " + str(len(ws)))

    if args.baseline is None or args.baseline == 'glove':
        print("Running GLoVe models")
        bs.extend(["GLoVe Perceptron", "GLoVe 1l FF", "GLoVe 2l FF"])
        rs.extend([{}, {}, {}])
        for p in preps:
            tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                     range(len(train[p]["ob1"]))]
            te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                     range(len(test[p]["ob1"]))]
            rs[-3][p] = run_ff_model([g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     epochs=30, verbose=verbose)
            rs[-2][p] = run_ff_model([g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim_l // 2], epochs=30, verbose=verbose)
            rs[-1][p] = run_ff_model([g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim_l // 2, emb_dim_l // 4], epochs=30, verbose=verbose)

    # Average BoW embeddings use to predict class.
    if args.baseline is None or args.baseline == 'nn_bow':
        print("Preparing infrastructure to run BoW-based feed forward models...")
        ws = set()
        for p in preps:
            ws.update(word_to_i_all[p].keys())
        wv = get_bow_vectors(ws)
        emb_dim = len(wv)
        print("... done")

        print("Running BoW FF models")
        bs.extend(["BoW Perceptron", "BoW 1l FF", "BoW 2l FF"])
        rs.extend([{}, {}, {}])
        for p in preps:
            tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            rs[-3][p] = run_ff_model([wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     epochs=30, verbose=verbose)
            rs[-2][p] = run_ff_model([wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim // 2], epochs=30, verbose=verbose)
            rs[-1][p] = run_ff_model([wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim // 2, emb_dim // 4], epochs=30, verbose=verbose)

    # Average BoW embeddings use to predict class.
    idx_to_v = None
    emb_dim_v = None
    if args.baseline is None or args.baseline == 'resnet' or args.baseline == 'glove+resnet':
        print("Preparing infrastructure to run ResNet-based feed forward models...")
        idx_to_v = {}
        resnet_m = resnet.resnet152(pretrained=True)
        plm = nn.Sequential(*list(resnet_m.children())[:-1])
        tt = ToTensor()
        for idx in range(len(names)):
            ffn = imgs[idx] + '.resnet'
            if os.path.isfile(ffn):
                with open(ffn, 'r') as f:
                    d = f.read().strip().split(' ')
                    v = np.array([float(n) for n in d])
                    idx_to_v[idx] = v
            else:
                pil = Image.open(imgs[idx])
                pil = resize(pil, (224, 244))
                im = tt(pil)
                im = torch.unsqueeze(im, 0)
                idx_to_v[idx] = plm(im).detach().data.numpy().flatten()
                with open(ffn, 'w') as f:
                    f.write(' '.join([str(i) for i in idx_to_v[idx]]))
        emb_dim_v = len(idx_to_v[0])
        print("... done")

    if args.baseline is None or args.baseline == 'resnet':
        print("Running ResNet FF models")
        bs.extend(["ResNet Perceptron", "ResNet 1l FF", "ResNet 2l FF"])
        rs.extend([{}, {}, {}])
        for p in preps:
            # FF expects a sequences of indices to be looked up in the vectors dictionary and averaged, so here
            # we just make each sequence [[oidx]] for the object idx to be looked up in the dictionary of pre-computed
            # vectors. It doesn't need to be averaged with anything. The double wrapping is because we expect first
            # a list of referring expressions, then a list of words inside each expression.
            tr_f = [[[[train[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[[[test[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            rs[-3][p] = run_ff_model([idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     epochs=30, verbose=verbose)
            rs[-2][p] = run_ff_model([idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim_v // 2], epochs=30, verbose=verbose)
            rs[-1][p] = run_ff_model([idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                     layers=[emb_dim_v // 2, emb_dim_v // 4], epochs=30, verbose=verbose)

    if args.baseline is None or args.baseline == 'glove+resnet':
        print("Running Glove+ResNet FF models")
        bs.extend(["GLoVe+ResNet Perceptron", "GLoVe+ResNet 1l FF", "GLoVe+ResNet 2l FF"])
        rs.extend([{}, {}, {}])
        emb_dim = emb_dim_l + emb_dim_v
        for p in preps:
            tr_f_l = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f_l = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            tr_f_v = [[[[train[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f_v = [[[[test[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            rs[-3][p] = run_ff_model([g, idx_to_v], [tr_f_l, tr_f_v], train[p]["label"],
                                     [te_f_l, te_f_v], test[p]["label"],
                                     epochs=30, verbose=verbose)
            rs[-2][p] = run_ff_model([g, idx_to_v], [tr_f_l, tr_f_v], train[p]["label"],
                                     [te_f_l, te_f_v], test[p]["label"],
                                     layers=[emb_dim // 2], epochs=30, verbose=verbose)
            rs[-1][p] = run_ff_model([g, idx_to_v], [tr_f_l, tr_f_v], train[p]["label"],
                                     [te_f_l, te_f_v], test[p]["label"],
                                     layers=[emb_dim // 2, emb_dim // 4], epochs=30, verbose=verbose)

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
    parser.add_argument('--baseline', type=str, required=False,
                        help="if None, all will run, else 'majority', 'nb_names', 'nb_bow', 'lstm', 'glove'," +
                             " 'nn_bow', 'resnet', 'glove+resnet'")
    parser.add_argument('--glove_infile', type=str, required=False,
                        help="input glove vector text file if running glove baseline")
    parser.add_argument('--verbose', type=int, required=False,
                        help="1 if desired")
    main(parser.parse_args())
