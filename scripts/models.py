#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Models used by other scripts.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
from utils import *


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
    trcm = np.zeros(shape=(len(cl), len(cl)))
    for labels, conmat in [[te_l, cm], [tr_l, trcm]]:
        for l in labels:
            conmat[l][mc] += 1

    # Return accuracy and cm.
    return get_acc(cm), cm, get_acc(trcm), trcm


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
    trcm = np.zeros(shape=(len(cp), len(cp)))
    for feats, labels, conmat in [[te_f, te_l, cm], [tr_f, tr_l, trcm]]:
        for idx in range(len(feats)):
            tc = labels[idx]
            x = feats[idx]
            y_probs = [np.log(cp[c]) + reduce(lambda _x, _y: _x + _y, [np.log(fp_c[c][jdx][x[jdx]])
                                                                       for jdx in range(len(x))])
                       for c in cf]
            conmat[tc][y_probs.index(max(y_probs))] += 1
    if verbose:
        print("NB: ... done")

    # Return accuracy and cm.
    return get_acc(cm), cm, get_acc(trcm), trcm


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
    loss_function = nn.CrossEntropyLoss(ignore_index=word_to_i['<_>'])
    optimizer = optim.SGD(m.parameters(), lr=0.001)  # TODO: this could be touched up.
    idxs = list(range(len(tr_inputs)))
    np.random.shuffle(idxs)
    tr_inputs = [tr_inputs[idx] for idx in idxs]
    tr_outputs = tr_outputs[idxs, :]
    idx = 0
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
            optimizer.step()  # this is fun!  these boots were made for walking, and that's just what they'll do
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
        trcm = np.zeros(shape=(len(classes), len(classes)))
        for feats, labels, conmat in [[te_inputs, te_l, cm], [tr_inputs, tr_l, trcm]]:
            for jdx in range(len(feats)):
                slogits = torch.zeros(1, len(classes))
                for vidx in range(len(feats[jdx])):
                    _, logits = m(feats[jdx][vidx])
                    slogits += logits
                pc = slogits.max(1)[1]
                conmat[labels[jdx]][pc] += 1
    if verbose:
        print("L2L: ... done")

    # Return accuracy and cm.
    return get_acc(cm), cm, get_acc(trcm), trcm


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
def run_ff_model(dv, wv, tr_f, tr_l, te_f, te_l,
                 layers=None, batch_size=None, epochs=None,
                 dropout=0, learning_rate=0.001, opt='sgd',
                 verbose=False):
    assert batch_size is not None or epochs is not None

    # Prepare the data.
    if verbose:
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
            model_in.append(torch.tensor(cat_v, dtype=torch.float).to(dv))
            inwidth = len(model_in[-1])
            if orig_out[idx] not in classes:
                classes.append(orig_out[idx])
    outwidth = len(classes)
    tr_outputs = torch.tensor(tr_l, dtype=torch.long).to(dv).view(len(tr_l), 1)
    if epochs is None:  # iterate given the batch size to cover all data at least once
        epochs = int(np.ceil(len(tr_inputs) / float(batch_size)))
    if batch_size is None:
        batch_size = len(tr_inputs)
    if verbose:
        print("FF: ... done")

    # Construct the model with specified number of hidden layers / dimensions (or none) and relu activations between.
    if verbose:
        print("FF: constructing model...")
    if layers is not None:
        lr = [nn.Linear(inwidth, layers[0])]
        for idx in range(1, len(layers)):
            lr.append(torch.nn.ReLU())
            if dropout > 0:
                lr.append(torch.nn.Dropout(dropout))
            lr.append(nn.Linear(layers[idx - 1], layers[idx]))
        lr.append(torch.nn.ReLU())
        lr.append(nn.Linear(layers[-1], outwidth))
    else:
        lr = [nn.Linear(inwidth, outwidth)]
    model = nn.Sequential(*lr).to(dv)
    if verbose:
        print("FF: ... done")

    # Train: train a neural model for a fixed number of epochs.
    if verbose:
        print("FF: training on " + str(len(tr_inputs)) + " inputs with batch size " + str(batch_size) +
              " for " + str(epochs) + " epochs...")
    loss_function = nn.CrossEntropyLoss()

    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Unrecognized opt specification "' + opt + '".')

    best_acc = best_cm = tr_acc_at_best = trcm_at_best = None
    idxs = list(range(len(tr_inputs)))
    np.random.shuffle(idxs)
    tr_inputs = [tr_inputs[idx] for idx in idxs]
    tr_outputs = tr_outputs[idxs, :]
    idx = 0
    for epoch in range(epochs):
        tloss = 0
        c = 0
        trcm = np.zeros(shape=(len(classes), len(classes)))  # note: calculates train acc only over curr batch
        while c < batch_size:
            model.zero_grad()
            logits = model(tr_inputs[idx])
            loss = loss_function(logits.view(1, len(logits)), tr_outputs[idx])
            tloss += loss.data.item()
            trcm[tr_l[idx]][logits.max(0)[1]] += 1
            loss.backward()
            optimizer.step()
            c += 1
            idx += 1
            if idx == len(tr_inputs):
                idx = 0
        tloss /= batch_size
        tr_acc = get_acc(trcm)

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
                tr_acc_at_best = tr_acc
                trcm_at_best = trcm

        if verbose:
            print("... epoch " + str(epoch) + " train loss " + str(tloss) + "; train accuracy " + str(tr_acc) +
                  "; test accuracy " + str(acc))
    if verbose:
        print("FF: ... done")

    return best_acc, best_cm, tr_acc_at_best, trcm_at_best


# Based on:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class EarlyFusionFFModel(nn.Module):

    # modality_widths - list of input widths, e.g., [300, 1000] means first input width 300, second 1000
    # shrink_to_min - add FC shrinking layers on all but smallest modality to shrink them to smallest before fusion
    # hidden_layer_widths - list of hidden layer widths between fusion and prediction (possibly empty)
    # dropout - the dropout rate to add before hidden layer connections
    # num_classes - the number of classes to predict over in the output layer.
    def __init__(self, dv, modality_widths, shrink_to_min, hidden_layer_widths, dropout, num_classes):
        super(EarlyFusionFFModel, self).__init__()
        self.param_list = []
        self.num_modalities = len(modality_widths)

        # Add shrinking layers to bring larger modality inputs to smallest input width before fusion.
        if shrink_to_min and len(set(modality_widths)) != 1:
            target_width = min(modality_widths)
            self.shrink_layers = {sidx: nn.Linear(modality_widths[sidx], target_width).to(dv)
                                  for sidx in range(len(modality_widths)) if modality_widths[sidx] > target_width}
            input_cat_width = target_width * len(modality_widths)
        else:
            self.shrink_layers = {}
            input_cat_width = sum(modality_widths)
        for sidx in self.shrink_layers:
            self.param_list.extend(list(self.shrink_layers[sidx].parameters()))

        # In forward pass, inputs are then concatenated and fed to a number (possibly zero) of hidden layers.
        if len(hidden_layer_widths) > 0:
            self.hidden_layers = [nn.Linear(input_cat_width, hidden_layer_widths[0]).to(dv)]
            for idx in range(1, len(hidden_layer_widths)):
                self.hidden_layers.append(torch.nn.ReLU())
                if dropout > 0:
                    self.hidden_layers.append(torch.nn.Dropout(dropout))
                self.hidden_layers.append(nn.Linear(hidden_layer_widths[idx - 1], hidden_layer_widths[idx]).to(dv))
            self.hidden_layers.append(torch.nn.ReLU())
            self.hidden_layers.append(nn.Linear(hidden_layer_widths[-1], num_classes).to(dv))
        else:
            self.hidden_layers = [nn.Linear(input_cat_width, num_classes).to(dv)]  # e.g., no hidden layers just in->out
        for hl in self.hidden_layers:
            self.param_list.extend(list(hl.parameters()))

    # inputs - a list of tensor inputs
    def forward(self, inputs):

        # Shrink inputs.
        s_inputs = [self.shrink_layers[sidx](inputs[sidx]) if sidx in self.shrink_layers else inputs[sidx]
                    for sidx in range(self.num_modalities)]

        # Concatenate inputs.
        cat_inputs = torch.cat(s_inputs, 0)

        # Feed concatenation through hidden layers and predict classes.
        ff_inputs = [cat_inputs]
        for hl in self.hidden_layers:
            ff_inputs.append(hl(ff_inputs[-1]))
        logits = ff_inputs[-1]

        return logits
