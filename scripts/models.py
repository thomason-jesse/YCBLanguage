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
def run_majority_class(tr_l, te_l, num_classes):

    # Train: count the majority class across all labels.
    cl = {}
    for l in tr_l:
        if l not in cl:
            cl[l] = 0
        cl[l] += 1
    mc = sorted(cl, key=cl.get, reverse=True)[0]

    # Test: for every instance, guess the majority class.
    cm = np.zeros(shape=(num_classes, num_classes))
    trcm = np.zeros(shape=(num_classes, num_classes))
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


# Run a GLoVe-based model (either perceptron or FF network with relu activations)
# tr_inputs - feature vector inputs
# tr_outputs - training labels
# te_inputs - same as tr_f but for testing
# te_outputs - testing labels
# verbose - whether to print epoch-wise progress
def run_ff_model(dv, tr_inputs, tr_outputs, te_inputs, te_outputs,
                 inwidth, hidden_dim, outwidth,  # single hidden layer of specified size, projecting to outwidth classes
                 batch_size=None, epochs=None,
                 dropout=0, learning_rate=0.001, opt='sgd',
                 verbose=False):
    assert batch_size is not None or epochs is not None

    # Prepare the data.
    if verbose:
        print("FF: preparing training and testing data...")
    if epochs is None:  # iterate given the batch size to cover all data at least once
        epochs = int(np.ceil(len(tr_inputs) / float(batch_size)))
    if batch_size is None:
        batch_size = len(tr_inputs)
    if verbose:
        print("FF: ... done")

    # Construct the model with specified number of hidden layers / dimensions (or none) and relu activations between.
    if verbose:
        print("FF: constructing model...")
    # TODO: these dropout layers might be in a stupid place.
    lr = [nn.Linear(inwidth, hidden_dim), torch.nn.Dropout(dropout), torch.nn.ReLU(),
          nn.Linear(hidden_dim, outwidth), torch.nn.Dropout(dropout), ]
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
    tro = tr_outputs[idxs, :]
    result = None
    for epoch in range(epochs):
        tloss = 0
        trcm = np.zeros(shape=(outwidth, outwidth))
        tidx = 0
        batches_run = 0
        while tidx < len(tr_inputs):
            model.zero_grad()
            batch_in = torch.zeros((batch_size, tr_inputs[0].shape[0])).to(dv)
            batch_gold = torch.zeros(batch_size).to(dv)
            for bidx in range(batch_size):
                batch_in[bidx, :] = tr_inputs[tidx]
                batch_gold[bidx] = tro[tidx][0]

                tidx += 1
                if tidx == len(tr_inputs):
                    break

            logits = model(batch_in)
            loss = loss_function(logits, batch_gold.long())
            tloss += loss.data.item()
            batches_run += 1

            loss.backward()
            optimizer.step()

            # Calculated per instance, not per pair (e.g., x5 examples, individual voting).
            for jdx in range(logits.shape[0]):
                trcm[int(batch_gold[jdx])][int(logits[jdx].argmax(0))] += 1

        tloss /= batches_run
        tr_acc = get_acc(trcm)

        with torch.no_grad():
            cm = np.zeros(shape=(outwidth, outwidth))
            for jdx in range(len(te_inputs)):
                logits = model(te_inputs[jdx])
                cm[int(te_outputs[jdx])][int(logits.argmax(0))] += 1

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


class ConvFFModel(nn.Module):

    # TODO: Tie this to general EarlyFusionModel instead of using directly for prediction (to use with L, V)
    def __init__(self, dv, conv_inputs, conv_inputs_size, num_classes):
        super(ConvFFModel, self).__init__()
        self.conv_inputs = conv_inputs
        self.out = torch.nn.Linear(conv_inputs_size, num_classes).to(dv)
        self.param_list = list(self.out.parameters())
        for m in self.conv_inputs:
            self.param_list.extend(list(m.parameters()))

    def forward(self, ins):
        hcat = self.conv_inputs[0](ins[0])
        for idx in range(1, len(self.conv_inputs)):
            hcat = torch.cat((hcat, self.conv_inputs[idx](ins[idx])), 1)
        logits = self.out(hcat)
        return logits


class ConvToLinearModel(nn.Module):

    def __init__(self, dv, channels, hidden_dim):
        super(ConvToLinearModel, self).__init__()
        out_channels_factor = 2
        kernel = (3, 3)
        stride = (3, 3)
        self.enc1 = torch.nn.Conv2d(channels, out_channels_factor * channels, kernel, stride=1).to(dv)
        self.mp1 = torch.nn.MaxPool2d(kernel, stride=stride).to(dv)
        self.relu1 = torch.nn.ReLU().to(dv)
        self.enc2 = torch.nn.Conv2d(out_channels_factor * channels,
                                    channels * out_channels_factor * out_channels_factor,
                                    kernel).to(dv)
        self.mp2 = torch.nn.MaxPool2d(kernel, stride=stride).to(dv)
        self.relu2 = torch.nn.ReLU().to(dv)
        self.enc3 = torch.nn.Conv2d(channels * out_channels_factor * out_channels_factor,
                                    channels * out_channels_factor * out_channels_factor * out_channels_factor,
                                    kernel).to(dv)
        self.mp3 = torch.nn.MaxPool2d(kernel, stride=stride, padding=1).to(dv)
        self.relu3 = torch.nn.ReLU().to(dv)
        self.conv_out_dim = [1, 2]  # Output dimensions from final max pool.
        self.final_output_channels = channels * out_channels_factor * out_channels_factor * out_channels_factor
        self.fc = torch.nn.Linear(self.conv_out_dim[0] * self.conv_out_dim[1] *
                                  self.final_output_channels,
                                  hidden_dim).to(dv)

    def forward(self, im):

        # print("im\t" + str(im.shape))  # DEBUG
        eim = self.enc1(im)
        # print("enc1\t" + str(eim.shape))  # DEBUG
        eim = self.mp1(eim)
        # print("mp1\t" + str(eim.shape))  # DEBUG
        eim = self.relu1(eim)
        eim = self.enc2(eim)
        # print("enc2\t" + str(eim.shape))  # DEBUG
        eim = self.mp2(eim)
        # print("mp2\t" + str(eim.shape))  # DEBUG
        eim = self.relu2(eim)
        eim = self.enc3(eim)
        # print("enc3\t" + str(eim.shape))  # DEBUG
        eim = self.mp3(eim)
        eim = self.relu3(eim)
        # print("mp3\t" + str(eim.shape))  # DEBUG
        fc_in = eim.view((im.shape[0], self.final_output_channels * self.conv_out_dim[0] * self.conv_out_dim[1]))
        # print("view\t" + str(fc_in.shape))  # DEBUG
        h = self.fc(fc_in)

        return h
