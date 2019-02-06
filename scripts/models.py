#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Models used by other scripts.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
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
    return get_acc(cm), cm, get_acc(trcm), trcm, 0, 1


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
    return get_acc(cm), cm, get_acc(trcm), trcm, None


# Run a GloVe-based model (either perceptron or FF network with relu activations)
# outdir - directory to save best-performing models during training
# model_desc - string description of model for filename
# tr_inputs - feature vector inputs
# tr_outputs - training labels
# te_inputs - same as tr_f but for testing
# te_outputs - testing labels
# verbose - whether to print epoch-wise progress
def run_ff_model(dv, outdir, model_desc,
                 tr_inputs, tr_outputs, te_inputs, te_outputs,
                 inwidth, hidden_dim, outwidth,  # single hidden layer of specified size, projecting to outwidth classes
                 num_modalities,  # If greater than 1, uses EarlyFusionModel to implement, else simple single hidden layer.
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
    if num_modalities == 1:
        # TODO: these dropout layers might be in a stupid place.
        lr = [nn.Linear(inwidth, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout), 
              nn.Linear(hidden_dim, outwidth)]
        model = nn.Sequential(*lr).to(dv)
        mparams = model.parameters()
    else:
        widths = [len(tr_inputs[midx][0]) for midx in range(num_modalities)]
        model = EarlyFusionFFModel(dv, widths, hidden_dim, dropout, outwidth)
        mparams = model.param_list
    if verbose:
        print("FF: ... done")

    # Train: train a neural model for a fixed number of epochs.
    if verbose:
        print("FF: training on " + str(len(tr_inputs)) + " inputs with batch size " + str(batch_size) +
              " for " + str(epochs) + " epochs...")
    loss_function = nn.CrossEntropyLoss()

    if opt == 'sgd':
        optimizer = optim.SGD(mparams, lr=learning_rate)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(mparams, lr=learning_rate)
    elif opt == 'adam':
        optimizer = optim.Adam(mparams, lr=learning_rate)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(mparams, lr=learning_rate)
    else:
        raise ValueError('Unrecognized opt specification "' + opt + '".')

    best_acc = best_cm = tr_acc_at_best = trcm_at_best = tloss_at_best = t_epochs = None
    if num_modalities == 1:
        idxs = list(range(len(tr_inputs)))
        np.random.shuffle(idxs)
        tr_inputs = [tr_inputs[idx] for idx in idxs]
        num_tr = len(tr_inputs)
        num_te = len(te_inputs)
    else:
        idxs = list(range(len(tr_inputs[0])))
        np.random.shuffle(idxs)
        for midx in range(num_modalities):
            tr_inputs[midx] = [tr_inputs[midx][idx] for idx in idxs]
            num_tr = len(tr_inputs[midx])
            num_te = len(te_inputs[midx])
    tro = tr_outputs[idxs, :]
    result = None
    last_saved_fn = None
    for epoch in range(epochs):
        tloss = 0
        trcm = np.zeros(shape=(outwidth, outwidth))
        tidx = 0
        batches_run = 0
        while tidx < num_tr:
            model.zero_grad()
            if num_modalities == 1:
                batch_in = torch.zeros((batch_size, tr_inputs[0].shape[0])).to(dv)
            else:
                batch_in = []
                for midx in range(num_modalities):
                    batch_in.append(torch.zeros((batch_size, tr_inputs[midx][0].shape[0])).to(dv))
            batch_gold = torch.zeros(batch_size).to(dv)
            for bidx in range(batch_size):
                if num_modalities == 1:
                    batch_in[bidx, :] = tr_inputs[tidx]
                else:
                    for midx in range(num_modalities):
                        batch_in[midx][bidx, :] = tr_inputs[midx][tidx]
                batch_gold[bidx] = tro[tidx][0]

                tidx += 1
                if tidx == num_tr:
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
            for jdx in range(num_te):
                if num_modalities == 1:
                    tein = te_inputs[jdx]
                    logit_am_dim = 0
                else:
                    tein = []
                    for midx in range(num_modalities):
                        tein.append(torch.zeros((1, te_inputs[midx][0].shape[0])).to(dv))
                    for midx in range(num_modalities):
                        tein[midx][0, :] = te_inputs[midx][jdx]
                    logit_am_dim = 1
                logits = model(tein)
                cm[int(te_outputs[jdx])][int(logits.argmax(logit_am_dim))] += 1

            acc = get_acc(cm)
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_cm = cm
                tr_acc_at_best = tr_acc
                trcm_at_best = trcm
                tloss_at_best = tloss
                t_epochs = epoch

                # Save best-performing model at this seed
                # (overwritten each time better perf happens across epochs)
                fn = os.path.join(outdir, "%s_acc-%.3f.pt" % (model_desc, best_acc))
                torch.save(model.state_dict(), fn)
                if last_saved_fn is not None:  # remove previous save
                    os.system("rm %s" % last_saved_fn)
                last_saved_fn = fn

        if verbose:
            print("... epoch " + str(epoch) + " train loss " + str(tloss) + "; train accuracy " + str(tr_acc) +
                  "; test accuracy " + str(acc))
    if verbose:
        print("FF: ... done")

    return best_acc, best_cm, tr_acc_at_best, trcm_at_best, tloss_at_best, t_epochs


# Based on:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class EarlyFusionFFModel(nn.Module):

    # adds FC shrinking layers on all but smallest modality to shrink them to smallest before fusion
    # modality_widths - list of input widths, e.g., [300, 1000] means first input width 300, second 1000
    # hidden_layer_width - hidden layer width after fusion before prediction
    # dropout - the dropout rate to add before hidden layer connections
    # num_classes - the number of classes to predict over in the output layer.
    def __init__(self, dv, modality_widths, hidden_layer_width, dropout, num_classes):
        super(EarlyFusionFFModel, self).__init__()
        self.param_list = []
        self.num_modalities = len(modality_widths)

        # Add shrinking layers to bring larger modality inputs to hidden layer width.
        self.shrink_layers = {sidx: nn.Linear(modality_widths[sidx], hidden_layer_width).to(dv)
                              for sidx in range(len(modality_widths)) if modality_widths[sidx] > hidden_layer_width}
        input_cat_width = hidden_layer_width * len(modality_widths)

        for sidx in self.shrink_layers:
            self.param_list.extend(list(self.shrink_layers[sidx].parameters()))

        # In forward pass, inputs are then concatenated and fed to a number (possibly zero) of hidden layers.
        self.hidden_layers = [nn.Linear(input_cat_width, hidden_layer_width).to(dv),
                              torch.nn.ReLU().to(dv), torch.nn.Dropout(dropout).to(dv),
                              nn.Linear(hidden_layer_width, num_classes).to(dv)]
        for hl in self.hidden_layers:
            self.param_list.extend(list(hl.parameters()))

    # inputs - a list of tensor inputs
    def forward(self, inputs):

        # Shrink inputs.
        s_inputs = [self.shrink_layers[sidx](inputs[sidx]) if sidx in self.shrink_layers else inputs[sidx]
                    for sidx in range(self.num_modalities)]

        # Concatenate inputs.
        # cat_inputs = torch.cat(s_inputs, 1)  # concatenate along the feature dimension (batch is dim 0)

        # Feed concatenation through hidden layers and predict classes.
        # ff_inputs = [cat_inputs]
        # for hl in self.hidden_layers:
        #     ff_inputs.append(hl(ff_inputs[-1]))
        # logits = ff_inputs[-1]

        # Hidden layer as a simple dot between modalities demonstrably improves performance on Dev.
        logits = self.hidden_layers[-1](s_inputs[0] * s_inputs[1])

        return logits

# DEBUG - stats before hidden dot (Glove+ResNet)
# Results:
#  Majority Class:
#   in:   acc 0.660+/-0.000       (train: 0.610+/-0.000)
#         f1  0.737+/-0.000       (train: 0.678+/-0.000)
#   on:   acc 0.460+/-0.000       (train: 0.386+/-0.000)
#         f1  0.533+/-0.000       (train: 0.474+/-0.000)
#  GloVe+ResNet FF:
#   in:   acc 0.661+/-0.003       (train: 0.582+/-0.038)
#         f1  0.739+/-0.003       (train: 0.649+/-0.040)
#   on:   acc 0.527+/-0.024       (train: 0.556+/-0.084)
#         f1  0.568+/-0.029       (train: 0.580+/-0.098)

# 300 dim hidden (using)
# GloVe+ResNet FF:
#   in:   acc 0.682+/-0.015       (train: 0.732+/-0.079)
#         f1  0.746+/-0.017       (train: 0.807+/-0.079)
#   on:   acc 0.541+/-0.012       (train: 0.711+/-0.059)
#         f1  0.594+/-0.016       (train: 0.750+/-0.055)

# 100 dim hidden (shrink both L and V to h, then dot to get combined h)
# GloVe+ResNet FF:
#   in:   acc 0.676+/-0.022       (train: 0.628+/-0.099)
#         f1  0.748+/-0.027       (train: 0.702+/-0.102)
#   on:   acc 0.536+/-0.022       (train: 0.609+/-0.051)
#         f1  0.604+/-0.026       (train: 0.647+/-0.065)



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
        self.enc1 = torch.nn.Conv2d(channels,
                                    channels * out_channels_factor,
                                    kernel, stride=1).to(dv)
        self.mp1 = torch.nn.MaxPool2d(kernel, stride=stride).to(dv)
        self.relu1 = torch.nn.ReLU().to(dv)
        self.enc2 = torch.nn.Conv2d(channels * out_channels_factor,
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
        # TODO: is this im.shape[0] necessary? is that... the batch? seems wonky.
        fc_in = eim.view((im.shape[0], self.final_output_channels * self.conv_out_dim[0] * self.conv_out_dim[1]))
        # print("view\t" + str(fc_in.shape))  # DEBUG
        h = self.fc(fc_in)

        return h
