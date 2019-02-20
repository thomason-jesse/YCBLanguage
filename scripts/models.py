#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Models used by other scripts.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
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
    return get_acc(cm), cm, get_acc(trcm), trcm, 0, 1, {}


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
                 dropout=0, learning_rate=0.001, opt='sgd', activation='relu',
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
        shrink_model = ShrinkingFFModel(dv, inwidth, hidden_dim, hidden_dim, dropout, activation).to(dv)
        fusion_model = None
        model = PredModel(dv, shrink_model, hidden_dim, outwidth).to(dv)
    elif num_modalities == 2:
        fusion_model = ShrinkingFusionFFModel(dv, len(tr_inputs[0][0]), len(tr_inputs[1][0]),
                                              hidden_dim, dropout, activation).to(dv)
        model = PredModel(dv, fusion_model, hidden_dim, outwidth).to(dv)
    else:
        sys.exit("ERROR: unsupported number of modalities for run_ff_model %d" % num_modalities)
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

    best_acc = best_cm = tr_acc_at_best = trcm_at_best = tloss_at_best = t_epochs = best_loss = None
    if num_modalities == 1:
        idxs = list(range(len(tr_inputs)))
        np.random.shuffle(idxs)
        tr_inputs = [tr_inputs[idx] for idx in idxs]
        num_tr = len(tr_inputs)
        num_te = len(te_inputs)
    else:
        idxs = list(range(len(tr_inputs[0])))
        np.random.shuffle(idxs)
        num_tr = len(tr_inputs[0])
        num_te = len(te_inputs[0])
        for midx in range(num_modalities):
            tr_inputs[midx] = [tr_inputs[midx][idx] for idx in idxs]
    tro = tr_outputs[idxs, :]
    last_saved_acc_fn = None
    last_saved_loss_fn = None
    for epoch in range(epochs):
        tloss = 0
        trcm = np.zeros(shape=(outwidth, outwidth))
        tidx = 0
        batches_run = 0
        while tidx < num_tr:
            model.train()
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
            model.eval()
            te_loss = 0
            cm = np.zeros(shape=(outwidth, outwidth))
            for jdx in range(num_te):
                if num_modalities == 1:
                    tein = te_inputs[jdx]
                else:
                    tein = []
                    for midx in range(num_modalities):
                        tein.append(torch.zeros((1, te_inputs[midx][0].shape[0])).to(dv))
                    for midx in range(num_modalities):
                        tein[midx][0, :] = te_inputs[midx][jdx]
                logits = model(tein)
                gold_logits = torch.tensor(te_outputs[jdx]).to(dv)
                if num_modalities == 1:
                    logits = logits.unsqueeze(0)
                loss = loss_function(logits, gold_logits.long())
                te_loss += loss.data.item()
                cm[int(te_outputs[jdx])][int(logits.argmax(1))] += 1
            te_loss /= num_te

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
                fn = os.path.join(outdir, "%s_epoch-%d_acc-%.3f.pt" % (model_desc, epoch, best_acc))
                torch.save(model.state_dict(), fn)
                if fusion_model is not None:
                    torch.save(fusion_model.state_dict(), fn+".fm")
                if last_saved_acc_fn is not None:  # remove previous save
                    os.system("rm %s" % last_saved_acc_fn)
                    if fusion_model is not None:
                        os.system("rm %s.fm" % last_saved_acc_fn)
                last_saved_acc_fn = fn

            if best_loss is None or te_loss < best_loss:
                best_loss = te_loss
                fn = os.path.join(outdir, "%s_epoch-%d_loss-%.3f.pt" % (model_desc, epoch, best_loss))
                torch.save(model.state_dict(), fn)
                if fusion_model is not None:
                    torch.save(fusion_model.state_dict(), fn + ".fm")
                if last_saved_loss_fn is not None:  # remove previous save
                    os.system("rm %s" % last_saved_loss_fn)
                    if fusion_model is not None:
                        os.system("rm %s.fm" % last_saved_loss_fn)
                last_saved_loss_fn = fn

        if verbose:
            print("... epoch " + str(epoch) + " train loss " + str(tloss) + "; train accuracy " + str(tr_acc) +
                  "; test accuracy " + str(acc))
    if verbose:
        print("FF: ... done")

    # TODO: calculate and store predictions for analysis.
    return best_acc, best_cm, tr_acc_at_best, trcm_at_best, tloss_at_best, t_epochs, {}


# Predicts output layer from two model outputs.
# Assumes two fusion models output same size representations.
# Concatenates outputs and learns fc layer to output classes.
class MultiPredModel(nn.Module):

    def __init__(self, dv, model_1, model_2, model_output_size, output_size):
        super(MultiPredModel, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.out = torch.nn.Linear(model_output_size * 2, output_size).to(dv)

    def forward(self, ins):
        h1 = self.model_1(ins[0])
        h2 = self.model_2(ins[1])
        # TODO: could try dot product here.
        # TODO: could try fully connected layer here.
        logits = self.out(torch.cat((h1, h2), dim=1))  # concatenate along feature, not batch, dim
        return logits


# Predicts output layer from single model output.
class PredModel(nn.Module):

    def __init__(self, dv, model, model_output_size, output_size):
        super(PredModel, self).__init__()
        self.model = model
        self.out = torch.nn.Linear(model_output_size, output_size).to(dv)

    def forward(self, ins):
        h = self.model(ins)
        return self.out(h)


# Single linear layer shrinks the input to the target size then computes a hidden layer.
class ShrinkingFFModel(nn.Module):

    def __init__(self, dv, input_width, shrink_width, output_width, dropout, activation):
        super(ShrinkingFFModel, self).__init__()
        # TODO: activation on shrinking layer or leave linear? Dropout?
        self.shrink_layer = nn.Linear(input_width, shrink_width).to(dv)
        if activation == 'relu':
            self.activation = torch.nn.ReLU().to(dv)
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh().to(dv)
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid().to(dv)
        else:
            sys.exit("unrecognized activation '%s'" % activation)
        self.dropout = torch.nn.Dropout(dropout).to(dv)
        self.hidden_layer = nn.Linear(shrink_width, output_width).to(dv)

    def forward(self, inputs):
        h = self.shrink_layer(inputs)
        h = self.activation(h)
        # TODO: this dropout layer might be in a stupid place.
        h = self.dropout(h)
        h = self.hidden_layer(h)
        return h


# Takes in two input modalities, shrinks them to the same target size w a linear layer, and dots them.
class ShrinkingFusionFFModel(nn.Module):

    # in1_width - first modality input width
    # in2_width - second modality input width
    # out_width - output width to shrink modalities to (will be dotted together)
    def __init__(self, dv, in1_width, in2_width, out_width, dropout, activation):
        super(ShrinkingFusionFFModel, self).__init__()
        self.shrinking_1 = ShrinkingFFModel(dv, in1_width, out_width, out_width, dropout, activation)
        self.shrinking_2 = ShrinkingFFModel(dv, in2_width, out_width, out_width, dropout, activation)

    # inputs - a list of tensor inputs.
    # outputs - dot product of shrunken, equal sized input modality representations.
    def forward(self, inputs):

        # Shrink inputs.
        shrunk_1 = self.shrinking_1(inputs[0])
        shrunk_2 = self.shrinking_2(inputs[1])
        h = shrunk_1 * shrunk_2

        return h


# Takes in two input modalities, shrinks them to the same target size with convolution layers, and dots them.
class ConvFusionFFModel(nn.Module):

    # adds FC shrinking layers on all but smallest modality to shrink them to smallest before fusion
    # in1_channels - first modality input channels
    # in2_channels - second modality input channels
    # out_width - output width to shrink modalities to (will be dotted together)
    def __init__(self, dv, in1_channels, in2_channels, out_width, activation):
        super(ConvFusionFFModel, self).__init__()
        self.shrinking_1 = ConvToLinearModel(dv, in1_channels, out_width, activation)
        self.shrinking_2 = ConvToLinearModel(dv, in2_channels, out_width, activation)

    # inputs - a list of tensor inputs.
    # outputs - dot product of shrunken, equal sized input modality representations.
    def forward(self, inputs):

        # Shrink inputs.
        shrunk_1 = self.shrinking_1(inputs[0])
        shrunk_2 = self.shrinking_2(inputs[1])
        h = shrunk_1 * shrunk_2

        return h


# Serires of 3 convolutions + max pools followed by a fully connected layer to shrink
# a multi-channel, 2d input into a single linear representation.
class ConvToLinearModel(nn.Module):

    def __init__(self, dv, channels, hidden_dim, activation):
        super(ConvToLinearModel, self).__init__()
        out_channels_factor = 2
        kernel = (3, 3)
        stride = (3, 3)

        if activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'tanh':
            act = torch.nn.Tanh()
        elif activation == 'sigmoid':
            act = torch.nn.Sigmoid()
        else:
            sys.exit("unrecognized activation '%s'" % activation)

        layers = [
            nn.Conv2d(channels, channels * out_channels_factor,
                      kernel, stride=1),
            nn.MaxPool2d(kernel, stride=stride),
            act,
            nn.Conv2d(channels * out_channels_factor,
                      channels * out_channels_factor **2, kernel),
            nn.MaxPool2d(kernel, stride=stride),
            act,
            nn.Conv2d(channels * out_channels_factor ** 2,
                      channels * out_channels_factor ** 3, kernel),
            nn.MaxPool2d(kernel, stride=stride, padding=1),
            act,
        ]
        self.conv_stack = nn.Sequential(*layers)

        conv_out_dim = [1, 2]  # Output dimensions from final max pool.
        final_output_channels = channels * out_channels_factor ** 3
        self.fc = torch.nn.Linear(conv_out_dim[0] * conv_out_dim[1] *
                                  final_output_channels, hidden_dim)

    def forward(self, im):
        eim = self.conv_stack(im)
        return self.fc(eim.view((im.shape[0], -1)))
