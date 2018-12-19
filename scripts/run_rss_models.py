#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
import pandas as pd
from PIL import Image
from models import *
from sklearn.naive_bayes import GaussianNB
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from utils import *


def main(args, dv):

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.infile + "'...")
    with open(args.infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        lf = all_d["folds"]
        train = lf['train']
        preps = train.keys()
        if args.test:
            test = lf['test']
        else:
            test = lf['dev']  # only ever peak at the dev set.
        # TODO: test['label'] should be drawn from gold ground truth
    print("... done")

    # Read in metadata.
    print("Reading in metadata from '" + args.metadata_infile + "'...")
    with open(args.metadata_infile, 'r') as f:
        d = json.load(f)
        res = d["res"]
        imgs = d["imgs"]
    print("... done")

    # Read in RBDG features.
    print("Reading in RGBD features from '" + args.robot_infile + "'...")
    with open(args.robot_infile, 'r') as f:
        rgbd = json.load(f)
        rgbd_tr = rgbd['train']
        if args.test:
            rgbd_te = rgbd['test']
        else:
            rgbd_te = rgbd['dev']
    print("... done")

    # Select subset of data on which to evaluate.
    print("Selecting evaluation data...")
    models = args.models.split(',')
    available_train = {p: None for p in preps}
    available_test = {p: None for p in preps}
    for p in preps:
        if True or 'rgbd' in models:  # DEBUG
            available_train[p] = [[ix, train[p]["ob1"][ix], train[p]["ob2"][ix]]
                                  for ix in range(len(train[p]["ob1"]))
                                  if str(train[p]["ob1"][ix]) in rgbd_tr and
                                  str(train[p]["ob2"][ix]) in rgbd_tr[str(train[p]["ob1"][ix])]]
            available_test[p] = [[ix, test[p]["ob1"][ix], test[p]["ob2"][ix]]
                                 for ix in range(len(test[p]["ob1"]))
                                 if str(test[p]["ob1"][ix]) in rgbd_te and
                                 str(test[p]["ob2"][ix]) in rgbd_te[str(test[p]["ob1"][ix])]]
            print("... done; %d / %d available training and %d / %d available testing examples with RGBD data for %s" %
                  (len(available_train[p]), len(train[p]["ob1"]), len(available_test[p]), len(test[p]["ob1"]), p))
        else:
            available_train[p] = [[ix, str(train[p]["ob1"][ix]), str(train[p]["ob2"][ix])]
                                  for ix in range(len(train[p]["ob1"]))]
            available_test[p] = [[ix, str(test[p]["ob1"][ix]), str(test[p]["ob2"][ix])]
                                 for ix in range(len(test[p]["ob1"]))]
            print("... done; %d / %d available training and %d / %d available testing examples for %s" %
                  (len(available_train[p]), len(train[p]["ob1"]), len(available_test[p]), len(test[p]["ob1"]), p))
    tr_o = {p: [train[p]["label"][ix] for ix, _, _ in available_train[p]] for p in preps}
    te_o = {p: [test[p]["label"][ix] for ix, _, _ in available_test[p]] for p in preps}

    # Read in ground truth labels.
    if args.gt_infile is not None:
        print("Reading in robo ground truth labels from '" + args.gt_infile + "'...")
        df = pd.read_csv(args.gt_infile)
        gt_labels = {p: {} for p in preps}
        l2c = {"Y": 2, "M": 1, "N": 0}  # Matching 3 class split.
        for idx in df.index:
            k = '(' + ', '.join(df["pair"][idx].split(' + ')) + ')'
            for p in preps:
                if df[p][idx] in l2c:
                    gt_labels[p][k] = l2c[df[p][idx]]
        for p in preps:
            gt_found = 0
            for idx in range(len(te_o[p])):
                key = "(%s, %s)" % (names[test[p]['ob1'][idx]], names[test[p]['ob2'][idx]])
                if key in gt_labels[p]:
                    te_o[p][idx] = gt_labels[p][key]
                    gt_found += 1
            print("... done; %d / %d ground truth test labels found for %s" % (gt_found, len(te_o[p]), p))
            if len(te_o[p]) > gt_found:
                print(
                    "... WARNING: using human annotations for remaining labels (TODO: annotate remainder of data)")

    bs = []
    rs = []

    # Majority class baseline.
    if 'mc' in models:
        print("Running majority class baseline...")
        bs.append("Majority Class")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_majority_class(tr_o[p], te_o[p])
        print("... done")

    # Majority class | object ids baseline.
    if 'nb' in models:
        print("Running Naive Bayes baseline...")
        bs.append("Naive Bayes Obj One Hots")
        rs.append({})
        fs = [range(len(names)), range(len(names))]  # Two one-hot vectors of which object name was seen.
        for p in preps:
            tr_f = [[oidx, ojdx] for _, oidx, ojdx in available_train[p]]
            te_f = [[oidx, ojdx] for _, oidx, ojdx in available_test[p]]
            rs[-1][p] = run_cat_naive_bayes(fs, tr_f, tr_o[p], te_f, te_o[p], verbose=args.verbose)
        print("... done")

    # Read in hyperparameters for feed forward networks or set defaults.
    ff_epochs = 10 if args.ff_epochs is None else args.ff_epochs
    if args.hyperparam_infile is not None:
        with open(args.hyperparam_infile, 'r') as f:
            d = json.load(f)
            ff_layers = d['layers']
            ff_width_decay = d['width_decay']
            ff_dropout = d['dropout']
            ff_lr = d['lr']
            ff_opt = d['opt']
            print("Loaded ff hyperparams: " + str(d))
    else:
        ff_layers = 0
        ff_width_decay = 2
        ff_dropout = 0
        ff_lr = 0.001
        ff_opt = 'adagrad'
    ff_random_restarts = None if args.ff_random_restarts is None else \
        [int(s) for s in args.ff_random_restarts.split(',')]

    if 'rgbd' in models:
        print("Running RGBD models")
        bs.append("RGBD")
        rs.append({})
        for p in preps:

            # Prepare input to model.
            tr_f = [rgbd_tr[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_train[p]]
            te_f = [rgbd_te[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_test[p]]

            # Convert structured data into tensor inputs.
            tr_inputs = []
            te_inputs = []
            classes = set()
            for model_in, orig_in, orig_out in [[tr_inputs, tr_f, tr_o[p]],
                                                [te_inputs, te_f, te_o[p]]]:
                for idx in range(len(orig_in)):
                    # Convert RGB and D numpy arrays to tensors and add batch dimension at axis 0.
                    model_in.append([torch.tensor(orig_in[idx][0], dtype=torch.float).unsqueeze(0).to(dv),
                                       torch.tensor(orig_in[idx][1], dtype=torch.float).unsqueeze(0).to(dv)])
                    if orig_out[idx] not in classes:
                        classes.add(orig_out[idx])
            tr_outputs = torch.tensor(tr_o[p], dtype=torch.long).view(len(tr_o[p]), 1).to(dv)
            batch_size = len(tr_inputs)

            # Instantiate convolutional RGB and Depth models, then tie them together with a conv FF model.
            hidden_dim = 100  # TODO: hyperparam
            rgb_conv_model = ConvToLinearModel(dv, 3, hidden_dim).to(dv)
            depth_conv_model = ConvToLinearModel(dv, 1, hidden_dim).to(dv)
            model = ConvFFModel(dv, [rgb_conv_model, depth_conv_model], hidden_dim * 2, len(classes)).to(dv)

            # Instantiate loss and optimizer.
            loss_function = nn.CrossEntropyLoss()
            if ff_opt == 'sgd':
                optimizer = optim.SGD(model.param_list, lr=ff_lr)
            elif ff_opt == 'adagrad':
                optimizer = optim.Adagrad(model.param_list, lr=ff_lr)
            elif ff_opt == 'adam':
                optimizer = optim.Adam(model.param_list, lr=ff_lr)
            elif ff_opt == 'rmsprop':
                optimizer = optim.RMSprop(model.param_list, lr=ff_lr)
            else:
                raise ValueError('Unrecognized opt specification "' + ff_opt + '".')

            # Train model.
            if ff_random_restarts is None:
                seeds = [None]
            else:
                seeds = ff_random_restarts
                rs[-1][p] = []
            for seed in seeds:
                if seed is not None:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                # Run training for specified number of epochs.
                best_acc = best_cm = tr_acc_at_best = trcm_at_best = None
                idxs = list(range(len(tr_inputs)))
                np.random.shuffle(idxs)
                tr_inputs = [tr_inputs[idx] for idx in idxs]
                tr_outputs = tr_outputs[idxs, :]
                idx = 0
                result = None
                for epoch in range(ff_epochs):
                    tloss = 0
                    c = 0
                    trcm = np.zeros(shape=(len(classes), len(classes)))  # note: calculates train acc on curr batch only
                    while c < batch_size:
                        model.zero_grad()
                        logits = model(tr_inputs[idx])
                        loss = loss_function(logits, tr_outputs[idx])
                        tloss += loss.data.item()
                        trcm[tr_o[p][idx]][logits.max(0)[1]] += 1

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
                            cm[te_o[p][jdx]][logits.max(0)[1]] += 1

                        acc = get_acc(cm)
                        if best_acc is None or acc > best_acc:
                            best_acc = acc
                            best_cm = cm
                            tr_acc_at_best = tr_acc
                            trcm_at_best = trcm

                    result = best_acc, best_cm, tr_acc_at_best, trcm_at_best

                if seed is None:
                    rs[-1][p] = result
                else:
                    rs[-1][p].append(result)
        print("... done")

    # Prep language dictionary.
    print("Preparing infrastructure to include GloVe info...")
    word_to_i = {}
    word_to_i_all = {}
    i_to_word = {}
    maxlen = {}
    tr_enc_exps = {}
    te_enc_exps = {}
    for p in preps:
        tr_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_train[p]]
        te_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_test[p]]
        word_to_i[p], i_to_word[p], maxlen[p], tr_enc_exps[p], te_enc_exps[p] = make_lang_structures(tr_f, te_f)
        word_to_i_all[p], _, _, _, _ = make_lang_structures(tr_f, te_f, inc_test=True)
    print("... done")

    # Language naive bayes (bag of words)
    if 'bownb' in models:
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
                tr_outputs.append(tr_o[p][idx])
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
            rs[-1][p] = run_cat_naive_bayes(fs, tr_inputs, tr_o[p], te_inputs, te_o[p],
                                            smooth=1, verbose=args.verbose)
        print ("... done")

    # Average GLoVe embeddings concatenated and used to predict class.
    print("Preparing infrastructure to run GLoVe-based feed forward model...")
    ws = set()
    for p in preps:
        ws.update(word_to_i_all[p].keys())
    g, missing = get_glove_vectors(args.glove_infile, ws)
    emb_dim_l = len(g[list(g.keys())[0]])
    print("... done; missing " + str(missing) + " vectors out of " + str(len(ws)))

    if 'glove' in models:
        print("Running GLoVe models")
        bs.append("GLoVe FF")
        rs.append({})
        for p in preps:
            tr_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_train[p]]
            te_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_test[p]]
            layers = None if ff_layers == 0 else [int(emb_dim_l / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [g], [tr_f], tr_o[p], [te_f], te_o[p],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [g], [tr_f], tr_o[p], [te_f], te_o[p],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose))

    # Average BoW embeddings use to predict class.
    if 'bowff' in models:
        print("Preparing infrastructure to run BoW-based feed forward model...")
        ws = set()
        for p in preps:
            ws.update(word_to_i_all[p].keys())
        wv = get_bow_vectors(ws)
        emb_dim = len(wv)
        print("... done")

        print("Running BoW FF models")
        bs.append("BoW FF")
        rs.append({})
        for p in preps:
            tr_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_train[p]]
            te_f = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_test[p]]
            layers = None if ff_layers == 0 else [int(emb_dim / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [wv], [tr_f], tr_o[p], [te_f], te_o[p],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [wv], [tr_f], tr_o[p], [te_f], te_o[p],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose))

    # Average BoW embeddings use to predict class.
    # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # bc of imagenet pretraining
    print("Preparing infrastructure to run ResNet-based feed forward model...")
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
            im = normalize(im)
            im = torch.unsqueeze(im, 0)
            idx_to_v[idx] = plm(im).detach().data.numpy().flatten()
            with open(ffn, 'w') as f:
                f.write(' '.join([str(i) for i in idx_to_v[idx]]))
    emb_dim_v = len(idx_to_v[0])
    print("... done")

    if 'resnet' in models:
        print("Running ResNet FF models")
        bs.append("ResNet FF")
        rs.append({})
        for p in preps:
            # FF expects a sequences of indices to be looked up in the vectors dictionary and averaged, so here
            # we just make each sequence [[oidx]] for the object idx to be looked up in the dictionary of pre-computed
            # vectors. It doesn't need to be averaged with anything. The double wrapping is because we expect first
            # a list of referring expressions, then a list of words inside each expression.
            # TODO: This will look a little less weird when we have multiple viewpoints per image, in which case each
            # TODO: object will have multiple image 'tokens' whose embedding is to be looked up as part of the BoW
            # TODO: that will come to represent the image input.
            tr_f = [[[[oidx]], [[ojdx]]] for _, oidx, ojdx in available_train[p]]
            te_f = [[[[oidx]], [[ojdx]]] for _, oidx, ojdx in available_test[p]]
            layers = None if ff_layers == 0 else [int(emb_dim_v / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [idx_to_v], [tr_f], tr_o[p], [te_f], te_o[p],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [idx_to_v], [tr_f], tr_o[p], [te_f], te_o[p],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose))

    if 'glove' in models and 'resnet' in models:
        print("Running Glove+ResNet (Shrink) models")
        bs.append("GLoVe+ResNet (Shrink)")
        rs.append({})
        for p in preps:

            # Prepare input to model.
            tr_f_l = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_train[p]]
            te_f_l = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_test[p]]
            tr_f_v = [[[[oidx]], [[ojdx]]] for _, oidx, ojdx in available_train[p]]
            te_f_v = [[[[oidx]], [[ojdx]]] for _, oidx, ojdx in available_test[p]]
            wv = [g, idx_to_v]

            # Convert structured data into tensor inputs.
            tr_inputs = []
            te_inputs = []
            classes = set()
            for model_in, orig_in, orig_out in [[tr_inputs, [tr_f_l, tr_f_v], tr_o[p]],
                                                [te_inputs, [te_f_l, te_f_v], te_o[p]]]:
                for idx in range(len(orig_in[0])):
                    all_in = []
                    for midx in range(len(orig_in)):
                        v = wv[midx]
                        modality = orig_in[midx]
                        ob1_ws = [w for ws in modality[idx][0] for w in ws]
                        avg_ob1_v = np.sum([v[w] for w in ob1_ws], axis=0) / len(ob1_ws)
                        ob2_ws = [w for ws in modality[idx][1] for w in ws]
                        avg_ob2_v = np.sum([v[w] for w in ob2_ws], axis=0) / len(ob2_ws)
                        incat = torch.tensor(np.concatenate((avg_ob1_v, avg_ob2_v)), dtype=torch.float).to(dv)
                        all_in.append(incat)
                    model_in.append(all_in)
                    if orig_out[idx] not in classes:
                        classes.add(orig_out[idx])
            tr_outputs = torch.tensor(tr_o[p], dtype=torch.long).view(len(tr_o[p]), 1).to(dv)
            batch_size = len(tr_inputs)

            # Instantiate model and optimizer.
            emb_dim = 2 * min(emb_dim_l, emb_dim_v)
            hidden_layer_widths = [int(emb_dim / np.power(ff_width_decay, lidx) + 0.5) for lidx in range(ff_layers)]
            model = EarlyFusionFFModel(dv, [emb_dim_l * 2, emb_dim_v * 2],  # input is concatenated across two objects
                                       True, hidden_layer_widths, ff_dropout, len(classes)).to(dv)
            loss_function = nn.CrossEntropyLoss()
            if ff_opt == 'sgd':
                optimizer = optim.SGD(model.param_list, lr=ff_lr)
            elif ff_opt == 'adagrad':
                optimizer = optim.Adagrad(model.param_list, lr=ff_lr)
            elif ff_opt == 'adam':
                optimizer = optim.Adam(model.param_list, lr=ff_lr)
            elif ff_opt == 'rmsprop':
                optimizer = optim.RMSprop(model.param_list, lr=ff_lr)
            else:
                raise ValueError('Unrecognized opt specification "' + ff_opt + '".')

            # Train model.
            if ff_random_restarts is None:
                seeds = [None]
            else:
                seeds = ff_random_restarts
                rs[-1][p] = []
            for seed in seeds:
                if seed is not None:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                # Run training for specified number of epochs.
                best_acc = best_cm = tr_acc_at_best = trcm_at_best = None
                idxs = list(range(len(tr_inputs)))
                np.random.shuffle(idxs)
                tr_inputs = [tr_inputs[idx] for idx in idxs]
                tr_outputs = tr_outputs[idxs, :]
                idx = 0
                result = None
                for epoch in range(ff_epochs):
                    tloss = 0
                    c = 0
                    trcm = np.zeros(shape=(len(classes), len(classes)))  # note: calculates train acc on curr batch only
                    while c < batch_size:
                        model.zero_grad()
                        logits = model(tr_inputs[idx])
                        loss = loss_function(logits.view(1, len(logits)), tr_outputs[idx])
                        tloss += loss.data.item()
                        trcm[tr_o[p][idx]][logits.max(0)[1]] += 1

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
                            cm[te_o[p][jdx]][logits.max(0)[1]] += 1

                        acc = get_acc(cm)
                        if best_acc is None or acc > best_acc:
                            best_acc = acc
                            best_cm = cm
                            tr_acc_at_best = tr_acc
                            trcm_at_best = trcm

                    result = best_acc, best_cm, tr_acc_at_best, trcm_at_best

                if seed is None:
                    rs[-1][p] = result
                else:
                    rs[-1][p].append(result)

    if ff_random_restarts is None:
        # Show results.
        print("Results:")
        for idx in range(len(bs)):
            print(" " + bs[idx] + ":")
            for p in preps:
                print("  " + p + ":\tacc %0.3f" % rs[idx][p][0] + "\t(train: %0.3f" % rs[idx][p][2] + ")")
                print("  \tf1  %0.3f" % get_f1(rs[idx][p][1]) + "\t(train: %0.3f" % get_f1(rs[idx][p][3]) + ")")
                print('\t(CM\t' + '\n\t\t'.join(['\t'.join([str(int(ct)) for ct in rs[idx][p][1][i]])
                                                 for i in range(len(rs[idx][p][1]))]) + ")")

        # Write val accuracy results.
        if args.perf_outfile is not None:
            print("Writing results to '" + args.perf_outfile + "'...")
            with open(args.perf_outfile, 'w') as f:
                json.dump([{p: [rs[idx][p][0], get_f1(rs[idx][p][1])] for p in preps} for idx in range(len(rs))], f)
            print("... done")

    else:
        print("Results:")
        for idx in range(len(bs)):
            print(" " + bs[idx] + ":")
            for p in preps:
                avg_acc = np.average([rs[idx][p][jdx][0] for jdx in range(len(ff_random_restarts))])
                avg_tr_acc = np.average([rs[idx][p][jdx][2] for jdx in range(len(ff_random_restarts))])
                avg_f1 = np.average([get_f1(rs[idx][p][jdx][1]) for jdx in range(len(ff_random_restarts))])
                avg_tr_f1 = np.average([get_f1(rs[idx][p][jdx][3]) for jdx in range(len(ff_random_restarts))])
                print("  " + p + ":\tacc %0.3f" % avg_acc + "\t(train: %0.3f" % avg_tr_acc + ")")
                print("  \tf1  %0.3f" % avg_f1 + "\t(train: %0.3f" % avg_tr_f1 + ")")

        # Write out results for all seeds so that a downstream script can process them for stat sig.
        if args.perf_outfile is not None:
            print("Writing results to '" + args.perf_outfile + "'...")
            with open(args.perf_outfile, 'w') as f:
                json.dump([{p: {"acc": [rs[idx][p][jdx][0] for jdx in range(len(ff_random_restarts))],
                                "tr_acc": [rs[idx][p][jdx][2] for jdx in range(len(ff_random_restarts))],
                                "f1": [get_f1(rs[idx][p][jdx][1]) for jdx in range(len(ff_random_restarts))],
                                "tr_f1": [get_f1(rs[idx][p][jdx][3]) for jdx in range(len(ff_random_restarts))]}
                            for p in preps} for idx in range(len(rs))], f)
            print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    parser.add_argument('--glove_infile', type=str, required=True,
                        help="input glove vector text file")
    parser.add_argument('--robot_infile', type=str, required=True,
                        help="input robot feature file")
    parser.add_argument('--models', type=str, required=True,
                        help="models to run (mc, nb, bownb, bowff, glove, resnet, rgbd)")
    parser.add_argument('--verbose', type=int, required=False, default=0,
                        help="verbosity level")
    parser.add_argument('--hyperparam_infile', type=str, required=False,
                        help="input json for model hyperparameters")
    parser.add_argument('--perf_outfile', type=str, required=False,
                        help="output json for model performance")
    parser.add_argument('--ff_epochs', type=int, required=False,
                        help="override default number of epochs")
    parser.add_argument('--ff_random_restarts', type=str, required=False,
                        help="comma-separated list of random seeds to use; presents avg + stddev data instead of cms")
    parser.add_argument('--test', type=int, required=False,
                        help="if set to 1, evaluates on the test set; NOT FOR TUNING")
    parser.add_argument('--gt_infile', type=str, required=False,
                        help="input csv of ground truth affordance labels; if provided, overrides dev/test labels")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args(), device)
