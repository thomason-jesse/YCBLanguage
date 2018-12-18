#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
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

    models = args.models.split(',')
    bs = []
    rs = []

    # Majority class baseline.
    if 'mc' in models:
        print("Running majority class baseline...")
        bs.append("Majority Class")
        rs.append({})
        for p in preps:
            mc_tr = train[p]["label"]
            mc_te = train[p]["label"]
            rs[-1][p] = run_majority_class(mc_tr, mc_te)
        print("... done")

    # Majority class | object ids baseline.
    if 'nb' in models:
        print("Running Naive Bayes baseline...")
        bs.append("Naive Bayes Obj One Hots")
        rs.append({})
        fs = [range(len(names)), range(len(names))]  # Two one-hot vectors of which object name was seen.
        for p in preps:
            tr_f = [[train[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[test[p][s][idx] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            rs[-1][p] = run_cat_naive_bayes(fs, tr_f, train[p]["label"], te_f, test[p]["label"], verbose=args.verbose)
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

            available_train = [[ix, str(train[p]["ob1"][ix]), str(train[p]["ob2"][ix])]
                               for ix in range(len(train[p]["ob1"]))
                               if str(train[p]["ob1"][ix]) in rgbd_tr and
                               str(train[p]["ob2"][ix]) in rgbd_tr[str(train[p]["ob1"][ix])]]
            available_test = [[ix, str(test[p]["ob1"][ix]), str(test[p]["ob2"][ix])]
                              for ix in range(len(test[p]["ob1"]))
                              if str(test[p]["ob1"][ix]) in rgbd_te and
                              str(test[p]["ob2"][ix]) in rgbd_te[str(test[p]["ob1"][ix])]]
            print("... %d available training and %d available testing examples with RGBD data for %s" %
                  (len(available_train), len(available_test), p))

            # Prepare input to model.
            tr_f = [rgbd_tr[oidx][ojdx] for _, oidx, ojdx in available_train]
            te_f = [rgbd_te[oidx][ojdx] for _, oidx, ojdx in available_test]
            tr_o = [train[p]["label"][ix] for ix, _, _ in available_train]
            te_o = [test[p]["label"][ix] for ix, _, _ in available_test]

            # Convert structured data into tensor inputs.
            tr_inputs = []
            te_inputs = []
            classes = set()
            for model_in, orig_in, orig_out in [[tr_inputs, tr_f, tr_o],
                                                [te_inputs, te_f, te_o]]:
                for idx in range(len(orig_in)):
                    # Convert RGB and D numpy arrays to tensors and add batch dimension at axis 0.
                    model_in.append([torch.tensor(orig_in[idx][0], dtype=torch.float).unsqueeze(0).to(dv),
                                       torch.tensor(orig_in[idx][1], dtype=torch.float).unsqueeze(0).to(dv)])
                    if orig_out[idx] not in classes:
                        classes.add(orig_out[idx])
            tr_outputs = torch.tensor(tr_o, dtype=torch.long).view(len(tr_o), 1).to(dv)
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
                        trcm[train[p]["label"][idx]][logits.max(0)[1]] += 1

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
                            cm[test[p]["label"][jdx]][logits.max(0)[1]] += 1

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
    maxlen = tr_enc_exps = te_enc_exps = word_to_i_all = word_to_i = None
    print("Preparing infrastructure to include GloVe info...")
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
            tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            layers = None if ff_layers == 0 else [int(emb_dim_l / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
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
            tr_f = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            layers = None if ff_layers == 0 else [int(emb_dim / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
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
            tr_f = [[[[train[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(train[p]["ob1"]))]
            te_f = [[[[test[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                    range(len(test[p]["ob1"]))]
            layers = None if ff_layers == 0 else [int(emb_dim_v / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=args.verbose))

    if 'glove' in models and 'resnet' in models:
        print("Running Glove+ResNet (Shrink) models")
        bs.append("GLoVe+ResNet (Shrink)")
        rs.append({})
        for p in preps:

            # Prepare input to model.
            tr_f_l = [[res[train[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                      range(len(train[p]["ob1"]))]
            te_f_l = [[res[test[p][s][idx]] for s in ["ob1", "ob2"]] for idx in
                      range(len(test[p]["ob1"]))]
            tr_f_v = [[[[train[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                      range(len(train[p]["ob1"]))]
            te_f_v = [[[[test[p][s][idx]]] for s in ["ob1", "ob2"]] for idx in
                      range(len(test[p]["ob1"]))]
            wv = [g, idx_to_v]

            # Convert structured data into tensor inputs.
            tr_inputs = []
            te_inputs = []
            classes = set()
            for model_in, orig_in, orig_out in [[tr_inputs, [tr_f_l, tr_f_v], train[p]["label"]],
                                                [te_inputs, [te_f_l, te_f_v], test[p]["label"]]]:
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
            tr_outputs = torch.tensor(train[p]["label"], dtype=torch.long).view(len(train[p]["label"]), 1).to(dv)
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
                        trcm[train[p]["label"][idx]][logits.max(0)[1]] += 1

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
                            cm[test[p]["label"][jdx]][logits.max(0)[1]] += 1

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args(), device)
