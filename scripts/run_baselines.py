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
from torchvision.transforms.functional import resize
from utils import *


def main(args, dv):
    assert args.baseline is None or args.baseline in ['majority', 'nb_names', 'nb_bow',
                                                      'lstm', 'glove', 'nn_bow', 'resnet', 'glove+resnet',
                                                      'glove+resnetS', 'robot', 'glove+resnetS+robot']
    assert args.glove_infile is not None or (args.baseline is not None and 'glove' not in args.baseline)
    assert args.robot_infile is not None or (args.baseline is not None and 'robot' not in args.baseline)
    verbose = True if args.verbose == 1 else False
    test_run = True if args.test == 1 else False
    assert (not test_run or args.baseline not in ['glove', 'nn_bow', 'resnet', 'glove+resnet', 'glove+resnetS']
            or args.hyperparam_infile is not None)

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.infile + "'...")
    robo_train = robo_test = robodata = None
    if args.robot_infile is not None:
        with open(args.robot_infile, 'r') as f:
            robodata = json.load(f)
    with open(args.infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        lf = all_d["folds"]
        train = lf['train']
        if robodata is not None:
            robo_train = robodata['train']
        preps = train.keys()
        if test_run:
            test = lf['test']
            dev_aug = lf['dev']
            for p in preps:
                for k in dev_aug[p].keys():
                    train[p][k].extend(dev_aug[p][k])
            if robodata is not None:
                robo_test = robodata['test']
                dev_aug = robodata['dev']
                for p in preps:
                    for k in dev_aug[p].keys():
                        robo_train[p][k].extend(dev_aug[p][k])
        else:
            test = lf['dev']  # only ever peak at the dev set.
            if robodata is not None:
                robo_test = robodata['dev']
    print("... done")
    if args.task is not None:
        preps = [args.task]

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
    if args.baseline is None or args.baseline in ['nb_bow', 'lstm', 'glove', 'nn_bow', 'glove+resnet', 'glove+resnetS',
                                                  'glove+resnetS+robot']:
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
    # TODO: work on this until it's not terrible.
    if False and (args.baseline is None or args.baseline == 'lstm'):
        print("Running language encoder lstms...")
        bs.append("Language Encoder")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_lang_2_label(maxlen[p], word_to_i[p],
                                         tr_enc_exps[p], train[p]["label"], te_enc_exps[p], test[p]["label"],
                                         verbose=verbose, batch_size=10000)
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

    # Average GLoVe embeddings concatenated and used to predict class.
    g = None
    emb_dim_l = None
    if args.baseline is None or 'glove' in args.baseline:
        print("Preparing infrastructure to run GLoVe-based feed forward model...")
        ws = set()
        for p in preps:
            ws.update(word_to_i_all[p].keys())
        g, missing = get_glove_vectors(args.glove_infile, ws)
        emb_dim_l = len(g[list(g.keys())[0]])
        print("... done; missing " + str(missing) + " vectors out of " + str(len(ws)))

    if args.baseline is None or args.baseline == 'glove':
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
                                         learning_rate=ff_lr, opt=ff_opt, verbose=verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [g], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=verbose))

    # Average BoW embeddings use to predict class.
    if args.baseline is None or args.baseline == 'nn_bow':
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
                                         learning_rate=ff_lr, opt=ff_opt, verbose=verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [wv], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=verbose))

    # Average BoW embeddings use to predict class.
    idx_to_v = None
    emb_dim_v = None
    if args.baseline is None or 'resnet' in args.baseline:
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
                im = torch.unsqueeze(im, 0)
                idx_to_v[idx] = plm(im).detach().data.numpy().flatten()
                with open(ffn, 'w') as f:
                    f.write(' '.join([str(i) for i in idx_to_v[idx]]))
        emb_dim_v = len(idx_to_v[0])
        print("... done")

    if args.baseline is None or args.baseline == 'resnet':
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
                                         learning_rate=ff_lr, opt=ff_opt, verbose=verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [idx_to_v], [tr_f], train[p]["label"], [te_f], test[p]["label"],
                                                  layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                                  learning_rate=ff_lr, opt=ff_opt, verbose=verbose))

    if args.baseline is None or args.baseline == 'glove+resnet':
        print("Running Glove+ResNet FF models")
        bs.append("GLoVe+ResNet FF")
        rs.append({})
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
            layers = None if ff_layers == 0 else [int(emb_dim / np.power(ff_width_decay, lidx) + 0.5)
                                                  for lidx in range(ff_layers)]
            if ff_random_restarts is None:
                rs[-1][p] = run_ff_model(dv, [g, idx_to_v], [tr_f_l, tr_f_v], train[p]["label"],
                                         [te_f_l, te_f_v], test[p]["label"],
                                         layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                         learning_rate=ff_lr, opt=ff_opt, verbose=verbose)
            else:
                rs[-1][p] = []
                for seed in ff_random_restarts:
                    print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rs[-1][p].append(run_ff_model(dv, [g, idx_to_v], [tr_f_l, tr_f_v], train[p]["label"],
                                     [te_f_l, te_f_v], test[p]["label"],
                                     layers=layers, epochs=ff_epochs, dropout=ff_dropout,
                                     learning_rate=ff_lr, opt=ff_opt, verbose=verbose))

    robot_softmax = {}  # map from pairs of objects to softmax over classes
    tr_robot_softmax = {}  # map from pairs of objects to softmax over classes
    if args.baseline is None or 'robot' in args.baseline:
        print("Running Robot Gaussian Naive Bayes models")
        bs.append("Robot GNB")
        rs.append({})
        for p in preps:
            robo_test = robo_train  # DEBUG
            test = train  # DEBUG

            tr_f = np.asmatrix(
                [robo_train[p]['feats'][idx][jdx]
                 for idx in range(len(train[p]["label"])) if robo_train[p]['feats'][idx] is not None
                 for jdx in range(len(robo_train[p]['feats'][idx]))])
            tr_l = np.asarray([robo_train[p]['label'][idx]
                               for idx in range(len(train[p]["label"])) if robo_train[p]['feats'][idx] is not None
                               for _ in range(len(robo_train[p]['feats'][idx]))])
            te_f = np.asmatrix(
                [robo_test[p]['feats'][idx][jdx]
                 for idx in range(len(test[p]["label"])) if robo_test[p]['feats'][idx] is not None
                 for jdx in range(len(robo_test[p]['feats'][idx]))])
            te_l = np.asarray([robo_test[p]['label'][idx]
                               for idx in range(len(test[p]["label"])) if robo_test[p]['feats'][idx] is not None
                               for _ in range(len(robo_test[p]['feats'][idx]))])
            classes = set(train[p]["label"])

            # DEBUG: show majority class results for comparison.
            print(run_majority_class(tr_l, te_l))[0]

            gnb = GaussianNB()
            gnb.fit(tr_f, tr_l)
            if te_f.shape[1] > 0:
                pcst = gnb.predict(te_f)
                pcs = get_classes_by_vote([robo_test[p]['feats'][idx] for idx in range(len(test[p]["label"]))
                                           if robo_test[p]['feats'][idx] is not None], pcst)
                robot_softmax_l = get_softmax_by_vote([robo_test[p]['feats'][idx]
                                                       for idx in range(len(test[p]["label"]))
                                                       if robo_test[p]['feats'][idx] is not None], pcst, len(classes))
                robot_softmax_k = [(robo_test[p]["ob1"][idx], robo_test[p]["ob2"][idx])
                                   for idx in range(len(test[p]["label"])) if robo_test[p]['feats'][idx] is not None]
                robot_softmax = {robot_softmax_k[idx]: robot_softmax_l[idx] for idx in range(len(robot_softmax_k))}
                gcs = [robo_test[p]['label'][idx] for idx in range(len(test[p]["label"]))
                       if robo_test[p]['feats'][idx] is not None]
                cm = np.zeros(shape=(len(classes), len(classes)))
                for idx in range(len(gcs)):
                    cm[gcs[idx]][pcs[idx]] += 1
            else:
                cm = np.zeros(shape=(len(classes), len(classes)))
                print("WARNING: no data in the test set!")

            pcst = gnb.predict(tr_f)
            pcs = get_classes_by_vote([robo_train[p]['feats'][idx] for idx in range(len(train[p]["label"]))
                                       if robo_train[p]['feats'][idx] is not None], pcst)
            tr_robot_softmax_l = get_softmax_by_vote([robo_train[p]['feats'][idx]
                                                      for idx in range(len(train[p]["label"]))
                                                      if robo_train[p]['feats'][idx] is not None], pcst, len(classes))
            tr_robot_softmax_k = [(robo_train[p]["ob1"][idx], robo_train[p]["ob2"][idx])
                                  for idx in range(len(train[p]["label"])) if robo_train[p]['feats'][idx] is not None]
            tr_robot_softmax = {tr_robot_softmax_k[idx]: tr_robot_softmax_l[idx]
                                for idx in range(len(tr_robot_softmax_k))}
            gcs = [robo_train[p]['label'][idx] for idx in range(len(train[p]["label"]))
                   if robo_train[p]['feats'][idx] is not None]
            trcm = np.zeros(shape=(len(classes), len(classes)))
            for idx in range(len(gcs)):
                trcm[gcs[idx]][pcs[idx]] += 1

            rs[-1][p] = get_acc(cm), cm, get_acc(trcm), trcm

    if args.baseline is None or 'glove+resnetS' in args.baseline:
        if 'robot' not in args.baseline:
            print("Running Glove+ResNet (Shrink) FF models")
            bs.append("GLoVe+ResNet (Shrink) FF")
        else:
            print("Running Glove+ResNet+Robot models")
            bs.append("GLoVe+ResNet+Robot")
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

                        # Record performance only on subset of data that we have robot features for in MM model.
                        # FF model is still trained on all data, but we only want the CM for robot features.
                        if 'robot' in args.baseline:
                            detatched_softmax = F.softmax(logits.clone(), dim=0)
                            ob1 = train[p]["ob1"][idx]
                            ob2 = train[p]["ob2"][idx]
                            if (ob1, ob2) in tr_robot_softmax:
                                detatched_softmax += torch.tensor(tr_robot_softmax[(ob1, ob2)],
                                                                  dtype=torch.float).to(dv)
                                trcm[train[p]["label"][idx]][detatched_softmax.max(0)[1]] += 1
                        else:
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

                            if 'robot' in args.baseline:
                                detatched_softmax = F.softmax(logits.clone(), dim=0)
                                ob1 = test[p]["ob1"][jdx]
                                ob2 = test[p]["ob2"][jdx]
                                if (ob1, ob2) in robot_softmax:
                                    # TODO: this is where we can get the glove+resnet vote, the robot vote,
                                    # TODO: and the pooled vote for visualization / analysis in the paper.
                                    detatched_softmax += torch.tensor(robot_softmax[(ob1, ob2)],
                                                                      dtype=torch.float).to(dv)
                                    cm[test[p]["label"][jdx]][detatched_softmax.max(0)[1]] += 1
                            else:
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
    parser.add_argument('--baseline', type=str, required=False,
                        help="if None, all will run, else 'majority', 'nb_names', 'nb_bow', 'lstm', 'glove'," +
                             " 'nn_bow', 'resnet', 'glove+resnet', 'glove+resnetS', 'robot'," +
                             " 'gloveS+resnetS+robot")
    parser.add_argument('--task', type=str, required=False,
                        help="the task to perform; if None, will do both 'on' and 'in'")
    parser.add_argument('--glove_infile', type=str, required=False,
                        help="input glove vector text file if running glove baseline")
    parser.add_argument('--robot_infile', type=str, required=False,
                        help="input robot feature file")
    parser.add_argument('--hyperparam_infile', type=str, required=False,
                        help="input json for model hyperparameters")
    parser.add_argument('--perf_outfile', type=str, required=False,
                        help="output json for model performance")
    parser.add_argument('--verbose', type=int, required=False,
                        help="1 if desired")
    parser.add_argument('--ff_epochs', type=int, required=False,
                        help="override default number of epochs")
    parser.add_argument('--ff_random_restarts', type=str, required=False,
                        help="comma-separated list of random seeds to use; presents avg + stddev data instead of cms")
    parser.add_argument('--test', type=int, required=False,
                        help="if set to 1, evaluates on the test set; NOT FOR TUNING")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args(), device)
