#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
from models import *
from scipy.stats import ttest_ind
from tqdm import tqdm
from utils import *


def keep_all_but(l1, l2, keep_but):
    return [l1[idx] for idx in range(len(l1)) if l2[idx] != keep_but]


def return_input(x):
    return x


def main(args, dv):

    # Near universals it would be better to read from data but which we're hard-coding.
    preps = ["in", "on"]
    num_trials = 5  # fixed back in prep_torch_data

    # Set hyperparameters, some of which are on a per-preposition basis.
    batch_size = 8
    num_epochs = 30

    hyperparam = {p: {} for p in preps}
    # Input transformation (RGBD)
    hyperparam["in"]["rgbd_inp_trans"] = None
    hyperparam["on"]["rgbd_inp_trans"] = None
    # fixed hidden dimension (RGBD, L+V)
    hyperparam["in"]["hidden_dim"] = 64
    hyperparam["on"]["hidden_dim"] = 64
    # Dropout (RGBD, L+V)
    hyperparam["in"]["dropout"] = 0.3
    hyperparam["on"]["dropout"] = 0.3
    # Learning rate (RGBD, L+V)
    hyperparam["in"]["rgbd_learning_rate"] = 0.01
    hyperparam["on"]["rgbd_learning_rate"] = 0.01
    hyperparam["in"]["lv_learning_rate"] = 0.01
    hyperparam["on"]["lv_learning_rate"] = 0.001
    # Optimizer (RGBD, L+V)
    hyperparam["in"]["opt"] = 'adam'
    hyperparam["on"]["opt"] = 'adam'
    # Activation (RGBD, L+V)
    hyperparam["in"]["activation"] = 'relu'
    hyperparam["on"]["activation"] = 'relu'

    # labels to use.
    train_label = args.train_objective + "_label"
    test_label = args.test_objective + "_label"
    models = args.models.split(',')

    # Read in torch-ready input data.
    print("Reading in torch-ready lists from json and converting them to tensors...")
    tr_outputs = {}
    tr_names = {}
    tr_inputs_l = {}
    tr_inputs_v = {}
    tr_inputs_rgb = {}
    tr_inputs_d = {}
    te_outputs = {}
    te_names = {}
    te_inputs_l = {}
    te_inputs_v = {}
    te_inputs_rgb = {}
    te_inputs_d = {}

    for p in preps:
        fn = args.input + '.' + p
        with open(fn, 'r') as f:
            d = json.load(f)

            # # Can round "maybe" (1, of 0,1,2) class down to "no" at training time if we're testing on RGBD data
            # in general, if using five trial voting for Y/N instead of MC with M as a training example.
            # |
            # Need to round "maybe" (1, of 0,1,2) class down to "no" at training time in mturk labels if test objective
            # is two-class Y/N (1/0) only 'human' labels.
            if (('rgbd' in models and args.rgbd_m_as_disagreement) or
                    (args.test_objective == "human" and args.train_objective == "mturk")):
                cmtr = d["train"][train_label].count([1])
                d["train"][train_label] = [[2] if v[0] > 1 else [0] for v in d["train"][train_label]]
                print("... for %s training data, rounded %d Maybe labels down to No values" %
                      (p, cmtr))
            if args.round_m_to_n == 1:
                # WARNING: makes labels 0 -> N, 1 -> Y, unlike triple label scheme!
                cmtr = d["train"][train_label].count([1])
                d["train"][train_label] = [[1] if v[0] > 1 else [0] for v in d["train"][train_label]]
                print("... for %s training data, rounded %d Maybe labels down to No values" %
                      (p, cmtr))
                cmte = d["test"][test_label].count([1])
                d["test"][test_label] = [[1] if v[0] > 1 else [0] for v in d["test"][test_label]]
                print("... for %s testing data, rounded %d Maybe labels down to No values" %
                      (p, cmte))

            tr_outputs[p] = torch.tensor(keep_all_but(d["train"][train_label], d["train"][train_label], [-1]),
                                         dtype=torch.float).to(dv)
            tr_names[p] = keep_all_but(d["train"]["names"], d["train"][train_label], [-1])
            tr_inputs_l[p] = torch.tensor(keep_all_but(d["train"]["lang"], d["train"][train_label], [-1]),
                                          dtype=torch.float).to(dv)
            tr_inputs_v[p] = torch.tensor(keep_all_but(d["train"]["vis"], d["train"][train_label], [-1]),
                                          dtype=torch.float).to(dv)
            tr_inputs_rgb[p] = torch.tensor(keep_all_but(d["train"]["rgb"], d["train"][train_label], [-1]),
                                            dtype=torch.float).to(dv) if d["train"]["rgb"] is not None else None
            tr_inputs_d[p] = torch.tensor(keep_all_but(d["train"]["d"], d["train"][train_label], [-1]),
                                          dtype=torch.float).to(dv) if d["train"]["d"] is not None else None
            te_outputs[p] = torch.tensor(keep_all_but(d["test"][test_label], d["test"][test_label], [-1]),
                                         dtype=torch.float).to(dv)
            te_names[p] = keep_all_but(d["test"]["names"], d["test"][test_label], [-1])
            te_inputs_l[p] = torch.tensor(keep_all_but(d["test"]["lang"], d["test"][test_label], [-1]),
                                          dtype=torch.float).to(dv)
            te_inputs_v[p] = torch.tensor(keep_all_but(d["test"]["vis"], d["test"][test_label], [-1]),
                                          dtype=torch.float).to(dv)
            te_inputs_rgb[p] = torch.tensor(keep_all_but(d["test"]["rgb"], d["test"][test_label], [-1]),
                                            dtype=torch.float).to(dv) if d["test"]["rgb"] is not None else None
            te_inputs_d[p] = torch.tensor(keep_all_but(d["test"]["d"], d["test"][test_label], [-1]),
                                          dtype=torch.float).to(dv) if d["test"]["d"] is not None else None

            if args.zero_lang:
                print("... zeroing out language modality for %s" % p)
                tr_inputs_l[p] = torch.zeros(tr_inputs_l[p].shape)
                te_inputs_l[p] = torch.zeros(te_inputs_l[p].shape)
            if args.zero_vis:
                print("... zeroing out vision modality for %s" % p)
                tr_inputs_v[p] = torch.zeros(tr_inputs_v[p].shape)
                te_inputs_v[p] = torch.zeros(te_inputs_v[p].shape)
            if args.zero_ego:
                print("... zeroing out egocentric modalities for %s" % p)
                tr_inputs_rgb[p] = torch.zeros(tr_inputs_rgb[p].shape)
                tr_inputs_d[p] = torch.zeros(tr_inputs_d[p].shape)
                te_inputs_rgb[p] = torch.zeros(te_inputs_rgb[p].shape)
                te_inputs_d[p] = torch.zeros(te_inputs_d[p].shape)

        print("... %s done; num train out %d, num test out %d" % (p, tr_outputs[p].shape[0], te_outputs[p].shape[0]))

        train_classes = set([int(v.item()) for v in tr_outputs[p]])
        test_classes = set([int(v.item()) for v in te_outputs[p]])
        if train_classes != test_classes:
            print("...... WARNING: train classes " + str(train_classes) + " do not match test classes "
                  + str(test_classes) + " for " + p + "; will use union")
        classes = list(train_classes.union(test_classes))

    print("... classes: " + str(classes))
    bs = []
    rs = []
    random_restarts = None if args.random_restarts is None else \
        [int(s) for s in args.random_restarts.split(',')]

    # If running with seeds, set deterministic CuDNN behavior.
    if random_restarts is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Random baseline.
    if 'random' in models:
        print("Running random baseline...")
        bs.append("Random")
        rs.append({})
        for p in preps:
            if random_restarts is None:
                seeds = [None]
            else:
                seeds = random_restarts
                rs[-1][p] = []
            for seed in tqdm(seeds, desc="for '%s'" % p):
                if seed is not None:
                    # print("... %s with seed %d ..." % (p, seed))
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                results = run_random([int(c[0]) for c in tr_outputs[p].detach().data.cpu().numpy().tolist()],
                                     [int(c[0]) for c in te_outputs[p].detach().data.cpu().numpy().tolist()],
                                     len(classes))

                if random_restarts is None:
                    rs[-1][p] = results
                else:
                    rs[-1][p].append(results)
        print("... done")

    # Majority class baseline.
    if 'mc' in models:
        print("Running majority class baseline...")
        bs.append("Majority Class")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_majority_class([int(c[0]) for c in tr_outputs[p].detach().data.cpu().numpy().tolist()],
                                           [int(c[0]) for c in te_outputs[p].detach().data.cpu().numpy().tolist()],
                                           len(classes))
            if random_restarts is not None:
                rs[-1][p] = [rs[-1][p] for _ in range(len(random_restarts))]
        print("... done")

    # Majority class baseline.
    if 'omc' in models:
        print("Running oracle majority class baseline...")
        bs.append("Oracle Majority Class")
        rs.append({})
        for p in preps:
            rs[-1][p] = run_majority_class([int(c[0]) for c in te_outputs[p].detach().data.cpu().numpy().tolist()],
                                           [int(c[0]) for c in te_outputs[p].detach().data.cpu().numpy().tolist()],
                                           len(classes))
            if random_restarts is not None:
                rs[-1][p] = [rs[-1][p] for _ in range(len(random_restarts))]
        print("... done")

    if 'rgbd' in models:
        print("Running RGBD models")
        bs.append("RGBD")
        rs.append({})
        for p in preps:
            if tr_inputs_rgb[p] is None:
                print("... ERROR: data had no RGBD features for prep %s" % p)
                del bs[-1]
                del rs[-1]
                continue

            if hyperparam[p]["rgbd_inp_trans"] is None:
                intrans = return_input
            elif hyperparam[p]["rgbd_inp_trans"] == 'tanh':
                intrans = torch.tanh
            elif hyperparam[p]["rgbd_inp_trans"] == 'sigmoid':
                intrans = torch.sigmoid
            else:
                sys.exit("Unrecognized input transformation %s" % hyperparam[p]["in_trans"])

            # Couple the RGB and D inputs to feed into the paired inputs of the conv network.
            tr_inputs = [[tr_inputs_rgb[p][idx], tr_inputs_d[p][idx]] for idx in range(len(tr_outputs[p]))]
            te_inputs = [[te_inputs_rgb[p][idx], te_inputs_d[p][idx]] for idx in range(len(te_outputs[p]))]

            # Train model.
            if random_restarts is None:
                seeds = [None]
            else:
                seeds = random_restarts
                rs[-1][p] = []
            for seed in tqdm(seeds, desc="for '%s'" % p):
                if seed is not None:
                    # print("... %s with seed %d ..." % (p, seed))
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                # Instantiate fusion model for RGB and D inputs, then run them through a prediction model to get logits.
                rgbd_fusion_model = ConvFusionFFModel(dv, 4, 1, hyperparam[p]["hidden_dim"],
                                                      hyperparam[p]["activation"]).to(dv)
                model = PredModel(dv, rgbd_fusion_model, hyperparam[p]["hidden_dim"], len(classes)).to(dv)

                # Instantiate loss and optimizer.
                # TODO: optimizer is a hyperparam to set.
                loss_function = nn.CrossEntropyLoss()
                if hyperparam[p]["opt"] == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                elif hyperparam[p]["opt"] == 'adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                elif hyperparam[p]["opt"] == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                elif hyperparam[p]["opt"] == 'rmsprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                else:
                    raise ValueError('Unrecognized opt specification "' + hyperparam[p]["opt"] + '".')

                # Run training for specified number of epochs.
                best_acc = best_cm = tr_acc_at_best = trcm_at_best = tloss_at_best = t_epochs = predictions = None
                idxs = list(range(len(tr_inputs)))
                np.random.shuffle(idxs)
                tr_inputs = [tr_inputs[idx] for idx in idxs]
                tro = tr_outputs[p][idxs, :]
                result = None
                last_saved_fn = None
                for epoch in range(num_epochs):
                    tloss = 0
                    trcm = np.zeros(shape=(len(classes), len(classes)))
                    tidx = 0
                    batches_run = 0
                    while tidx < len(tr_inputs):
                        model.train()
                        model.zero_grad()
                        batch_in = [torch.zeros((batch_size * num_trials, tr_inputs[0][0].shape[1],
                                                 tr_inputs[0][0].shape[2], tr_inputs[0][0].shape[3])).to(dv),
                                    torch.zeros((batch_size * num_trials, tr_inputs[0][1].shape[1],
                                                 tr_inputs[0][1].shape[2], tr_inputs[0][1].shape[3])).to(dv)]
                        batch_gold = torch.zeros(batch_size * num_trials).to(dv)
                        for bidx in range(batch_size):
                            batch_in[0][bidx:bidx+num_trials, :, :, :] = tr_inputs[tidx][0]
                            batch_in[1][bidx:bidx+num_trials, :] = intrans(tr_inputs[tidx][1])
                            batch_gold[bidx:bidx+num_trials] = np.repeat(tro[tidx][0], num_trials)

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
                        model.eval()
                        cm = np.zeros(shape=(len(classes), len(classes)))
                        curr_predictions = {}  # map from pair -> label
                        for jdx in range(len(te_inputs)):
                            trials_logits = model([te_inputs[jdx][0], intrans(te_inputs[jdx][1])])
                            v = np.zeros(len(classes))
                            for tdx in range(num_trials):  # take a vote over trials (not whole logit size)
                                v[int(trials_logits[tdx].argmax(0))] += 1
                            
                            if args.rgbd_m_as_disagreement:
                                if v[1] == v[2] == 0:  # all votes are for class negative
                                    chosen_class = 0
                                elif v[0] == v[1] == 0:  # all votes are for class positive
                                    chosen_class = 2
                                else:  # votes are split among different classes, so conservatively vote maybe
                                    chosen_class = 1
                            else:
                                # If N/M/Y are all available, just use the majority vote across the trials
                                # to decide the predicted label of the pair.
                                chosen_class = int(v.argmax(0))

                            cm[int(te_outputs[p][jdx])][chosen_class] += 1
                            curr_predictions["(%s, %s)" % (te_names[p][jdx][0], te_names[p][jdx][1])] = chosen_class

                        acc = get_acc(cm)
                        if best_acc is None or acc > best_acc:
                            best_acc = acc
                            best_cm = cm
                            tr_acc_at_best = tr_acc
                            trcm_at_best = trcm
                            tloss_at_best = tloss
                            t_epochs = epoch + 1  # record how many epochs of training have happened at this time
                            predictions = curr_predictions  # record pair -> class predictions for at best epoch.

                            # Save best-performing model at this seed
                            if seed is not None:
                                fn = os.path.join(args.outdir, "p-%s_m-rgbd_tr-%s_te-%s_s-%s_acc-%.3f.pt" %
                                    (p, args.train_objective, args.test_objective, seed, best_acc))
                            else:
                                fn = os.path.join(args.outdir, "p-%s_m-rgbd_tr-%s_te-%s_acc-%.3f.pt" %
                                    (p, args.train_objective, args.test_objective, best_acc))
                            torch.save(model.state_dict(), fn)
                            if last_saved_fn is not None:  # remove previous save
                                os.system("rm %s" % last_saved_fn)
                            last_saved_fn = fn

                    result = best_acc, best_cm, tr_acc_at_best, trcm_at_best, tloss_at_best, t_epochs, predictions

                if seed is None:
                    rs[-1][p] = result
                else:
                    rs[-1][p].append(result)
        print("... done")

    if 'glove' in models:
        print("Running GloVe models")
        bs.append("GloVe FF")
        rs.append({})
        for p in preps:
            if random_restarts is None:
                model_desc = "p-%s_m-glove_tr-%s_te-%s" % (p, args.train_objective, args.test_objective)
                rs[-1][p] = run_ff_model(dv, args.outdir, model_desc,
                                         tr_inputs_l[p], tr_outputs[p], te_inputs_l[p], te_outputs[p],
                                         tr_inputs_l[p].shape[1], hyperparam[p]["hidden_dim"],
                                         len(classes), num_modalities=1,
                                         epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                         learning_rate=hyperparam[p]["lv_learning_rate"], opt=hyperparam[p]["opt"],
                                         activation=hyperparam[p]["activation"],
                                         verbose=args.verbose, batch_size=batch_size)
            else:
                rs[-1][p] = []
                for seed in tqdm(random_restarts, desc="for '%s'" % p):
                    # print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    model_desc = "p-%s_m-glove_tr-%s_te-%s_s-%s" % (p, args.train_objective, args.test_objective, seed)
                    rs[-1][p].append(run_ff_model(dv, args.outdir, model_desc,
                                                  tr_inputs_l[p], tr_outputs[p], te_inputs_l[p], te_outputs[p],
                                                  tr_inputs_l[p].shape[1], hyperparam[p]["hidden_dim"],
                                                  len(classes), num_modalities=1,
                                                  epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                                  learning_rate=hyperparam[p]["lv_learning_rate"], opt=hyperparam[p]["opt"],
                                                  activation=hyperparam[p]["activation"],
                                                  verbose=args.verbose, batch_size=batch_size))
        print("... done")

    if 'resnet' in models:
        print("Running ResNet FF models")
        bs.append("ResNet FF")
        rs.append({})
        for p in preps:
            if random_restarts is None:
                model_desc = "p-%s_m-resnet_tr-%s_te-%s" % (p, args.train_objective, args.test_objective)
                rs[-1][p] = run_ff_model(dv, args.outdir, model_desc,
                                         tr_inputs_v[p], tr_outputs[p], te_inputs_v[p], te_outputs[p],
                                         tr_inputs_v[p].shape[1], hyperparam[p]["hidden_dim"],
                                         len(classes), num_modalities=1,
                                         epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                         learning_rate=hyperparam[p]["lv_learning_rate"], opt=hyperparam[p]["opt"],
                                         activation=hyperparam[p]["activation"],
                                         verbose=args.verbose, batch_size=batch_size)
            else:
                rs[-1][p] = []
                for seed in tqdm(random_restarts, desc="for '%s'" % p):
                    # print("... with seed " + str(seed) + "...")
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    model_desc = "p-%s_m-resnet_tr-%s_te-%s_s-%s" % (p, args.train_objective, args.test_objective, seed)
                    rs[-1][p].append(run_ff_model(dv, args.outdir, model_desc,
                                                  tr_inputs_v[p], tr_outputs[p], te_inputs_v[p], te_outputs[p],
                                                  tr_inputs_v[p].shape[1], hyperparam[p]["hidden_dim"],
                                                  len(classes),num_modalities=1,
                                                  epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                                  learning_rate=hyperparam[p]["lv_learning_rate"], opt=hyperparam[p]["opt"],
                                                  activation=hyperparam[p]["activation"],
                                                  verbose=args.verbose, batch_size=batch_size))
        print("... done")

    # Load all possible pretraining parameters from given dir.
    lv_pretrained_fns_settings = [None]
    if args.lv_pretrained_dir is not None:
        in_fns = []
        on_fns = []
        for _, _, fns in os.walk(args.lv_pretrained_dir):
            for fn in fns:
                if fn.split('.')[-1] == "fm" and "loss" in fn:
                    if "p-on" in fn:
                        on_fns.append(os.path.join(args.lv_pretrained_dir, fn))
                    elif "p-in" in fn:
                        in_fns.append(os.path.join(args.lv_pretrained_dir, fn))
        for idx in range(len(in_fns)):
            lv_pretrained_fns_settings.append("%s,%s" % (on_fns[idx], in_fns[idx]))

    if 'glove+resnet' in models:
        print("Running GloVe+ResNet FF models")
        for lv_pretrained_fns in lv_pretrained_fns_settings:
            bs.append("GloVe+ResNet FF")
            if lv_pretrained_fns is not None:
                bs[-1] += "_lv-Pretrained_" + lv_pretrained_fns
            rs.append({})
            for p in preps:

                # Optionally load pre-trained weights for the L+V model component.
                if lv_pretrained_fns is not None:
                    lv_pretrained_fn = lv_pretrained_fns.split(',')[0] if p == 'on' \
                        else lv_pretrained_fns.split(',')[1]
                else:
                    lv_pretrained_fn = None

                if random_restarts is None:
                    model_desc = "p-%s_m-glove+resnet_tr-%s_te-%s" % (p, args.train_objective, args.test_objective)
                    rs[-1][p] = run_ff_model(dv, args.outdir, model_desc,
                                             [tr_inputs_l[p], tr_inputs_v[p]], tr_outputs[p],
                                             [te_inputs_l[p], te_inputs_v[p]], te_outputs[p],
                                             None, hyperparam[p]["hidden_dim"], len(classes), num_modalities=2,
                                             epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                             learning_rate=hyperparam[p]["lv_learning_rate"], opt=hyperparam[p]["opt"],
                                             activation=hyperparam[p]["activation"],
                                             verbose=args.verbose, batch_size=batch_size,
                                             pretrained=lv_pretrained_fn)
                else:
                    rs[-1][p] = []
                    for seed in tqdm(random_restarts, desc="for '%s'" % p):
                        # print("... with seed " + str(seed) + "...")
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        model_desc = "p-%s_m-glove+resnet_tr-%s_te-%s_s-%s" % (p, args.train_objective,
                                                                               args.test_objective, seed)
                        rs[-1][p].append(run_ff_model(dv, args.outdir, model_desc,
                                                      [tr_inputs_l[p], tr_inputs_v[p]], tr_outputs[p],
                                                      [te_inputs_l[p], te_inputs_v[p]], te_outputs[p],
                                                      None, hyperparam[p]["hidden_dim"], len(classes), num_modalities=2,
                                                      epochs=num_epochs, dropout=hyperparam[p]["dropout"],
                                                      learning_rate=hyperparam[p]["lv_learning_rate"],
                                                      opt=hyperparam[p]["opt"],
                                                      activation=hyperparam[p]["activation"],
                                                      verbose=args.verbose, batch_size=batch_size,
                                                      pretrained=lv_pretrained_fn,
                                                      pretrained_freeze=True if args.lv_pretrained_freeze == 1
                                                      else False))
        print("... done")

    if 'rgbd+glove+resnet' in models:
        print("Running RGBD+GloVe+ResNet models")
        for lv_pretrained_fns in lv_pretrained_fns_settings:
            bs.append("RGBD+GloVe+ResNet")
            if lv_pretrained_fns is not None:
                bs[-1] += "_lv-Pretrained_" + lv_pretrained_fns
            rs.append({})
            for p in preps:
                if tr_inputs_rgb[p] is None:
                    print("... ERROR: data had no RGBD features for prep %s" % p)
                    del bs[-1]
                    del rs[-1]
                    continue

                if hyperparam[p]["rgbd_inp_trans"] is None:
                    intrans = return_input
                elif hyperparam[p]["rgbd_inp_trans"] == 'tanh':
                    intrans = torch.tanh
                elif hyperparam[p]["rgbd_inp_trans"] == 'sigmoid':
                    intrans = torch.sigmoid
                else:
                    sys.exit("Unrecognized input transformation %s" % hyperparam[p]["in_trans"])

                # Couple the RGB, D, L, V inputs to feed into the conv and FF networks.
                tr_inputs = [[tr_inputs_rgb[p][idx], tr_inputs_d[p][idx], tr_inputs_l[p][idx], tr_inputs_v[p][idx]]
                             for idx in range(len(tr_outputs[p]))]
                te_inputs = [[te_inputs_rgb[p][idx], te_inputs_d[p][idx], te_inputs_l[p][idx], te_inputs_v[p][idx]]
                             for idx in range(len(te_outputs[p]))]

                # Train model.
                if lv_pretrained_fns is not None:
                    print("... loading pretrained model weights for glove+resnet from '" + lv_pretrained_fns + "'...")
                    if args.lv_pretrained_freeze == 1:
                        print("...... and freezing pretrained layers")
                if random_restarts is None:
                    seeds = [None]
                else:
                    seeds = random_restarts
                    rs[-1][p] = []
                for seed in tqdm(seeds, desc="for '%s'" % p):
                    if seed is not None:
                        # print("... %s with seed %d ..." % (p, seed))
                        np.random.seed(seed)
                        torch.manual_seed(seed)

                    # Instantiate convolutional RGBD fusion model and feed-forward L+V fusion model,
                    # then tie them together with a multi-model prediction model that predicts
                    # logits from their outputs.
                    rgbd_fusion_model = ConvFusionFFModel(dv, 4, 1, hyperparam[p]["hidden_dim"],
                                                          hyperparam[p]["activation"]).to(dv)
                    lv_fusion_model = ShrinkingFusionFFModel(dv, len(tr_inputs[0][2]), len(tr_inputs[0][3]),
                                                             hyperparam[p]["hidden_dim"], hyperparam[p]["dropout"],
                                                             hyperparam[p]["activation"]).to(dv)

                    # Optionally load pre-trained weights for the L+V model component.
                    if lv_pretrained_fns is not None:
                        lv_pretrained_fn = lv_pretrained_fns.split(',')[0] if p == 'on' \
                            else lv_pretrained_fns.split(',')[1]
                        lv_fusion_model.load_state_dict(torch.load(lv_pretrained_fn))

                    model = MultiPredModel(dv, rgbd_fusion_model, lv_fusion_model,
                                           hyperparam[p]["hidden_dim"], len(classes)).to(dv)

                    # Instantiate loss and optimizer.
                    loss_function = nn.CrossEntropyLoss()
                    if hyperparam[p]["opt"] == 'sgd':
                        optimizer = optim.SGD(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                    elif hyperparam[p]["opt"] == 'adagrad':
                        optimizer = optim.Adagrad(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                    elif hyperparam[p]["opt"] == 'adam':
                        optimizer = optim.Adam(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                    elif hyperparam[p]["opt"] == 'rmsprop':
                        optimizer = optim.RMSprop(model.parameters(), lr=hyperparam[p]["rgbd_learning_rate"])
                    else:
                        raise ValueError('Unrecognized opt specification "' + hyperparam[p]["opt"] + '".')

                    # Freeze weights.
                    if lv_pretrained_fns is not None and args.lv_pretrained_freeze == 1:
                        for param in lv_fusion_model.parameters():
                            param.requires_grad = False

                    # Run training for specified number of epochs.
                    best_acc = best_cm = tr_acc_at_best = trcm_at_best = tloss_at_best = t_epochs = predictions = None
                    idxs = list(range(len(tr_inputs)))
                    np.random.shuffle(idxs)
                    tr_inputs = [tr_inputs[idx] for idx in idxs]
                    tro = tr_outputs[p][idxs, :]
                    result = None
                    last_saved_fn = None
                    for epoch in range(num_epochs):
                        tloss = 0
                        trcm = np.zeros(shape=(len(classes), len(classes)))
                        tidx = 0
                        batches_run = 0
                        while tidx < len(tr_inputs):
                            model.train()
                            model.zero_grad()
                            batch_in = [[torch.zeros((batch_size * num_trials, tr_inputs[0][0].shape[1],
                                                     tr_inputs[0][0].shape[2], tr_inputs[0][0].shape[3])).to(dv),
                                         torch.zeros((batch_size * num_trials, tr_inputs[0][1].shape[1],
                                                      tr_inputs[0][1].shape[2], tr_inputs[0][1].shape[3])).to(dv)],
                                        [torch.zeros((batch_size * num_trials, tr_inputs[0][2].shape[0])).to(dv),
                                         torch.zeros((batch_size * num_trials, tr_inputs[0][3].shape[0])).to(dv)]]
                            batch_gold = torch.zeros(batch_size * num_trials).to(dv)
                            for bidx in range(batch_size):
                                batch_in[0][0][bidx:bidx+num_trials, :, :, :] = tr_inputs[tidx][0]
                                batch_in[0][1][bidx:bidx+num_trials, :] = intrans(tr_inputs[tidx][1])
                                # use the same L, V vector across all trials for this pair
                                batch_in[1][0][bidx:bidx+num_trials, :] = tr_inputs[tidx][2].repeat(num_trials, 1)
                                batch_in[1][1][bidx:bidx+num_trials, :] = tr_inputs[tidx][3].repeat(num_trials, 1)
                                batch_gold[bidx:bidx+num_trials] = np.repeat(tro[tidx][0], num_trials)

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
                            model.eval()
                            cm = np.zeros(shape=(len(classes), len(classes)))
                            curr_predictions = {}
                            for jdx in range(len(te_inputs)):

                                te_in = [[torch.tensor(te_inputs[jdx][0]).to(dv),
                                          torch.tensor(intrans(te_inputs[jdx][1])).to(dv)],
                                         [torch.tensor(te_inputs[jdx][2].repeat(num_trials, 1)).to(dv),
                                          torch.tensor(te_inputs[jdx][3].repeat(num_trials, 1)).to(dv)]]
                                trials_logits = model(te_in)  # will be full batch size wide
                                v = np.zeros(len(classes))
                                for tdx in range(num_trials):  # take a vote over trials (not whole logit size)
                                    v[int(trials_logits[tdx].argmax(0))] += 1

                                if args.rgbd_m_as_disagreement:
                                    if v[1] == v[2] == 0:  # all votes are for class negative
                                        chosen_class = 0
                                    elif v[0] == v[1] == 0:  # all votes are for class positive
                                        chosen_class = 2
                                    else:  # votes are split among different classes, so conservatively vote maybe
                                        chosen_class = 1
                                else:
                                    # If N/M/Y are all available, just use the majority vote across the trials
                                    # to decide the predicted label of the pair.
                                    chosen_class = int(v.argmax(0))

                                cm[int(te_outputs[p][jdx])][chosen_class] += 1
                                curr_predictions["(%s, %s)" % (te_names[p][jdx][0], te_names[p][jdx][1])] = chosen_class

                            acc = get_acc(cm)
                            if best_acc is None or acc > best_acc:
                                best_acc = acc
                                best_cm = cm
                                tr_acc_at_best = tr_acc
                                trcm_at_best = trcm
                                tloss_at_best = tloss
                                t_epochs = epoch + 1  # record how many epochs of training have happened at this time
                                predictions = curr_predictions

                                # Save best-performing model at this seed
                                if seed is not None:
                                    fn = os.path.join(args.outdir,
                                                      "p-%s_m-rgbd+glove+resnet_tr-%s_te-%s_s-%s_acc-%.3f.pt" %
                                                      (p, args.train_objective, args.test_objective, seed, best_acc))
                                else:
                                    fn = os.path.join(args.outdir, "p-%s_m-rgbd+glove+resnet_tr-%s_te-%s_acc-%.3f.pt" %
                                                      (p, args.train_objective, args.test_objective, best_acc))
                                torch.save(model.state_dict(), fn)
                                if last_saved_fn is not None:  # remove previous save
                                    os.system("rm %s" % last_saved_fn)
                                last_saved_fn = fn

                        result = best_acc, best_cm, tr_acc_at_best, trcm_at_best, tloss_at_best, t_epochs, predictions

                    if seed is None:
                        rs[-1][p] = result
                    else:
                        rs[-1][p].append(result)
        print("... done")

    if random_restarts is None:
        # Show results.
        print("Results:")
        for idx in range(len(bs)):
            print(" " + bs[idx] + ":")
            for p in preps:
                print("  " + p + ":\tacc %0.3f" % rs[idx][p][0] +
                      "\t(train: %0.3f; loss: %f; epochs %d" % (rs[idx][p][2], rs[idx][p][4], rs[idx][p][5]) + ")")
                print("  \tf1  %0.3f" % get_f1(rs[idx][p][1]) + "\t(train: %0.3f" % get_f1(rs[idx][p][3]) + ")")
                print('\t(TeCM\t' + '\n\t\t'.join(['\t'.join([str(int(ct)) for ct in rs[idx][p][1][i]])
                                                 for i in range(len(rs[idx][p][1]))]) + ")")
                print('\t(TrCM\t' + '\n\t\t'.join(['\t'.join([str(int(ct)) for ct in rs[idx][p][3][i]])
                                                 for i in range(len(rs[idx][p][3]))]) + ")")

    else:
        print("Results:")
        for idx in range(len(bs)):
            print(" " + bs[idx] + ":")
            for p in preps:
                avg_acc = np.average([rs[idx][p][jdx][0] for jdx in range(len(random_restarts))])
                std_acc = np.std([rs[idx][p][jdx][0] for jdx in range(len(random_restarts))])
                avg_tr_acc = np.average([rs[idx][p][jdx][2] for jdx in range(len(random_restarts))])
                std_tr_acc = np.std([rs[idx][p][jdx][2] for jdx in range(len(random_restarts))])
                avg_f1 = np.average([get_f1(rs[idx][p][jdx][1]) for jdx in range(len(random_restarts))])
                std_f1 = np.std([get_f1(rs[idx][p][jdx][1]) for jdx in range(len(random_restarts))])
                avg_tr_f1 = np.average([get_f1(rs[idx][p][jdx][3]) for jdx in range(len(random_restarts))])
                std_tr_f1 = np.std([get_f1(rs[idx][p][jdx][3]) for jdx in range(len(random_restarts))])
                avg_tr_loss = np.average([rs[idx][p][jdx][4] for jdx in range(len(random_restarts))])
                std_tr_loss = np.std([rs[idx][p][jdx][4] for jdx in range(len(random_restarts))])
                avg_tr_epoch = np.average([rs[idx][p][jdx][5] for jdx in range(len(random_restarts))])
                std_tr_epoch = np.std([rs[idx][p][jdx][5] for jdx in range(len(random_restarts))])
                print("  " + p + ":\tacc %0.3f+/-%0.3f" % (avg_acc, std_acc) +
                      "\t(train: %0.3f+/-%0.3f" % (avg_tr_acc, std_tr_acc) + ")")
                print("  \tf1  %0.3f+/-%0.3f" % (avg_f1, std_f1) +
                      "\t(train: %0.3f+/-%0.3f" % (avg_tr_f1, std_tr_f1) + ")")
                print("  \ttrain loss %.3f+/-%.3f" % (avg_tr_loss, std_tr_loss))
                print("  \ttrain epochs %.3f+/-%.3f" % (avg_tr_epoch, std_tr_epoch))

                # Stats tests.
                for jdx in range(idx):
                    _, pv = ttest_ind([rs[idx][p][kdx][0] for kdx in range(len(random_restarts))],
                                      [rs[jdx][p][kdx][0] for kdx in range(len(random_restarts))],
                                      equal_var=False)  # TODO: could maybe relax this to true for non-MC.
                    print("  \t%s p-value %.3f" % (bs[jdx], pv))

        # Write predictions outfile.
        if args.predictions_outfile is not None:
            print("Writing predictions to '%s'..." % args.predictions_outfile)
            with open(args.predictions_outfile, 'w') as f:
                d = {}
                for p in preps:
                    d[p] = {"model": [], "accuracy": [], "predictions": []}
                    for idx in range(len(bs)):
                        for jdx in range(len(random_restarts)):
                            mn = bs[idx] + "_s-" + str(random_restarts[jdx])
                            ma = rs[idx][p][jdx][0]
                            mp = rs[idx][p][jdx][6]
                            d[p]["model"].append(mn)
                            d[p]["accuracy"].append(ma)
                            d[p]["predictions"].append(mp)
                json.dump(d, f)
            print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="torch ready train/test input root to load as json")
    parser.add_argument('--models', type=str, required=True,
                        help="models to run (random, mc, glove, resnet, rgbd)")
    parser.add_argument('--train_objective', type=str, required=True,
                        help="either 'mturk', 'robo', or 'human'")
    parser.add_argument('--test_objective', type=str, required=True,
                        help="either 'mturk' or 'robo'")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to save trained model state_dicts to")
    parser.add_argument('--lv_pretrained_dir', type=str, required=False,
                        help="pretrained state dict dir for a L+V model to use with rgbd+glove+resent (on,in)")
    parser.add_argument('--lv_pretrained_freeze', type=int, required=False,
                        help="whether to freeze pretrained layers")
    parser.add_argument('--rgbd_m_as_disagreement', type=int, required=False,
                        help=("if true, treat the M label as an inference-time-only classification that happens" +
                              " when votes are split between Y/N on the trials available for a pair; at training time," +
                              " the M labels are rounded down to N."))
    parser.add_argument('--round_m_to_n', type=int, required=False,
                        help="if true, round all M labels down to N at train and test time")
    parser.add_argument('--verbose', type=int, required=False, default=0,
                        help="verbosity level")
    parser.add_argument('--random_restarts', type=str, required=False,
                        help="comma-separated list of random seeds to use; presents avg + stddev data instead of cms")
    parser.add_argument('--predictions_outfile', type=str, required=False,
                        help="file to dump prediction JSON across models")
    parser.add_argument('--zero_lang', type=int, required=False,
                        help="if true, zero out object language inputs to get ablated results")
    parser.add_argument('--zero_vis', type=int, required=False,
                        help="if true, zero out object vision inputs to get ablated results")
    parser.add_argument('--zero_ego', type=int, required=False,
                        help="if true, zero out egocentric inputs to get ablated results")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args(), device)
