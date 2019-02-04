#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
import pandas as pd
from PIL import Image
from models import *
from sklearn.metrics import cohen_kappa_score
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from utils import *


def main(args):
    assert not args.rgbd_only or args.exec_robo_indir is not None

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
    available_train = {p: None for p in preps}
    available_test = {p: None for p in preps}
    for p in preps:
        if args.rgbd_only:
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
            available_train[p] = [[ix, train[p]["ob1"][ix], train[p]["ob2"][ix]]
                                  for ix in range(len(train[p]["ob1"]))]
            available_test[p] = [[ix, test[p]["ob1"][ix], test[p]["ob2"][ix]]
                                 for ix in range(len(test[p]["ob1"]))]
            print("... done; %d / %d available training and %d / %d available testing examples for %s" %
                  (len(available_train[p]), len(train[p]["ob1"]), len(available_test[p]), len(test[p]["ob1"]), p))
    tr_o = {p: [train[p]["label"][ix] for ix, _, _ in available_train[p]] for p in preps}
    te_o = {p: [test[p]["label"][ix] for ix, _, _ in available_test[p]] for p in preps}

    # Read in robot execution ground truth labels from annotation CSVs.
    tr_robo_o = {p: [-1 for ix, _, _ in available_train[p]] for p in preps}
    te_robo_o = {p: [-1 for ix, _, _ in available_test[p]] for p in preps}
    if args.exec_robo_indir is not None:
        print("Reading in robot execution ground truth labels from '" + args.exec_robo_indir + "'...")
        gt_labels = {p: {} for p in preps}
        l2c = {"y": 2, "m": 1, "n": 0}  # Matching 3 class split.
        for _, _, fns in os.walk(args.exec_robo_indir):
            for fn in fns:
                ps = fn.split('.')
                if len(ps) >= 2 and ps[-1] == "csv" and ps[-2] == "labels":  # annotation file
                    print("... reading annotations from '%s'..." % fn)
                    num_annot = 0
                    df = pd.read_csv(os.path.join(args.exec_robo_indir, fn))
                    for idx in df.index:
                        if type(df["Pair"][idx]) is not str or " + " not in df["Pair"][idx]:
                            continue
                        lp = df["Pair"][idx].strip().split(" + ")
                        aidx = names.index(lp[0])
                        bidx = names.index(lp[1])
                        k = (aidx, bidx)
                        try:
                            nt = int(df["Num Trials"][idx])
                        except ValueError:
                            continue
                        if nt > 0:
                            for p in preps:
                                al = l2c[df["GT %s" % p][idx].lower()]
                                gt_labels[p][k] = al
                            num_annot += 1
                    print("...... done; read %d useful pair annotations" % num_annot)
        # Tie annotations to label structure.
        for p in preps:
            for fold, fold_struct, available in [["train", tr_robo_o, available_train],
                                                 ["test", te_robo_o, available_test]]:
                gt_found = 0
                for idx in range(len(fold_struct[p])):
                    key = (available[p][idx][1], available[p][idx][2])
                    if key in gt_labels[p]:
                        fold_struct[p][idx] = gt_labels[p][key]
                        gt_found += 1
                print("... done; %d / %d robot execution ground truth %s labels found for %s" %
                      (gt_found, len(fold_struct[p]), fold, p))

    # Read in human execution ground truth labels from annotation CSVs from multiple annotators and calculate agreement.
    tr_human_o = {p: [-1 for ix, _, _ in available_train[p]] for p in preps}
    te_human_o = {p: [-1 for ix, _, _ in available_test[p]] for p in preps}
    if args.exec_human_indir is not None:
        print("Reading in human execution ground truth labels from '" + args.exec_human_indir + "'...")
        annotators = []
        # indexed by p, then (aidx, bidx), then [annotator_id0_label, anntotor_id1_label, ...]
        annotations = {p: {} for p in preps}
        l2c = {"on": {"on": 1, "in": 0},
               "in": {"on": 1, "in": 1},
               "no": {"on": 0, "in": 0},
               "same": None, "-": None, "": None}
        for _, _, fns in os.walk(args.exec_human_indir):
            for fn in fns:
                ps = fn.split('.')
                if ps[-1] == 'csv':  # annotation file
                    annotator = ps[0]
                    if annotator not in annotators:
                        annotators.append(annotator)
                    fn = os.path.join(args.exec_human_indir, fn)
                    print("... reading annotator %s annotations from '%s'..." % (annotator, fn))
                    num_annot = 0
                    with open(fn, 'r') as f:
                        for lidx, line in enumerate(f.readlines()):
                            if lidx == 0:
                                continue
                            lp = line.strip().split(',')
                            aidx = names.index(lp[0])
                            bidx = names.index(lp[1])
                            k = (aidx, bidx)
                            al = l2c[lp[4]]
                            if al is not None:  # label is not "same" or blank, so we have an annotation to add
                                for p in preps:
                                    if k not in annotations[p]:
                                        annotations[p][k] = []
                                    annotations[p][k].append(al[p])
                                num_annot += 1
                    print("...... done; read %d useful pair annotations" % num_annot)
        # Reduce annotations to single values using majority vote and calculate kappa agreement.
        print("... taking majority vote among annotators and calculating agreement...")
        mv_annotations = {p: {} for p in preps}  # indexed p, (aidx, bidx), single value entry.
        for p in preps:
            lfk = [[] for _ in range(len(annotators))]  # labels for kappa calculation
            for k in annotations[p]:
                if len(annotations[p][k]) != len(annotators):
                    print("...... ERROR: %s pair (%s, %s) has %d / %d annotations" %
                          (p, names[k[0]], names[k[1]], len(annotations[p][k]), len(annotators)))
                c0 = annotations[p][k].count(0)
                c1 = annotations[p][k].count(1)
                if c1 > c0:  # favor 'no' in the event of a tie
                    mv_annotations[p][k] = 1
                else:
                    mv_annotations[p][k] = 0
                for an in range(len(annotators)):
                    lfk[an].append(annotations[p][k][an])
            kappas = [cohen_kappa_score(lfk[ani], lfk[anj]) for ani in range(len(annotators) - 1)
                      for anj in range(ani + 1, len(annotators))]
            print("...... inter-annotator Cohen's kappa avg for %s: %.3f +/- %.3f" % (p, np.average(kappas),
                                                                                      np.std(kappas)))
            mvfk = [mv_annotations[p][k] for k in annotations[p]]
            kappas = [cohen_kappa_score(lfk[ani], mvfk) for ani in range(len(annotators))]
            print("...... annotator vs MV Cohen's kappa avg for %s: %.3f +/- %.3f" % (p, np.average(kappas),
                                                                                      np.std(kappas)))
        # Tie annotations to label structure.
        for p in preps:
            for fold, fold_struct, available in [["train", tr_human_o, available_train],
                                                 ["test", te_human_o, available_test]]:
                gt_found = 0
                for idx in range(len(fold_struct[p])):
                    key = (available[p][idx][1], available[p][idx][2])
                    if key in mv_annotations[p]:
                        fold_struct[p][idx] = mv_annotations[p][key]
                        gt_found += 1
                print("... done; %d / %d human execution ground truth %s labels found for %s" %
                      (gt_found, len(fold_struct[p]), fold, p))

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

    # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # bc of imagenet pretraining
    print("Preparing infrastructure to run ResNet-based feed forward model...")
    ob_to_vs = {}  # object names to list of numpy vectors resulting from representative images being run through ResNet
    resnet_m = resnet.resnet152(pretrained=True)
    plm = nn.Sequential(*list(resnet_m.children())[:-1])
    tt = ToTensor()
    v_width = None
    for idx in range(len(names)):
        img_fns = imgs[idx] if type(imgs[idx]) is list else [imgs[idx]]
        for img_fn in img_fns:
            ffn = img_fn + '.resnet'
            if os.path.isfile(ffn):
                with open(ffn, 'r') as f:
                    d = f.read().strip().split(' ')
                    v = np.array([float(n) for n in d])
                    v_width = len(v)
                    if names[idx] not in ob_to_vs:
                        ob_to_vs[names[idx]] = []
                    ob_to_vs[names[idx]].append(v)
            else:
                pil = Image.open(imgs[idx])
                pil = resize(pil, (224, 244))
                im = tt(pil)
                im = normalize(im)
                im = torch.unsqueeze(im, 0)
                v = plm(im).detach().data.numpy().flatten()
                v_width = len(v)
                if names[idx] not in ob_to_vs:
                    ob_to_vs[names[idx]] = []
                ob_to_vs[names[idx]].append(v)
                with open(ffn, 'w') as f:
                    f.write(' '.join([str(i) for i in v]))
    print("... done")

    # Average GLoVe embeddings concatenated and used to predict class.
    print("Preparing infrastructure to run GLoVe-based feed forward model...")
    ws = set()
    for p in preps:
        ws.update(word_to_i_all[p].keys())
    g, missing = get_glove_vectors(args.glove_infile, ws)
    l_width = len(g['unk'])
    print("... done; missing " + str(missing) + " vectors out of " + str(len(ws)))

    for p in preps:
        print("Preparing tensors for " + p)

        print("... preparing output tensors...")
        tr_outputs = torch.tensor(tr_o[p], dtype=torch.long).view(len(tr_o[p]), 1).numpy()
        te_outputs = torch.tensor(te_o[p], dtype=torch.long).view(len(te_o[p]), 1).numpy()
        tr_robo_outputs = torch.tensor(tr_robo_o[p], dtype=torch.long).view(len(tr_robo_o[p]), 1).numpy()
        te_robo_outputs = torch.tensor(te_robo_o[p], dtype=torch.long).view(len(te_robo_o[p]), 1).numpy()
        tr_human_outputs = torch.tensor(tr_human_o[p], dtype=torch.long).view(len(tr_human_o[p]), 1).numpy()
        te_human_outputs = torch.tensor(te_human_o[p], dtype=torch.long).view(len(te_human_o[p]), 1).numpy()
        print("...... done")

        if args.rgbd_only:
            print("... preparing RGBD inputs...")
            # Prepare input to model.
            tr_f_rgbd = [rgbd_tr[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_train[p]]
            te_f_rgbd = [rgbd_te[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_test[p]]

            # Convert structured data into tensor inputs.
            # Dimensions:
            # 5 trials (maximum) per object pair
            # 3 input channels for RGB, 1 for depth
            # 48 x 64 is the region size downsampled from the camera.
            tr_inputs_rgb = np.zeros((len(available_train[p]), 5, 3, 48, 64))
            tr_inputs_d = np.zeros((len(available_train[p]), 5, 1, 48, 64))
            te_inputs_rgb = np.zeros((len(available_test[p]), 5, 3, 48, 64))
            te_inputs_d = np.zeros((len(available_test[p]), 5, 1, 48, 64))
            for rgb_in, d_in, orig_in in [[tr_inputs_rgb, tr_inputs_d, tr_f_rgbd],
                                          [te_inputs_rgb, te_inputs_d, te_f_rgbd]]:
                for idx in range(len(orig_in)):
                    # Convert RGB and D numpy arrays to tensors.
                    rgb_in[idx] = torch.tensor(orig_in[idx][0], dtype=torch.float).numpy()
                    d_in[idx] = torch.tensor(orig_in[idx][1], dtype=torch.float).numpy()
            print("...... done")
        else:
            tr_inputs_rgb = tr_inputs_d = te_inputs_rgb = te_inputs_d = None

        print("... preparing GloVe inputs...")
        tr_inputs_l = np.zeros((len(available_train[p]), l_width * 2))
        te_inputs_l = np.zeros((len(available_test[p]), l_width * 2))
        tr_f_l = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_train[p]]
        te_f_l = [[res[oidx], res[ojdx]] for _, oidx, ojdx in available_test[p]]
        for model_in, orig_in in [[tr_inputs_l, tr_f_l], [te_inputs_l, te_f_l]]:
            for idx in range(len(orig_in)):
                # For each object, take the average GloVe embedding of all tokens used in all referring expressions
                # for that object. The representation for this pair is the concatenation of those two average embeddings
                ob1_tks = [tk for re in orig_in[idx][0] for tk in re]
                avg_ob1_v = np.sum([g[w] for w in ob1_tks], axis=0) / len(ob1_tks)
                ob2_tks = [tk for re in orig_in[idx][1] for tk in re]
                avg_ob2_v = np.sum([g[w] for w in ob2_tks], axis=0) / len(ob2_tks)
                incat = np.concatenate((avg_ob1_v, avg_ob2_v))
                model_in[idx] = torch.tensor(incat, dtype=torch.float).numpy()
        print("...... done")

        print("... preparing ResNet inputs...")
        tr_inputs_v = np.zeros((len(available_train[p]), v_width * 2))
        te_inputs_v = np.zeros((len(available_test[p]), v_width * 2))
        tr_f_v = [[oidx, ojdx] for _, oidx, ojdx in available_train[p]]
        te_f_v = [[oidx, ojdx] for _, oidx, ojdx in available_test[p]]
        for model_in, orig_in in [[tr_inputs_v, tr_f_v], [te_inputs_v, te_f_v]]:
            for idx in range(len(orig_in)):
                # For each object, take the average ResNet embedding of all the image views available for that object
                # as its representation, and represent the pair as the concatenation of those two averages.
                avg_ob1_v = np.sum(ob_to_vs[names[orig_in[idx][0]]], axis=0) / len(ob_to_vs[names[orig_in[idx][0]]])
                avg_ob2_v = np.sum(ob_to_vs[names[orig_in[idx][1]]], axis=0) / len(ob_to_vs[names[orig_in[idx][1]]])
                incat = np.concatenate((avg_ob1_v, avg_ob2_v))
                model_in[idx] = torch.tensor(incat, dtype=torch.float).numpy()
        print("...... done")

        # Write resulting vectors to json file as numpy.
        out_fn = args.out_fn_prefix + "." + p
        print("... writing tensors as numpy jsons to '" + out_fn + "' for " + p)
        with open(out_fn, 'w') as f:
            d = {"train":
                 {"mturk_label": tr_outputs.tolist(),
                  "robo_label": tr_robo_outputs.tolist(),
                  "human_label": tr_human_outputs.tolist(),
                  "lang": tr_inputs_l.tolist(),
                  "vis": tr_inputs_v.tolist(),
                  "rgb": tr_inputs_rgb.tolist() if tr_inputs_rgb is not None else None,
                  "d": tr_inputs_d.tolist() if tr_inputs_d is not None else None},
                 "test":
                 {"mturk_label": te_outputs.tolist(),
                  "robo_label": te_robo_outputs.tolist(),
                  "human_label": te_human_outputs.tolist(),
                  "lang": te_inputs_l.tolist(),
                  "vis": te_inputs_v.tolist(),
                  "rgb": te_inputs_rgb.tolist() if te_inputs_rgb is not None else None,
                  "d": te_inputs_d.tolist() if te_inputs_d is not None else None}}
            json.dump(d, f)
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
    parser.add_argument('--rgbd_only', type=int, required=True,
                        help="whether to restrict to extracting rgbd data")
    parser.add_argument('--exec_robo_indir', type=str, required=False,
                        help="input dir of robot execution ground truth affordance labels csvs")
    parser.add_argument('--exec_human_indir', type=str, required=False,
                        help="input dir of human annotator execution ground truth affordance labels csvs")
    parser.add_argument('--out_fn_prefix', type=str, required=True,
                        help="where to write the json output torch data")
    parser.add_argument('--test', type=int, required=False,
                        help="if set to 1, evaluates on the test set; NOT FOR TUNING")
    main(parser.parse_args())
