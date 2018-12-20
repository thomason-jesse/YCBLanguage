#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
import pandas as pd
from PIL import Image
from models import *
from torchvision.models import resnet
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from utils import *


def main(args):

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
                print("... WARNING: using human annotations for remaining labels (TODO: annotate remainder of data)")

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
        print("...... done")

        if args.rgbd_only:
            print("... preparing RGBD inputs...")
            # Prepare input to model.
            tr_f_rgbd = [rgbd_tr[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_train[p]]
            te_f_rgbd = [rgbd_te[str(oidx)][str(ojdx)] for _, oidx, ojdx in available_test[p]]

            # Convert structured data into tensor inputs.
            # TODO: should probably read these sizes out of the file, but they're very fixed to this data.
            tr_inputs_rgb = np.zeros((len(available_train[p]), 1, 3, 48, 64))
            tr_inputs_d = np.zeros((len(available_train[p]), 1, 1, 48, 64))
            te_inputs_rgb = np.zeros((len(available_test[p]), 1, 3, 48, 64))
            te_inputs_d = np.zeros((len(available_test[p]), 1, 1, 48, 64))
            for rgb_in, d_in, orig_in in [[tr_inputs_rgb, tr_inputs_d, tr_f_rgbd],
                                          [te_inputs_rgb, te_inputs_d, te_f_rgbd]]:
                for idx in range(len(orig_in)):
                    # Convert RGB and D numpy arrays to tensors and add batch dimension at axis 0.
                    rgb_in[idx] = torch.tensor(orig_in[idx][0], dtype=torch.float).unsqueeze(0).numpy()
                    d_in[idx] = torch.tensor(orig_in[idx][1], dtype=torch.float).unsqueeze(0).numpy()
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
        out_fn = args.out_fn + "." + p
        print("... writing tensors as numpy jsons to '" + out_fn + "' for " + p)
        with open(out_fn, 'w') as f:
            d = {"train":
                 {"label": tr_outputs.tolist(),
                  "lang": tr_inputs_l.tolist(),
                  "vis": tr_inputs_v.tolist(),
                  "rgb": tr_inputs_rgb.tolist() if tr_inputs_rgb is not None else None,
                  "d": tr_inputs_d.tolist() if tr_inputs_d is not None else None},
                 "test":
                 {"label": te_outputs.tolist(),
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
    parser.add_argument('--out_fn', type=str, required=True,
                        help="where to write the json output torch data")
    parser.add_argument('--test', type=int, required=False,
                        help="if set to 1, evaluates on the test set; NOT FOR TUNING")
    parser.add_argument('--gt_infile', type=str, required=False,
                        help="input csv of ground truth affordance labels; if provided, overrides dev/test labels")
    main(parser.parse_args())
