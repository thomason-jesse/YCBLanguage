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
        test = lf['test']
        dev = lf['dev']
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
        rgbd_te = rgbd['test']
        rgbd_dv = rgbd['dev']
    print("... done")

    # Select subset of data on which to evaluate.
    print("Selecting evaluation data...")
    available_rgbd_train = {p: None for p in preps}
    available_rgbd_dev = {p: None for p in preps}
    available_rgbd_test = {p: None for p in preps}
    available_all_train = {p: None for p in preps}
    available_all_dev = {p: None for p in preps}
    available_all_test = {p: None for p in preps}
    for p in preps:
        available_rgbd_train[p] = [[train[p]["ob1"][ix], train[p]["ob2"][ix]]
                              for ix in range(len(train[p]["ob1"]))
                              if str(train[p]["ob1"][ix]) in rgbd_tr and
                              str(train[p]["ob2"][ix]) in rgbd_tr[str(train[p]["ob1"][ix])]]
        available_rgbd_dev[p] = [[dev[p]["ob1"][ix], dev[p]["ob2"][ix]]
                             for ix in range(len(dev[p]["ob1"]))
                             if str(dev[p]["ob1"][ix]) in rgbd_dv and
                             str(dev[p]["ob2"][ix]) in rgbd_dv[str(dev[p]["ob1"][ix])]]
        available_rgbd_test[p] = [[test[p]["ob1"][ix], test[p]["ob2"][ix]]
                             for ix in range(len(test[p]["ob1"]))
                             if str(test[p]["ob1"][ix]) in rgbd_te and
                             str(test[p]["ob2"][ix]) in rgbd_te[str(test[p]["ob1"][ix])]]
        print("... done; %d / %d available training, %d / %d available dev; and %d / %d available testing examples with RGBD data for %s" %
              (len(available_rgbd_train[p]), len(train[p]["ob1"]),
                len(available_rgbd_dev[p]), len(dev[p]["ob1"]),
                len(available_rgbd_test[p]), len(test[p]["ob1"]), p))
        available_all_train[p] = [[train[p]["ob1"][ix], train[p]["ob2"][ix]]
                              for ix in range(len(train[p]["ob1"]))]
        available_all_dev[p] = [[dev[p]["ob1"][ix], dev[p]["ob2"][ix]]
                             for ix in range(len(dev[p]["ob1"]))]
        available_all_test[p] = [[test[p]["ob1"][ix], test[p]["ob2"][ix]]
                             for ix in range(len(test[p]["ob1"]))]
        print("... done; %d / %d available training; %d / %d available dev; %d / %d available testing examples for %s" %
              (len(available_all_train[p]), len(train[p]["ob1"]),
                len(available_all_dev[p]), len(dev[p]["ob1"]),
                len(available_all_test[p]), len(test[p]["ob1"]), p))
    
    out_fn = args.out_fn
    print("... writing CSVs to '" + out_fn + "'")
    with open(out_fn, 'w') as f:
        headers = ["fold", "a", "b", "has_in_rgbd", "has_on_rgbd"]
        f.write('\t'.join(headers) + "\n")
        for fold, rgbd_d, all_d in [["train", available_rgbd_train, available_all_train],
                                    ["dev", available_rgbd_dev, available_all_dev],
                                    ["test", available_rgbd_test, available_all_test]]:
            pairs = set([(a_idx, b_idx) for a_idx, b_idx in all_d["on"]])
            pairs = pairs.union(set([(a_idx, b_idx) for a_idx, b_idx in all_d["in"]]))
            for a_idx, b_idx in pairs:
                has_in_rgbd = True if [a_idx, b_idx] in available_rgbd_test["in"] else False
                has_on_rgbd = True if [a_idx, b_idx] in available_rgbd_test["on"] else False
                f.write('\t'.join([fold, names[a_idx], names[b_idx], str(has_in_rgbd), str(has_on_rgbd)]) + '\n')
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    parser.add_argument('--robot_infile', type=str, required=True,
                        help="input robot feature file")
    parser.add_argument('--out_fn', type=str, required=True,
                        help="where to write the csv annotation file")
    main(parser.parse_args())
