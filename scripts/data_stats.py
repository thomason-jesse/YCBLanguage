#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json


def main(args):

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.infile + "'...")
    with open(args.infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        lf = all_d["folds"]
        folds = lf.keys()
        preps = lf[folds[0]].keys()
    print("... done")

    # Read in metadata.
    total_annotated_pairs = {p: None for p in preps}
    print("Reading in metadata from '" + args.metadata_infile + "'...")
    with open(args.metadata_infile, 'r') as f:
        d = json.load(f)
        for p in preps:
            total_annotated_pairs[p] = sum([len(d["votes"][p][oidx].keys()) for oidx in d["votes"][p].keys()])
    print("... done")

    # Kept after labeling.
    num_labeled = {p: sum([len(all_d["folds"][f][p]["ob1"]) for f in folds]) for p in preps}
    print("After labeling and splitting, kept:")
    for p in preps:
        print("\t" + p + ":\t%.2f" % (num_labeled[p] / float(total_annotated_pairs[p])) + "\t"
              + str(num_labeled[p]) + "/" + str(total_annotated_pairs[p]))

    # Fold distributions.
    print("Distribution by fold:")
    size_folds = {p: {f: len(all_d["folds"][f][p]["ob1"]) for f in folds} for p in preps}
    for p in preps:
        print("\t" + p + ":")
        for f in folds:
            print("\t\t" + f + ":\t%.2f" % (size_folds[p][f] / float(num_labeled[p])) + "\t"
                  + str(size_folds[p][f]) + "/" + str(num_labeled[p]))

    # Label distribution per fold.
    print("Label distribution by fold:")
    size_folds = {p: {f: len(all_d["folds"][f][p]["ob1"]) for f in folds} for p in preps}
    wood_ball_pair = None
    cube_cube_pair = None
    for p in preps:
        print("\t" + p + ":")
        for f in folds:
            print("\t\t" + f + ":")
            for l in range(0, 3):
                print ("\t\t\t%.2f" % (all_d["folds"][f][p]["label"].count(l) / float(size_folds[p][f])) +
                       "\t" + str(all_d["folds"][f][p]["label"].count(l)) + "/" + str(size_folds[p][f]))

                # Check for val pairs.
                for idx in range(len(all_d["folds"][f][p]["label"])):
                    if (all_d["folds"][f][p]["ob1"][idx] == names.index("036_wood_block") and
                            all_d["folds"][f][p]["ob2"][idx] == names.index("053_mini_soccer_ball")):
                        wood_ball_pair = (p, f, l)
                    elif (all_d["folds"][f][p]["ob1"][idx] == names.index("077_rubiks_cube") and
                          all_d["folds"][f][p]["ob2"][idx] == names.index("077_rubiks_cube")):
                        cube_cube_pair = (p, f, l)

    if wood_ball_pair is not None:
        print("val question 00 appears in set " + str(wood_ball_pair))
    if cube_cube_pair is not None:
        print("val question 36 appears in set " + str(cube_cube_pair))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    main(parser.parse_args())
