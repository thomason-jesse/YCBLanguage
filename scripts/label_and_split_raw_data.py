#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the raw dataset json and produces labels, then a class-balanced train/dev/test split written out as json.
# labeled fold json format:
# {"names": [obj1, obj2, ...],
# {"folds":
#   {fold:
#     {prep:
#       {"pair":  [obji+objj, ...]}
#       {"label": [lij, ...]}
#     }
#   }
# }

import argparse
import numpy as np
import json


def main(args):
    assert args.label_technique in ['majority', '3class', 'full']

    # Read in raw dataset.
    print("Loading raw data from '" + args.infile + "'...")
    with open(args.infile, 'r') as f:
        all_d = json.load(f)
        prep_d = all_d["votes"]
        prep = prep_d.keys()
    print("... done")

    # Create labels for each pair based on votes.
    print("Adding class labels with '" + args.label_technique + "' technique")
    lbs = {p: {} for p in prep}
    cr = range(2) if args.label_technique == 'majority' else (range(3) if args.label_technique == '3class' else
                                                              range(4) if args.label_technique == 'full' else None)
    for p in prep_d:
        d = prep_d[p]
        for k in d:

            # With 'majority', assign classes:
            #  0 : majority of annotator votes were -1 ("no")
            #  1 : majority of annotator votes were 1 ("yes")
            #  Does not include object pairs with no majority class in the labeled folds.
            if args.label_technique == 'majority':
                s = sum(d[k])
                if s < 0:
                    lbs[p][k] = 0
                elif s > 0:
                    lbs[p][k] = 1

            # With '3class', assign classes:
            #  0 : all annotator votes were -1 ("no")
            #  1 : annotator votes were not uniform
            #  2 : all annotator votes were 1 ("yes")
            if args.label_technique == '3class':
                s = sum(d[k])
                if s == -len(d[k]):
                    lbs[p][k] = 0
                elif s == len(d[k]):
                    lbs[p][k] = 2
                else:
                    lbs[p][k] = 1

            # With 'full', assign classes:
            #  0 : three annotators voted -1 ("no")
            #  1 : two annotators voted -1 ("no")
            #  2 : two annotators voted 1 ("yes")
            #  3 : three annotators voted 1 ("yes")
            #  Does not include object pairs with fewer than 3 annotations.
            #  When a pair has four annotations, the dissenting annotator, if any, is ignored.
            #  Does not include pairs with four annotations and an even vote split.
            if args.label_technique == 'full':
                s = sum(d[k])
                if len(d[k]) == 4:
                    if s <= -3:  # one dissenting "yes" or agreement
                        lbs[p][k] = 0
                    elif s >= 3:  # one dissenting "no" or agreement
                        lbs[p][k] = 3
                elif len(d[k]) == 3:
                    lbs[p][k] = (s + 3) // 2
    print("... done")

    # Show class breakdown for whole dataset.
    target_dist = {}
    for p in prep_d:
        d = prep_d[p]
        skipped = 0
        print("Class distribution for " + p + ":")
        s = [0] * len(cr)
        for k in d:
            if k in lbs[p]:
                s[lbs[p][k]] += 1
            else:
                skipped += 1
        for c in cr:
            print("\t" + str(c) + ":\t" + str(s[c]) + "\t(%0.2f" % (s[c] / float(len(d))) + ")")
        print("\tSkip:\t" + str(skipped) + "\t(%0.2f" % (skipped / float(len(d))) + ")")
        target_dist[p] = [s[c] / float(len(d) - skipped) for c in cr]

    # Do train/dev/test split that tries to match each fold's label distribution with global.
    print("Performing train/dev/test split while preserving class distribution...")
    folds = [('train', 0.8), ('dev', 0.1), ('test', 0.1)]
    lf = {fold: {p: {"pair": [], "label": []} for p in prep} for fold, _ in folds}
    cc = {fold: {p: [0] * len(cr) for p in prep} for fold, _ in folds}
    for p in lbs:
        rkeys = list(lbs[p].keys())[:]
        np.random.shuffle(rkeys)
        for k in rkeys:
            for fold, prop in folds:
                if sum(cc[fold][p]) / float(len(rkeys)) < prop:  # this fold needs more
                    if cc[fold][p][lbs[p][k]] / float(prop * len(rkeys)) \
                            < target_dist[p][lbs[p][k]]:  # this class needs more
                        cc[fold][p][lbs[p][k]] += 1
                        lf[fold][p]["pair"].append(k)
                        lf[fold][p]["label"].append(lbs[p][k])
    print("... done")

    # Show class breakdown per fold.
    for fold, prop in folds:
        print(fold + " dist:")
        for p in lbs:
            print("\t" + p + ": " + str(sum(cc[fold][p])) + "\t(%0.2f" % (sum(cc[fold][p]) / float(len(lbs[p]))) + ")")
            for c in cr:
                print("\t\t" + str(c) + ":\t" + str(cc[fold][p][c]) +
                      "\t(%0.2f" % (cc[fold][p][c] / float(sum(cc[fold][p]))) +
                      ") of target (%0.2f" % target_dist[p][c] + ")")

    # Write outfile.
    print("Writing labeled folds to '" + args.outfile + "'...")
    with open(args.outfile, 'w') as f:
        json.dump({"names": all_d["names"], "folds": lf}, f)
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input dataset json file")
    parser.add_argument('--label_technique', type=str, required=True,
                        help="either 'majority', '3class', or 'full'")
    parser.add_argument('--outfile', type=str, required=True,
                        help="the file to write the json labeled folds to")
    main(parser.parse_args())
