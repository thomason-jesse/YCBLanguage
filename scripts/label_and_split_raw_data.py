#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the raw dataset json and produces labels, then a class-balanced train/dev/test split written out as json.
# labeled fold json format:
# {"names": [obj1, obj2, ...],
# {"folds":
#   {fold:
#     {prep:
#       {"ob1": [oidx, ...}
#       {"ob2": [ojdx, ...}
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
        names = all_d["names"]
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
        for soidx in d:
            oidx = int(soidx)
            for sojdx in d[soidx]:
                ojdx = int(sojdx)

                # With 'majority', assign classes:
                #  0 : majority of annotator votes were -1 ("no")
                #  1 : majority of annotator votes were 1 ("yes")
                #  Does not include object pairs with no majority class in the labeled folds.
                to_label = None
                if args.label_technique == 'majority':
                    s = sum(d[soidx][sojdx])
                    if s < 0:
                        to_label = 0
                    elif s > 0:
                        to_label = 1

                # With '3class', assign classes:
                #  0 : all annotator votes were -1 ("no")
                #  1 : annotator votes were not uniform
                #  2 : all annotator votes were 1 ("yes")
                if args.label_technique == '3class':
                    s = sum(d[soidx][sojdx])
                    if s == -len(d[soidx][sojdx]):
                        to_label = 0
                    elif s == len(d[soidx][sojdx]):
                        to_label = 2
                    else:
                        to_label = 1

                # With 'full', assign classes:
                #  0 : three annotators voted -1 ("no")
                #  1 : two annotators voted -1 ("no")
                #  2 : two annotators voted 1 ("yes")
                #  3 : three annotators voted 1 ("yes")
                #  Does not include object pairs with fewer than 3 annotations.
                #  When a pair has four annotations, the dissenting annotator, if any, is ignored.
                #  Does not include pairs with four annotations and an even vote split.
                if args.label_technique == 'full':
                    s = sum(d[soidx][sojdx])
                    if len(d[soidx][sojdx]) == 4:
                        if s <= -3:  # one dissenting "yes" or agreement
                            to_label = 0
                        elif s >= 3:  # one dissenting "no" or agreement
                            to_label = 3
                    elif len(d[soidx][sojdx]) == 3:
                        to_label = (s + 3) // 2

                if to_label is not None:
                    if oidx not in lbs[p]:
                        lbs[p][oidx] = {}
                    lbs[p][oidx][ojdx] = to_label
    print("... done")

    # Show class breakdown for whole dataset.
    all_d_pairs = {p: [(oidx, ojdx) for oidx in prep_d[p] for ojdx in prep_d[p][oidx]] for p in prep}
    target_dist = {}
    for p in prep_d:
        d = prep_d[p]
        skipped = 0
        print("Class distribution for " + p + ":")
        s = [0] * len(cr)
        for soidx in d:
            oidx = int(soidx)
            for sojdx in d[soidx]:
                ojdx = int(sojdx)
                if oidx in lbs[p] and ojdx in lbs[p][oidx]:
                    s[lbs[p][oidx][ojdx]] += 1
                else:
                    skipped += 1
        for c in cr:
            print("\t" + str(c) + ":\t" + str(s[c]) + "\t(%0.2f" % (s[c] / float(len(all_d_pairs[p]))) + ")")
        print("\tSkip:\t" + str(skipped) + "\t(%0.2f" % (skipped / float(len(all_d_pairs[p]))) + ")")
        target_dist[p] = [s[c] / float(len(all_d_pairs[p]) - skipped) for c in cr]

    # Do train/dev/test split that tries to match each fold's label distribution with global.
    print("Performing train/dev/test split while preserving class distribution...")
    folds = [('train', 0.8), ('dev', 0.1), ('test', 0.1)]
    lf = {fold: {p: {"ob1": [], "ob2": [], "label": []} for p in prep} for fold, _ in folds}
    cc = {fold: {p: [0] * len(cr) for p in prep} for fold, _ in folds}
    all_pairs = {p: [(oidx, ojdx) for oidx in lbs[p] for ojdx in lbs[p][oidx]] for p in prep}
    pairs_kept = {p: [] for p in prep}

    # Each object is assigned to one fold, and its resulting pairs are committed to that fold only if
    # the paired objects are both in the same fold. This results in a block diagonal matrix of usable pairs
    # in the final dataset, with blocks of "train" pairs, "test" pairs, and "dev" pairs, since pairs of objects
    # that are labeled but appear in different folds are excluded.
    rnames = names[:]
    np.random.shuffle(rnames)
    fns = {fold: set() for fold, _ in folds}
    for n in rnames:
        f_avg_rel_dist = [np.average([abs(prop - sum(cc[fold][p]) / float(len(all_pairs[p]))) / prop
                                      for p in prep]) for fold, prop in folds]
        fidx = f_avg_rel_dist.index(max(f_avg_rel_dist))  # next fold to populate
        fold = folds[fidx][0]
        prop = folds[fidx][1]
        nidx = names.index(n)
        for p in lbs:
            for oidx in lbs[p]:
                if oidx == nidx or oidx in fns[fold]:
                    for ojdx in lbs[p][oidx]:
                        if nidx == ojdx or ojdx in fns[fold]:
                            if oidx == nidx or ojdx == nidx:
                                if cc[fold][p][lbs[p][oidx][ojdx]] / float(prop * len(all_pairs[p])) \
                                        < target_dist[p][lbs[p][oidx][ojdx]]:
                                    cc[fold][p][lbs[p][oidx][ojdx]] += 1
                                    lf[fold][p]["ob1"].append(oidx)
                                    lf[fold][p]["ob2"].append(ojdx)
                                    lf[fold][p]["label"].append(lbs[p][oidx][ojdx])
                                    pairs_kept[p].append((oidx, ojdx))
                                    fns[fold].add(nidx)  # include this object in the fold since we decided we needed at least one of its pairs

    # Assert that no object appears across folds even in different tasks.
    for fold, _ in folds:
        for p in lbs:
            for of, _ in folds:
                if of != fold:
                    for op in prep:
                        for oidx in lf[fold][p]["ob1"]:
                            assert oidx not in lf[of][op]["ob1"] and oidx not in lf[of][op]["ob2"]
                        for oidx in lf[fold][p]["ob2"]:
                            assert oidx not in lf[of][op]["ob1"] and oidx not in lf[of][op]["ob2"]
    print("... done")

    # Show included/excluded stats.
    for p in lbs:
        print("'" + p + "': Included " + str(len(pairs_kept[p])) + " of " + str(len(all_pairs[p])) +
              " available pairs (%0.2f" % (len(pairs_kept[p]) / float(len(all_pairs[p]))) + ")")

    # Show class breakdown per fold.
    for fold, prop in folds:
        print(fold + " dist:")
        for p in lbs:
            print("\t" + p + ": " + str(sum(cc[fold][p])) +
                  "\t(%0.2f" % (sum(cc[fold][p]) / float(len(pairs_kept[p]))) + ")")
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
