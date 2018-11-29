#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the raw robo features json and produces an outfile parallel in structure to the label split but with feats.
# robot dropping features json format:
# {fold:
#   {prep:
#     {"ob1": [oidx, ...]}  # Contains all pairs
#     {"ob2": [ojdx, ...]}
#     {"feats": [robot_feat_vij, ...]}  # Is set to None when the behavior hasn't been performed.
#     {"label": [lij, ...]}  # from robo ground truth annotations, not MTurk
#   }
# }
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import colors


def robo_data_ob_name_converter(n):
    if n[0] != '0':
        n = '0' + n
    if n[-3:] == "cup":
        n += 's'
    if n[-5:] == "-cups":
        n = n[:-5] + "_cups"
    if n[-3:] == "_ca":
        n = n[:-3] + "_can"
    if n[-3:] == "_bo":
        n = n[:-3] + "_box"
    if n[-3:] == "_ke":
        n = n[:-3] + "_key"
    if n[-4:] == "_tes":
        n = n[:-4] + "_test"
    if n[-5:] == "_plat":
        n = n[:-5] + "_plate"
    if n[-5:] == "_clam":
        n = n[:-5] + "_clamp"
    if n[-5:] == "_glas":
        n = n[:-5] + "_glass"
    if n[-8:] == "_softbal":
        n = n[:-8] + "_softball"
    if n[-11:] == "_screwdrive":
        n = n[:-11] + "_screwdriver"
    if n[-5:] == "_dupl":
        n = n[:-5] + "_duplo"
    if n[-6:] == "_orang":
        n = n[:-6] + "_orange"
    if n[-8:] == "_airplan":
        n = n[:-8] + "_airplane"
    if n[-7:] == "_marble":
        n = n[:-7] + "_marbles"
    if n[-7:] == "_padloc":
        n = n[:-7] + "_padlock"
    if n[-7:] == "_soccer":
        n = n[:-7] + "_soccer_ball"
    n = n.replace('philips', 'phillips')
    return n


def depthmap_to_radial(d, pooled_origin, num_feats, grade, verbose=False):
    # avg_depth = np.average(d)
    # d -= avg_depth  # move average depth change towards 0 to normalize against camera height differences
    max_spread = max(abs(np.max(d)), abs(np.min(d)))
    d = d / max_spread  # normalize depth_delta to [-1, 1] with 0 expected no change
    d = (d + 1) / 2  # normalize depth_delta to [0, 1] with 0.5 expected no change

    # Visualize raw depth changes.
    if verbose:
        cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                      for cidx in range(grade)])
        bounds = range(grade)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow([[d[xidx, yidx] * grade
                    for yidx in range(d.shape[1])]
                   for xidx in range(d.shape[0])], cmap=cmap, norm=norm)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off',
                        labelbottom='off', labeltop='off')
        plt.show()

    # Extract radial features from maps.
    # First, assign each cell to a feature or to the edge of the table.
    # To do this, normalize each cell's (x_idx, y_idx) index into an (x,y) \in [0,1].
    # The feature bands can then be defined as distances from the origin in the unit circle.
    cells_by_feature = [[] for _ in range(num_feats + 1)]
    # Band distances cap at 0.5 because that's the distance between the center of the image and the border in the
    # case that Object B was exactly centered in the image.
    band_upper_limits = [0.5 * (fidx + 1) / float(num_feats) for fidx in range(num_feats)]
    cell_to_feature = {}
    for xidx in range(d.shape[0]):
        x = (pooled_origin[0] - xidx) / float(d.shape[0])  # normalize to [-1, 1]
        for yidx in range(d.shape[1]):
            y = (pooled_origin[1] - yidx) / float(d.shape[1])  # normalize to [-1, 1]
            dist = np.linalg.norm([x, y])
            feat = None
            for fidx in range(num_feats):
                if dist < band_upper_limits[fidx]:
                    feat = fidx
                    break
            if feat is None:
                feat = num_feats
            cells_by_feature[feat].append([xidx, yidx])
            cell_to_feature[(xidx, yidx)] = feat

    # Take the average depth change per band.
    stddev = np.std([d[xidx, yidx] for yidx in range(d.shape[1]) for xidx in range(d.shape[0])])
    gavg = np.average([d[xidx, yidx] for yidx in range(d.shape[1]) for xidx in range(d.shape[0])])
    features = [1 if np.average([d[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]]) > gavg + stddev else (1 if np.average([d[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]]) < gavg - stddev else 0)
                for fidx in range(num_feats + 1)]
    return features, cells_by_feature, cell_to_feature, stddev, gavg


# Given depth map at time 0, d0, and time 1, d1, extract num_feats features describing the difference in these
# depth maps. For each feature, a radial band around the origin is constructed, and the feature is the average
# depth change between time 0 and time 1 in that band. If num_feats=1, the average is over the entire depth map.
# If num_feats>1, bands extend outwards from the origin, evenly spaced by radius to the border. Cells closer to
# the border than to a radial band centerline will be discarded (e.g., cells outside the normalized unit circle).
def depthmaps_to_features(raw_origin, num_feats, verbose=False,
                          rad_before=False, d0=None, d1=None, d=None):
    assert (rad_before and d0 is not None and d1 is not None) or (not rad_before and d is not None)

    grade = 255  # for verbose, how many distinct shades of to use (lower means higher contrast)
    pooled_origin = [dim / 10. for dim in raw_origin]
    pooled_origin = [pooled_origin[1], pooled_origin[0]]  # (coord x,y have to be swapped; Rosario might fix)

    if rad_before:
        t1dm = np.asmatrix(d1)
        t0dm = np.asmatrix(d0)
        d = t1dm - t0dm
        features_d0, _, _, _, _ = depthmap_to_radial(t0dm, pooled_origin, num_feats, grade, verbose=verbose)
        features_d1, cells_by_feature, cell_to_feature, _, _ = depthmap_to_radial(t1dm, pooled_origin, num_feats, grade,
                                                                            verbose=verbose)
        features = [0.5 + features_d1[fidx] - features_d0[fidx] for fidx in range(num_feats + 1)]
    else:
        features, cells_by_feature, cell_to_feature, stddev, gavg = depthmap_to_radial(d, pooled_origin, num_feats,
                                                                                       grade, verbose=verbose)

    # Visualize average depth changes.
    if verbose:
        if verbose:
            print("cells_by_feature:")
            print("\t" + "\n\t".join([str(fidx) + ":\t" + str(len(cells_by_feature[fidx])) + "\t" + str(np.min([d[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]])) + "\t%.2f" % np.average([d[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]]) + "\t" + str(np.max([d[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]]))
                                      for fidx in range(num_feats)]))
            print("cells discarded:\t" + str(len(cells_by_feature[num_feats])))
            print("features: " + str(features[:-1]))

        cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                      for cidx in range(grade)])
        bounds = range(grade)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow([[features[cell_to_feature[(xidx, yidx)]] * grade
                   for yidx in range(d.shape[1])]
                   for xidx in range(d.shape[0])], cmap=cmap, norm=norm)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off',
                        labelbottom='off', labeltop='off')
        plt.show()

    return features[:-1]


def main(args):
    assert args.features_infile is not None or args.features_indir is not None

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.splits_infile + "'...")
    with open(args.splits_infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        folds = all_d["folds"].keys()
        preps = all_d["folds"][folds[0]].keys()
    print("... done")

    # Read in data from robo drop.
    if args.features_indir is None:
        fns = [args.features_infile]
    else:
        fns = []
        for _, _, files in os.walk(args.features_indir):
            for fn in files:
                if fn.split('.')[-1] == 'json':
                    fns.append(os.path.join(args.features_indir, fn))
    d = {}
    for fn in fns:
        print("Reading in robo dropping behavior data from '" + fn + "'...")
        with open(fn, 'r') as f:
            _d = json.load(f)
            for k in _d.keys():
                if k not in d:
                    # print("... added new key '" + k + "' with " + str(len(_d[k])) + " entries")
                    d[k] = _d[k]
                else:
                    # print("... extending existing key '" + k + "' with " + str(len(_d[k])) + " entries")
                    d[k].extend(_d[k])
        print("... done; added entries from " + str(len(_d.keys())) + " keys")
    print("... total data keys: " + str(len(d)))

    # Read in gt labels from csv.
    print("Reading in robo ground truth labels from '" + args.gt_infile + "'...")
    df = pd.read_csv(args.gt_infile)
    gt_labels = {p: {} for p in preps}
    l2c = {"Y": 1, "N": 0}  # toss out 'M' data; note this means Y is class 1 not class 2
    for idx in df.index:
        k = '(' + ', '.join(df["pair"][idx].split(' + ')) + ')'
        for p in preps:
            if df[p][idx] in l2c:
                gt_labels[p][k] = l2c[df[p][idx]]
        if k in gt_labels["in"] and gt_labels["in"][k] == 1:  # DEBUG: test whether 'in'/'on' exclusivity helps
            gt_labels["on"][k] = 0  # DEBUG
    print("... done")

    # Get training and testing data.
    feats = {f: {p: {} for p in preps} for f in folds}
    labels = {f: {p: {} for p in preps} for f in folds}
    for p in preps:
        print("Collating available train/test data for '" + p + "'...")
        num_pairs = {f: 0 for f in folds}
        avg_trials = {f: 0 for f in folds}
        pair_not_in_gt = {f: 0 for f in folds}
        pair_assigned_to_fold = {k: None for k in d}
        for k in d.keys():
            if k in gt_labels[p]:
                gt = gt_labels[p][k]
            else:
                gt = None  # get from all_d
            ob1n, ob2n = k[1:-2].split(',')
            ob1 = names.index(robo_data_ob_name_converter(ob1n.strip()))
            ob2 = names.index(robo_data_ob_name_converter(ob2n.strip()))
            for f in folds:
                for idx in range(len(all_d["folds"][f][p]["ob1"])):
                    if ob1 == all_d["folds"][f][p]["ob1"][idx] and ob2 == all_d["folds"][f][p]["ob2"][idx]:
                        if pair_assigned_to_fold[k] is not None:
                            print("WARNING: pair '" + k + "' previously assigned to fold '"
                                  + pair_assigned_to_fold[k] + "' now in '" + f + "'")
                        pair_assigned_to_fold[k] = f
                        for trial in d[k].keys():
                            if ob1 not in feats[f][p]:
                                feats[f][p][ob1] = {}
                                labels[f][p][ob1] = {}
                            if ob2 not in feats[f][p][ob1]:
                                feats[f][p][ob1][ob2] = []
                            verbose = False
                            # if k == "(016_pear, 065-a_cups)" or k == "(052_extra_large_clamp, 065-i_cups)":
                            #     verbose = True  # DEBUG  # Figure for examples
                            # if gt_labels["on"][k] == 0 and gt_labels["in"][k] == 0:
                            #     verbose = True  # DEBUG
                            # Examples for cherry/lemon
                            # if (k == "(011_banana, 065-e_cups)" or  # MM says 1, robo says 0 (correct)
                            #         k == "(063-f_marbles, 003_cracker_box)" or  # MM says 2, robo says 0 (correct)
                            #         k == "(065-e_cups, 023_wine_glass)" or  # Robot says 2 (correct), MM 0
                            #         k == "(004_sugar_box, 065-e_cups)"):  # Robot says 2, MM 0 (correct)
                            #     verbose = True
                            if verbose:
                                print(k)  # DEBUG

                            # Depth features
                            t1dm = np.asmatrix(d[k][trial]['t1_depthmap'])
                            t0dm = np.asmatrix(d[k][trial]['t0_depthmap'])
                            dd = t1dm - t0dm
                            feats[f][p][ob1][ob2].append(depthmaps_to_features(d[k][trial]['center_point'],
                                                                               15, verbose=verbose,
                                                                               rad_before=False, d=dd))

                            # RGB features
                            t1cm = np.asarray(d[k][trial]['t1_rgbmap'])
                            t0cm = np.asarray(d[k][trial]['t0_rgbmap'])
                            dc = np.linalg.norm(t1cm - t0cm, axis=0)
                            feats[f][p][ob1][ob2].append(depthmaps_to_features(d[k][trial]['center_point'],
                                                                               15, verbose=verbose,
                                                                               rad_before=False, d=dc))

                            if gt is None:
                                pair_not_in_gt[f] += 1
                                gt = all_d["folds"][f][p]["label"][idx]
                            labels[f][p][ob1][ob2] = gt
                        num_pairs[f] += 1
                        avg_trials[f] += len(d[k].keys())
        for f in folds:
            avg_trials[f] /= float(num_pairs[f]) if num_pairs[f] > 0 else 1
            print("... for '" + f + "', got " + str(num_pairs[f]) + " pairs with avg " +
                  str(avg_trials[f]) + " trials (" + str(pair_not_in_gt[f]) + " lack gt)")
    print("... done")

    # Write out extracted features in label-file parallel format.
    print("Preparing output file format and writing to '" + args.features_outfile + "'...")
    out_d = {f: {p: {"ob1": all_d["folds"][f][p]["ob1"],
                     "ob2": all_d["folds"][f][p]["ob2"],
                     "feats": [feats[f][p][ob1][ob2] if ob1 in feats[f][p] and ob2 in feats[f][p][ob1] else None
                               for ob1, ob2 in zip(all_d["folds"][f][p]["ob1"], all_d["folds"][f][p]["ob2"])],
                     "label": [labels[f][p][ob1][ob2] if ob1 in labels[f][p] and ob2 in labels[f][p][ob1] else None
                                for ob1, ob2 in zip(all_d["folds"][f][p]["ob1"], all_d["folds"][f][p]["ob2"])]
                     } for p in preps} for f in folds}
    with open(args.features_outfile, 'w') as f:
        json.dump(out_d, f, indent=2)
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--features_infile', type=str, required=False,
                        help="input json from rosbag processing")
    parser.add_argument('--features_indir', type=str, required=False,
                        help="input dir containing input jsons from rosbag processing")
    parser.add_argument('--gt_infile', type=str, required=True,
                        help="input csv of ground truth affordance labels")
    parser.add_argument('--features_outfile', type=str, required=True,
                        help="output json mapping object pairs to dropping behavior feature vectors")
    main(parser.parse_args())
