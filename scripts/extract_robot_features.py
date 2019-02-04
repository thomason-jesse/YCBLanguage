#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the raw robo features json and produces an outfile parallel in structure to the label split but with feats.
# robot dropping features json format:
# {fold:
#   {prep:
#     {"ob1": [oidx, ...]}  # Contains all pairs
#     {"ob2": [ojdx, ...]}
#     {"feats": [robot_feat_vij, ...]}  # Is set to None when the behavior hasn't been performed.
#   }
# }
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors


# Fix typos and mispellings in JSON files that store RGBD recordings of robo trials.
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
    if n[-6:] == "_bottl":
        n = n[:-6] + "_bottle"
    if n[-8:] == "_airplan":
        n = n[:-8] + "_airplane"
    if n[-7:] == "_marble":
        n = n[:-7] + "_marbles"
    if n[-7:] == "_padloc":
        n = n[:-7] + "_padlock"
    if n[-7:] == "_soccer":
        n = n[:-7] + "_soccer_ball"
    if n[-7:] == "-skille":
        n = n[:-7] + "-skillet"
    n = n.replace('philips', 'phillips')
    n = n.replace('007_rubiks', '077_rubiks')
    n = n.replace('foam_block', 'foam_brick')
    return n


def normalize_d(d):
    max_spread = max(abs(np.max(d)), abs(np.min(d)))
    d = d / max_spread  # normalize depth_delta to [-1, 1] with 0 expected no change
    d = (d + 1) / 2  # normalize depth_delta to [0, 1] with 0.5 expected no change
    return d


def depthmap_to_radial(d, pooled_origin, num_feats, grade, verbose=False):
    # avg_depth = np.average(d)
    # d -= avg_depth  # move average depth change towards 0 to normalize against camera height differences
    d = normalize_d(d)

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


def read_robo_dir(indir):
    fns = []
    for _, _, files in os.walk(indir):
        for fn in files:
            if fn.split('.')[-1] == 'json':
                fns.append(os.path.join(indir, fn))
    d = {}
    for fn in fns:
        print("Reading in robo dropping behavior data from '" + fn + "'...")
        added = 0
        with open(fn, 'r') as f:
            _d = json.load(f)
            for k in _d.keys():
                if k not in d:
                    # print("... added new key '" + k + "' with " + str(len(_d[k])) + " entries")
                    d[k] = _d[k]
                    added += 1
                else:
                    print("...... WARNING: ignoring encountered key '" + k + "' with " + str(
                        len(_d[k])) + " entries; exists with " + str(len(d[k])) + " already")
                    continue
        print("... done; added entries from " + str(added) + " keys")
    print("... total data keys: " + str(len(d)))
    return d


def main(args):

    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.splits_infile + "'...")
    with open(args.splits_infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
        folds = all_d["folds"].keys()
        preps = all_d["folds"][folds[0]].keys()
    print("... done")

    # Read in data from robo drop.
    d = read_robo_dir(args.features_indir)

    # Get training and testing data.
    feats = {f: {p: {} for p in preps} for f in folds}
    rgbd_feats = {f: {} for f in folds}  # fold, ob1, ob2, [rgb, depth] feature input maps
    for p in preps:
        print("Collating available train/test data for '" + p + "'...")
        num_pairs = {f: 0 for f in folds}
        avg_trials = {f: 0 for f in folds}
        pair_not_in_gt = {f: 0 for f in folds}
        pair_assigned_to_fold = {k: None for k in d}
        for k in d.keys():
            ob1n, ob2n = k[1:-2].split(',')
            ob1 = names.index(robo_data_ob_name_converter(ob1n.strip()))
            ob2 = names.index(robo_data_ob_name_converter(ob2n.strip()))
            k_assigned = False
            for f in folds:
                for idx in range(len(all_d["folds"][f][p]["ob1"])):
                    if ob1 == all_d["folds"][f][p]["ob1"][idx] and ob2 == all_d["folds"][f][p]["ob2"][idx]:
                        k_assigned = True
                        if pair_assigned_to_fold[k] is not None:
                            print("WARNING: pair '" + k + "' previously assigned to fold '"
                                  + pair_assigned_to_fold[k] + "' now in '" + f + "'")
                        pair_assigned_to_fold[k] = f
                        for trial in d[k].keys():
                            if ob1 not in feats[f][p]:
                                feats[f][p][ob1] = {}
                            if ob2 not in feats[f][p][ob1]:
                                feats[f][p][ob1][ob2] = []
                            verbose = False
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
                            # RGBD features.
                            if int(ob1) not in rgbd_feats[f]:
                                rgbd_feats[f][int(ob1)] = {}
                            if int(ob2) not in rgbd_feats[f][int(ob1)]:
                                rgbd_feats[f][int(ob1)][int(ob2)] = []

                            # Un-normalized distance between final and initial image in RGB and depth space.
                            # For depth, record features delta of 1 / depth between final and initial image.
                            # TODO: some kind of normalization of both RGB and depth data.
                            t1dm = np.divide(np.ones_like(t1dm), t1dm, out=np.zeros_like(t1dm), where=t1dm != 0)
                            t0dm = np.divide(np.ones_like(t0dm), t0dm, out=np.zeros_like(t0dm), where=t0dm != 0)
                            rgbd_feats[f][int(ob1)][int(ob2)].append([(t1cm - t0cm).tolist(),
                                                                      np.expand_dims(t1dm - t0dm, axis=0).tolist()]

                        num_pairs[f] += 1
                        avg_trials[f] += len(d[k].keys())
            if not k_assigned:
                # print("... WARNING: (%s, %s) in %s has RGBD data but is not in all_data struct" %
                #       (names[ob1], names[ob2], p))
                # This is fine but annoying; the split means we have some pairs that are in "on" but not in "in"
                # They're guaranteed to share a fold if they're in one of the folds, but not guaranteed to both
                # be present since we were trying to balance the classes. They're all ~mostly~ there.
                pass
        for f in folds:
            avg_trials[f] /= float(num_pairs[f]) if num_pairs[f] > 0 else 1
            print("... for '" + f + "', got " + str(num_pairs[f]) + " pairs with avg " +
                  str(avg_trials[f]) + " trials (" + str(pair_not_in_gt[f]) + " lack gt)")
    print("... done")

    # Write out extracted features in label-file parallel format.
    print("Preparing output file format and writing to '" + args.hand_features_outfile + "'...")
    out_d = {f: {p: {"ob1": all_d["folds"][f][p]["ob1"],
                     "ob2": all_d["folds"][f][p]["ob2"],
                     "feats": [feats[f][p][ob1][ob2] if ob1 in feats[f][p] and ob2 in feats[f][p][ob1] else None
                               for ob1, ob2 in zip(all_d["folds"][f][p]["ob1"], all_d["folds"][f][p]["ob2"])]
                     } for p in preps} for f in folds}
    with open(args.hand_features_outfile, 'w') as f:
        json.dump(out_d, f, indent=2)
    print("... done")

    # Write out RGBD features in object pair format.
    print("Preparing RGBD output file and writing to '" + args.rgbd_features_outfile + "'...")
    with open(args.rgbd_features_outfile, 'w') as f:
        json.dump(rgbd_feats, f)
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--features_indir', type=str, required=False,
                        help="input dir containing input jsons from rosbag processing")
    parser.add_argument('--hand_features_outfile', type=str, required=True,
                        help="output json mapping object pairs to dropping behavior hand-crafted feature vectors")
    parser.add_argument('--rgbd_features_outfile', type=str, required=True,
                        help="output json mapping object pairs to their rgbd features")
    main(parser.parse_args())
