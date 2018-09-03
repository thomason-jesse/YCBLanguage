#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the raw robo features json and produces an outfile parallel in structure to the label split but with feats.
# robot dropping features json format:
# {fold:
#   {prep:
#     {"ob1": [oidx, ...]}  # Contains all pairs
#     {"ob2": [ojdx, ...]}
#     {"feats": [robot_feat_vij, ...]}  # Is set to None when the behavior hasn't been performed.
#     {"label": [lij, ...]}
#   }
# }
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
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
    return n


# Given depth map at time 0, d0, and time 1, d1, extract num_feats features describing the difference in these
# depth maps. For each feature, a radial band around the origin is constructed, and the feature is the average
# depth change between time 0 and time 1 in that band. If num_feats=1, the average is over the entire depth map.
# If num_feats>1, bands extend outwards from the origin, evenly spaced by radius to the border. Cells closer to
# the border than to a radial band centerline will be discarded (e.g., cells outside the normalized unit circle).
def depthmaps_to_features(d0, d1, raw_origin, num_feats, verbose=False):
    t1dm = np.asmatrix(d1)
    t0dm = np.asmatrix(d0)
    depth_delta = t1dm - t0dm
    avg_depth = np.average(depth_delta)
    depth_delta -= avg_depth  # move average depth change towards 0 to normalize against camera height differences
    max_spread = max(abs(np.max(depth_delta)), abs(np.min(depth_delta)))
    depth_delta = depth_delta / max_spread  # normalize depth_delta to [-1, 1] with 0 expected no change
    depth_delta = (depth_delta + 1) / 2  # normalize depth_delta to [0, 1] with 0.5 expected no change
    grade = 100  # for verbose, how many distinct shades of blue to use (lower means higher contrast)
    pooled_origin = [dim / 10. for dim in raw_origin]
    pooled_origin = [pooled_origin[1], pooled_origin[0]]  # (coord x,y have to be swapped; Rosario might fix)

    # Visualize raw depth changes.
    if verbose:
        cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                      for cidx in range(grade)])
        bounds = range(grade)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow([[depth_delta[xidx, yidx] * grade
                    for yidx in range(depth_delta.shape[1])]
                   for xidx in range(depth_delta.shape[0])], cmap=cmap, norm=norm)
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
    for xidx in range(depth_delta.shape[0]):
        x = (pooled_origin[0] - xidx) / float(depth_delta.shape[0])  # normalize to [-1, 1]
        for yidx in range(depth_delta.shape[1]):
            y = (pooled_origin[1] - yidx) / float(depth_delta.shape[1])   # normalize to [-1, 1]
            d = np.linalg.norm([x, y])
            feat = None
            for fidx in range(num_feats):
                if d < band_upper_limits[fidx]:
                    feat = fidx
                    break
            if feat is None:
                feat = num_feats
            cells_by_feature[feat].append([xidx, yidx])
            cell_to_feature[(xidx, yidx)] = feat

    # Take the average depth change per band.
    features = [np.average([depth_delta[xidx, yidx] for xidx, yidx in cells_by_feature[fidx]])
                for fidx in range(num_feats + 1)]

    # Visualize average depth changes.
    if verbose:
        if verbose:
            print("cells_by_feature:")
            print("\t" + "\n\t".join([str(fidx) + ":\t" + str(len(cells_by_feature[fidx]))
                                      for fidx in range(num_feats)]))
            print("cells discarded:\t" + str(len(cells_by_feature[num_feats])))
            print("features: " + str(features[:-1]))

        cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                      for cidx in range(grade)])
        bounds = range(grade)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow([[features[cell_to_feature[(xidx, yidx)]] * grade
                   for yidx in range(depth_delta.shape[1])]
                   for xidx in range(depth_delta.shape[0])], cmap=cmap, norm=norm)
        plt.show()

    return features[:-1]  # discard outer rim


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
    print("Reading in robo dropping behavior data from '" + args.features_infile + "'...")
    with open(args.features_infile, 'r') as f:
        d = json.load(f)
    print("... done")

    # Get training and testing data.
    feats = {f: {p: {} for p in preps} for f in folds}
    labels = {f: {p: {} for p in preps} for f in folds}
    for p in preps:
        print("Collating available train/test data for '" + p + "'...")
        num_pairs = {f: 0 for f in folds}
        avg_trials = {f: 0 for f in folds}
        for k in d.keys():
            ob1n, ob2n = k[1:-2].split(',')
            ob1 = names.index(robo_data_ob_name_converter(ob1n.strip()))
            ob2 = names.index(robo_data_ob_name_converter(ob2n.strip()))
            for f in folds:
                for idx in range(len(all_d["folds"][f][p]["ob1"])):
                    if ob1 == all_d["folds"][f][p]["ob1"][idx] and ob2 == all_d["folds"][f][p]["ob2"][idx]:
                        for trial in d[k].keys():
                            if ob1 not in feats[f][p]:
                                feats[f][p][ob1] = {}
                                labels[f][p][ob1] = {}
                            if ob2 not in feats[f][p][ob1]:
                                feats[f][p][ob1][ob2] = []
                            feats[f][p][ob1][ob2].append(depthmaps_to_features(d[k][trial]['t0_depthmap'],
                                                                               d[k][trial]['t1_depthmap'],
                                                                               d[k][trial]['center_point'],
                                                                               15, verbose=False))
                            labels[f][p][ob1][ob2] = all_d["folds"][f][p]["label"][idx]
                        num_pairs[f] += 1
                        avg_trials[f] += len(d[k].keys())
        for f in folds:
            avg_trials[f] /= float(num_pairs[f]) if num_pairs[f] > 0 else 1
            print("... for '" + f + "', got " + str(num_pairs[f]) + " pairs with avg " +
                  str(avg_trials[f]) + " trials")
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
    parser.add_argument('--features_infile', type=str, required=True,
                        help="input json from rosbag processing")
    parser.add_argument('--features_outfile', type=str, required=True,
                        help="output json mapping object pairs to dropping behavior feature vectors")
    main(parser.parse_args())
