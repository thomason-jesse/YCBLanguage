#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in predictions from models stored as JSON and produces metadata about "interesting" pairs that models
# generally disagree on.

import argparse
import json
from models import *
from scipy.stats import ttest_ind
from tqdm import tqdm
from utils import *


def main(args, dv):
    assert args.conf_thresh is not None or args.use_highest_acc

    # Read in splits.
    # Read in labeled folds.
    print("Reading in labeled folds from '" + args.splits_infile + "'...")
    with open(args.splits_infile, 'r') as f:
        all_d = json.load(f)
        names = all_d["names"]
    print("... done")

    # Read in metadata.
    print("Reading in metadata from '" + args.metadata_infile + "'...")
    with open(args.metadata_infile, 'r') as f:
        d = json.load(f)
        res = d["res"]
        imgs = d["imgs"]
    print("... done")

    # Read in predictions.
    print("Reading in predictions from '" + args.predictions_infile + "'...")
    with open(args.predictions_infile, 'r') as f:
        preds = json.load(f)
        preps = preds.keys()
    print("... done")

    for p in preps:
        print(p)

        # Read gold labels.
        fn = args.torch_infile + '.' + p
        with open(fn, 'r') as f:
            d = json.load(f)
            gold_labels = {'(%s, %s)' % (d["test"]["names"][idx][0], d["test"]["names"][idx][1]):
                            (0 if d["test"]["robo_label"][idx][0] < 2 else 1)
                            if args.round_m_to_n == 1 else d["test"]["robo_label"][idx][0]
                           for idx in range(len(d["test"]["names"]))}

        # Take votes of model decisions across seeds to get "typical" model responses.
        vote_predictions = {}
        accs = {}
        model_names = {}
        for midx in range(len(preds[p]["model"])):
            base = preds[p]["model"][midx][:min(preds[p]["model"][midx].find('_s-'),
                                                preds[p]["model"][midx].find('/')
                                                if preds[p]["model"][midx].find('/') > 0
                                                else len(preds[p]["model"][midx]))]
            if base not in vote_predictions:
                if len(preds[p]["predictions"][midx]) == 0:
                    continue  # no predictions saved for this model (MC, OMC)
                vote_predictions[base] = {pair: [] for pair in preds[p]["predictions"][midx]}
                accs[base] = []
                model_names[base] = []
            accs[base].append(float(preds[p]["accuracy"][midx]))
            model_names[base].append(preds[p]["model"][midx])
            for pair in preds[p]["predictions"][midx]:
                vote_predictions[base][pair].append(preds[p]["predictions"][midx][pair])

        # Turn votes into confidences across seeds.
        conf_predictions = {}  # the confidence of the chosen class
        max_predictions = {}  # the actual class chosen most
        highest_acc_models = {}
        for base in vote_predictions:
            conf_predictions[base] = {}
            max_predictions[base] = {}
            for pair in vote_predictions[base]:
                if base not in highest_acc_models:
                    highest_acc_models[base] = int(np.argmax(accs[base]))
                if args.conf_thresh is not None:
                    counts = [vote_predictions[base][pair].count(c) for c in range(3)]
                    chosen = int(np.argmax(counts))
                    conf = counts[chosen] / float(sum(counts))
                    if conf > args.conf_thresh:
                        conf_predictions[base][pair] = counts[chosen] / float(sum(counts))
                        max_predictions[base][pair] = chosen
                elif args.use_highest_acc:
                    highest_acc_idx = highest_acc_models[base]
                    chosen = vote_predictions[base][pair][highest_acc_idx]
                    conf_predictions[base][pair] = 1
                    max_predictions[base][pair] = chosen

        # Find and report contrasting pairs across models.
        bases = list(max_predictions.keys())
        for bidx in range(len(bases) - 1):
            print('\t%s\t(%.3f, %s)' % (bases[bidx], accs[bases[bidx]][highest_acc_models[bases[bidx]]],
                                        model_names[bases[bidx]][highest_acc_models[bases[bidx]]]))
            for bjdx in range(bidx + 1, len(bases)):
                print('\t\t%s\t(%.3f, %s)' % (bases[bjdx], accs[bases[bjdx]][highest_acc_models[bases[bjdx]]],
                                        model_names[bases[bjdx]][highest_acc_models[bases[bjdx]]]))
                for pair in max_predictions[bases[bidx]]:
                    if (pair in max_predictions[bases[bjdx]] and
                            max_predictions[bases[bidx]][pair] != max_predictions[bases[bjdx]][pair]):
                        print('\t\t\t%s\t%d != %d (g-%d)\t(%.3f, %.3f)' %
                              (pair, max_predictions[bases[bidx]][pair],
                               max_predictions[bases[bjdx]][pair],
                               gold_labels[pair],
                               conf_predictions[bases[bidx]][pair],
                               conf_predictions[bases[bjdx]][pair]))
                        obs = pair.strip('()').replace(' ', '').split(',')
                        print('\t\t\t    ' + '\n\t\t\t    '.join([imgs[names.index(obs[0])]]) +  # Remove nesting [] when imgs is array
                              '\n\t\t\t  ---\n\t\t\t    ' +
                              '\n\t\t\t    '.join([imgs[names.index(obs[1])]]))
                        print('\n\t\t\t    ' + '\n\t\t\t    '.join([' '.join(re) for re in res[names.index(obs[0])]]) +
                              '\n\t\t\t  ---\n\t\t\t    ' +
                              '\n\t\t\t    '.join([' '.join(re) for re in res[names.index(obs[1])]]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_infile', type=str, required=True,
                        help="torch ready train/test input root to load as json")
    parser.add_argument('--splits_infile', type=str, required=True,
                        help="input json file with data splits (contains names array)")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    parser.add_argument('--torch_infile', type=str, required=True,
                        help="input json file with torch data")
    parser.add_argument('--conf_thresh', type=float, required=False,
                        help="threshold of confidence to keep a pair for a model")
    parser.add_argument('--use_highest_acc', required=False, action='store_true',
                        help="whether to consider only the model with the highest accuracy")
    parser.add_argument('--round_m_to_n', type=int, required=False,
                        help="if true, round gold labels of M to N")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args(), device)
