#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
import numpy as np
from scipy.stats import ttest_ind


def main(args):

    preps = ["on", "in"]
    if args.robot == 1:
        models = ["mmS", "mmS+robot"]
        models_baseline_names = ["glove+resnetS", "glove+resnetS+robot"]
        folds = ["train", "dev", "test"]
    else:
        models = ["bow", "glove", "resnet", "mmS"]
        models_baseline_names = ["nn_bow", "glove", "resnet", "glove+resnetS"]
        folds = ["train", "dev", "test"]
    metric = "acc"
    test_accs = {p: [None for midx in range(len(models))] for p in preps}
    test_type = "test_robot" if args.robot == 1 else "test"

    for p in preps:

        # From results directory, check whether test set performance is missing and run those models if so.
        test_run = [False for _ in range(len(models))]
        for _, _, files in os.walk(args.results_dir):
            for fn in files:
                fnps = fn.split('.')
                if len(fnps) == 4 and fnps[1] == p and fnps[2] == metric and fnps[0] in models and fnps[3] == test_type:
                    m = fnps[0]
                    test_run[models.index(m)] = True
        for midx in range(len(models)):
            if not test_run[midx]:
                assert args.ff_random_restarts is not None
                hyp_fn = os.path.join(args.hyperparam_dir + '.'.join([models[midx], p, metric, "best"]))
                if not os.path.isfile(hyp_fn):
                    print("WARNING: hyperparameters " + hyp_fn + " are missing; cannot run tests on these!")
                    continue
                cmd = ("python run_baselines.py" +
                       " --infile " + args.infile +
                       " --metadata_infile " + args.metadata_infile +
                       " --baseline " + models_baseline_names[midx] +
                       " --glove_infile " + args.glove_infile +
                       " --task " + p +
                       " --hyperparam_infile " + os.path.join(args.hyperparam_dir +
                                                              '.'.join([models[midx], p, metric, "best"])) +
                       " --perf_outfile " + os.path.join(args.results_dir +
                                                         '.'.join([models[midx], p, metric, test_type])) +
                       " --ff_random_restarts " + args.ff_random_restarts)
                if args.robot:
                    cmd += " --test_robot 1"
                    cmd += " --robot_infile " + args.robot_infile
                else:
                    cmd += " --test 1"
                print("running " + test_type + " for '" + models[midx] + "' on task '" + p + "'...")
                print("command: '" + cmd + "'")
                os.system(cmd)  # blocking call
                print("... done")

        # From results directory, read in performance on train, dev, and test sets.
        results = {f: [None for _ in models] for f in folds}
        for _, _, files in os.walk(args.results_dir):
            for fn in files:
                fnps = fn.split('.')
                if len(fnps) == 4 and fnps[1] == p and fnps[2] == metric and fnps[0] in models:
                    m = fnps[0]
                    if fnps[3] == 'best':  # this contains dev and train results
                        with open(os.path.join(args.results_dir, fn), 'r') as f:
                            d = json.load(f)
                        dev_metric = d[0][p][metric]  # data is from single baselines at a time
                        train_metric = d[0][p]["tr_" + metric]
                        results['train'][models.index(m)] = train_metric
                        results['dev'][models.index(m)] = dev_metric
                    elif fnps[3] == test_type:  # this contains test and (train+dev) results
                        with open(os.path.join(args.results_dir, fn), 'r') as f:
                            d = json.load(f)
                        test_metric = d[0][p][metric]
                        results['test'][models.index(m)] = test_metric

        # Report average and standard deviation of each model on each fold.
        print(p + " average and stddev:")
        for f in folds:
            print("\t" + f + ":")
            for midx in range(len(models)):
                if results[f][midx] is None:
                    print("\t\tWARNING: no results yet for '" + models[midx] + "' on fold " + f)
                    continue
                print("\t\t" + models[midx] + "\t%0.3f" % np.average(results[f][midx]) +
                      " +/- %0.3f" % np.std(results[f][midx]))

        # Statistical tests.
        equal_var = True  # whether to assume equal variance between model architectures under different seeds
        pvt = 0.05  # threshold to alert significance
        print(p + " t-tests:")
        for f in folds:
            pv_m = [[None for mjdx in range(len(models))] for midx in range(len(models))]
            sig = [False for _ in range(len(models))]
            for midx in range(len(models)):
                if results[f][midx] is None:
                    continue
                for mjdx in range(midx, len(models)):
                    if results[f][mjdx] is None:
                        continue
                    _, pv = ttest_ind(results[f][midx], results[f][mjdx], equal_var=equal_var)
                    pv_m[midx][mjdx] = pv
                    pv_m[mjdx][midx] = pv
                sig[midx] = np.all([pv_m[midx][mjdx] is None or pv_m[midx][mjdx] < pvt
                                    for mjdx in range(len(models)) if mjdx != midx])
            print("\t" + f + ":")
            print("\t\t" + "\t".join(models))
            print("\t\t" + "\n\t\t".join(["\t".join(["%.2f" % pv if pv is not None else '?' for pv in pv_m[midx]])
                                            for midx in range(len(models))]))
            for midx in range(len(models)):
                if sig[midx]:
                    print("\t\tmodel '" + models[midx] + "' differs from others with p < " + str(pvt))

        for midx in range(len(models)):
            test_accs[p][midx] = results["test"][midx]

    # Print test acc harmonic means
    print("harmonic means of " + test_type + " accuracies across tasks:")
    for midx in range(len(models)):
        means = [2 * (test_accs["on"][midx][idx] * test_accs["in"][midx][idx]) /
                 (test_accs["on"][midx][idx] + test_accs["in"][midx][idx])
                 for idx in range(len(test_accs["on"][midx]))]
        print("\t\t" + models[midx] + "\t%0.3f" % np.average(means) + " +/- %0.3f" % np.std(means))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    parser.add_argument('--glove_infile', type=str, required=True,
                        help="input glove vector text file if running glove baseline")
    parser.add_argument('--hyperparam_dir', type=str, required=True,
                        help="directory for model hyperparameters")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="directory for model performance")
    parser.add_argument('--ff_random_restarts', type=str, required=False,
                        help="comma-separated list of random seeds to use")
    parser.add_argument('--robot', type=int, required=False,
                        help="whether to test on the robot subset of data")
    parser.add_argument('--robot_infile', type=str, required=False,
                        help="robot feature data filename")
    main(parser.parse_args())
