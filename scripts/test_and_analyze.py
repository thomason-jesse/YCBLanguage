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
<<<<<<< HEAD
    models = ["bow", "glove", "resnet", "mmS"]
    models_baseline_names = ["nn_bow", "glove", "resnet", "glove+resnetS"]
    metric = "acc"
    folds = ["train", "dev", "test"]
    test_accs = {p: [None for midx in range(len(models))] for p in preps}
=======
    models = ["bow", "glove", "resnet", "mm"]
    models_baseline_names = ["nn_bow", "glove", "resnet", "glove+resnet"]
    metric = "acc"
    folds = ["train", "dev", "test"]
>>>>>>> b45f75e50f56c5f4cbc18f51edf59e9f5042a889
    for p in preps:

        # From results directory, check whether test set performance is missing and run those models if so.
        test_run = [False for _ in range(len(models))]
        for _, _, files in os.walk(args.results_dir):
            for fn in files:
                fnps = fn.split('.')
                if len(fnps) == 4 and fnps[1] == p and fnps[2] == metric and fnps[0] in models and fnps[3] == 'test':
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
                                                         '.'.join([models[midx], p, metric, "test"])) +
                       " --ff_random_restarts " + args.ff_random_restarts +
                       " --test 1")
                print("running test for '" + models[midx] + "' on task '" + p + "'...")
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
                    elif fnps[3] == 'test':  # this contains test and (train+dev) results
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
                    print("\t\tWARNING: no results yet for '" + models[midx] + "'")
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
                sig[midx] = np.all([pv_m[midx][mjdx] < pvt or pv_m[midx][mjdx] is None
                                    for mjdx in range(len(models)) if mjdx != midx])
            print("\t" + f + ":")
            print("\t\t" + "\t".join(models))
            print("\t\t" + "\n\t\t".join(["\t".join(["%.2f" % pv if pv is not None else '?' for pv in pv_m[midx]])
                                            for midx in range(len(models))]))
            for midx in range(len(models)):
                if sig[midx]:
                    print("\t\tmodel '" + models[midx] + "' differs from others with p < " + str(pvt))

<<<<<<< HEAD
        for midx in range(len(models)):
            test_accs[p][midx] = results["test"][midx]

    # Print test acc harmonic means
    print("harmonic means of test accuracies across tasks:")
    for midx in range(len(models)):
        means = [2 * (test_accs["on"][midx][idx] * test_accs["in"][midx][idx]) /
                 (test_accs["on"][midx][idx] + test_accs["in"][midx][idx])
                 for idx in range(len(test_accs["on"][midx]))]
        print("\t\t" + models[midx] + "\t%0.3f" % np.average(means) + " +/- %0.3f" % np.std(means))

=======
>>>>>>> b45f75e50f56c5f4cbc18f51edf59e9f5042a889

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
    main(parser.parse_args())
