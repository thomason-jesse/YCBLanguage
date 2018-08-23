#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the split and labeled data folds, runs baselines and reports performance.

import argparse
import json
import os
import numpy as np
import subprocess
import time


def main(args):
    assert args.metric in ['acc', 'f1']

    # At every update stage, launch jobs up to num_jobs limit until the total number of launched jobs ever
    # is equal to the num_models we want to test. Report best-so-far model at each timestep that it changes.
    preps = ["in", "on"]
    best_params = {p: None for p in preps}
    best_score = {p: 0 for p in preps}
    score_idx = 0 if args.metric == 'acc' else 1
    total_models_finished = 0
    curr_jobs = {jidx: None for jidx in range(args.num_jobs)}
    ff_random_restarts = None if args.ff_random_restarts is None else \
        [int(s) for s in args.ff_random_restarts.split(',')]
    print("spinning until all " + str(args.num_models) + " models have been run...")
    null = open('/dev/null', 'w')
    while total_models_finished < args.num_models:

        # Launch new jobs if we have space and need to.
        free_job_slots = [jidx for jidx in range(args.num_jobs) if curr_jobs[jidx] is None]
        while (len(free_job_slots) > 0 and
               total_models_finished + (args.num_jobs - len(free_job_slots)) < args.num_models):
            jidx = free_job_slots.pop()

            # New hyperparamters for this job.
            d = {}
            d['layers'] = np.random.randint(0, 4)  # 0 - 3 layers
            d['width_decay'] = 3 + np.random.rand() * 2  # 3 - 5 width decay rate
            d['dropout'] = np.random.rand() * 0.5  # in [0, 0.5]
            d['lr'] = np.power(10, (-1 - np.random.rand() * 3))  # 0.1 to 0.0001
            d['opt'] = np.random.choice(['adagrad', 'adam', 'rmsprop', 'sgd'])

            # Write hyperparam file.
            fn = args.hyperparam_outfile_prefix + "." + str(jidx)
            with open(fn, 'w') as f:
                json.dump(d, f)
            fn_out = fn + ".out"

            # Launch job.
            cmd = ("python run_baselines.py" +
                   " --infile " + args.infile +
                   " --metadata_infile " + args.metadata_infile +
                   " --baseline " + args.baseline +
                   " --glove_infile " + args.glove_infile +
                   " --hyperparam_infile " + fn +
                   " --perf_outfile " + fn_out)
            if ff_random_restarts is not None:
                   cmd += " --ff_random_restarts " + args.ff_random_restarts
            p = subprocess.Popen(cmd.split(' '), stdin=null, stdout=null)
            curr_jobs[jidx] = p
            print("... launched a new model")

        # Check whether jobs have finished and record results.
        best_changed = {p: False for p in preps}
        jobs_finished = 0
        for jidx in range(args.num_jobs):
            if curr_jobs[jidx] is not None:
                jr = curr_jobs[jidx].poll()
                if jr is not None:
                    if jr == 0:
                        fn = args.hyperparam_outfile_prefix + "." + str(jidx)
                        fn_out = fn + ".out"
                        with open(fn_out, 'r') as f:
                            d = json.load(f)[0]  # only ran one model, so take it out of the list results
                            for p in preps:
                                if ff_random_restarts is None:
                                    model_score = d[p][score_idx]
                                else:
                                    model_score = np.average(d[p][args.metric])
                                if model_score > best_score[p]:  # val accuracy or f1
                                    best_score[p] = model_score
                                    with open(fn, 'r') as fhp:
                                        hp = json.load(fhp)
                                        best_params[p] = hp
                                    best_changed[p] = True
                                    os.system("cp " + fn + " " + args.hyperparam_outfile_prefix + "." + p + "." +
                                              args.metric + ".best")
                                    os.system("cp " + fn_out + " " + args.results_outfile_prefix + "." + p + "." +
                                              args.metric + ".best")
                            os.system("rm " + fn)
                        os.system("rm " + fn_out)
                        jobs_finished += 1
                        total_models_finished += 1
                        curr_jobs[jidx] = None
                    else:
                        print("... job " + str(jidx) + " crashed with error code " + str(jr))
                        curr_jobs[jidx] = None

        # Print status.
        if jobs_finished > 0:
            print("... finished " + str(jobs_finished) + " more jobs; (" +
                  str(total_models_finished) + "/" + str(args.num_models) + ")")
            for p in preps:
                if best_changed[p]:
                    print("...... new best '" + p + "' hyperparams: " +
                          str(best_params[p]) + " with " + args.metric + " " + str(best_score[p]))

        # Rest.
        time.sleep(10)

    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input json file with train/dev/test split")
    parser.add_argument('--metadata_infile', type=str, required=True,
                        help="input json file with object metadata")
    parser.add_argument('--baseline', type=str, required=True,
                        help="if None, all will run, else 'majority', 'nb_names', 'nb_bow', 'lstm', 'glove'," +
                             " 'nn_bow', 'resnet', 'glove+resnet'")
    parser.add_argument('--glove_infile', type=str, required=True,
                        help="input glove vector text file if running glove baseline")
    parser.add_argument('--results_outfile_prefix', type=str, required=True,
                        help="output json filename prefix for best parameter results")
    parser.add_argument('--hyperparam_outfile_prefix', type=str, required=True,
                        help="output json filename prefix for best model hyperparameters")
    parser.add_argument('--num_models', type=int, required=True,
                        help="the total number of models to try")
    parser.add_argument('--num_jobs', type=int, required=True,
                        help="the total of jobs to run in parallel")
    parser.add_argument('--metric', type=str, required=True,
                        help="either 'acc' or 'f1'")
    parser.add_argument('--ff_random_restarts', type=str, required=False,
                        help="comma-separated list of random seeds to use; otherwise single model eval")
    main(parser.parse_args())
