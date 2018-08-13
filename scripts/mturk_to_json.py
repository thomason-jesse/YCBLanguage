#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the output CSV and log dir from the MTurk run and stores processed dataset as json.
# dataset json format:
# {"names": [obj1, obj2, ...],
#  "res": [[re11, re12, ...], [re21, re22, ...] ...]
#  "imgs": [img_fp1, img_fp2, ...]
#  "votes":
#   {prep:
#     {obj1:
#       {obj2: [v1, v2, ...]
#        ...
#       }
#      ...
#     }
#    ...
#   }
# }

import argparse
import numpy as np
import pandas as pd
import json


def main(args):

    # Read in CSV from MTurk run 1.
    print("Reading data from '" + args.infile_run1 + "'...")
    df = pd.read_csv(args.infile_run1)
    objs = []  # object names
    obj_in = {}  # keyed (ob1, ob2) for object 1 can fit into object 2, value list of -1, 0, 1 votes
    obj_on = {}  # keyed (ob1, ob2) for object 1 can be stacked onto object 2, value list of -1, 0, 1 votes
    for idx in df.index:
        for aidx in range(1, 36):
            ns = "%02d" % aidx
            ob1 = df['Input.img_' + ns + 'a_name'][idx] if 'Input.img_' + ns + 'a_name' in df \
                else df['Answer.img_' + ns + 'a_name'][idx]
            ob2 = df['Input.img_' + ns + 'b_name'][idx] if 'Input.img_' + ns + 'b_name' in df \
                else df['Answer.img_' + ns + 'b_name'][idx]
            ob1 = ob1.strip().replace("\t", "")
            ob2 = ob2.strip().replace("\t", "")
            ans = df['Answer.annotation' + ns + '-mental'][idx]

            # Ignore "yes, but" votes (2)
            # nans = -1 if ans == 0 else (1 if ans == 1 else 0)
            # Round "yes, but" votes (2) down to "no"
            nans = 1 if ans == 1 else -1

            if ob1 not in objs:
                objs.append(ob1)
            if ob2 not in objs:
                objs.append(ob2)
            oidx = objs.index(ob1)
            ojdx = objs.index(ob2)

            if aidx < 10:
                if oidx not in obj_in:
                    obj_in[oidx] = {}
                if ojdx not in obj_in[oidx]:
                    obj_in[oidx][ojdx] = []
                obj_in[oidx][ojdx].append(nans)
            else:
                if oidx not in obj_on:
                    obj_on[oidx] = {}
                if ojdx not in obj_on[oidx]:
                    obj_on[oidx][ojdx] = []
                obj_on[oidx][ojdx].append(nans)
    print("... done")

    # Read in CSV from Mturk run 2 and replace votes for affected pairs.
    print("Reading data from '" + args.infile_run2 + "'...")
    df = pd.read_csv(args.infile_run2)
    overwrite_votes = set()
    for idx in df.index:
        for aidx in range(1, 36):
            ns = "%02d" % aidx
            ob1 = df['Input.img_' + ns + 'a_name'][idx] if 'Input.img_' + ns + 'a_name' in df \
                else df['Answer.img_' + ns + 'a_name'][idx]
            ob2 = df['Input.img_' + ns + 'b_name'][idx] if 'Input.img_' + ns + 'b_name' in df \
                else df['Answer.img_' + ns + 'b_name'][idx]
            ob1 = ob1.strip().replace("\t", "")
            ob2 = ob2.strip().replace("\t", "")
            ans = df['Answer.annotation' + ns + '-mental'][idx]

            # Ignore "yes, but" votes (2)
            # nans = -1 if ans == 0 else (1 if ans == 1 else 0)
            # Round "yes, but" votes (2) down to "no"
            nans = 1 if ans == 1 else -1

            assert ob1 in objs
            assert ob2 in objs
            oidx = objs.index(ob1)
            ojdx = objs.index(ob2)
            ow = True if (oidx, ojdx) not in overwrite_votes else False
            overwrite_votes.add((oidx, ojdx))

            if aidx < 10:
                assert oidx in obj_in
                assert ojdx in obj_in[oidx]
                if ow:
                    obj_in[oidx][ojdx] = []
                obj_in[oidx][ojdx].append(nans)
            else:
                assert oidx in obj_on
                assert ojdx in obj_on[oidx]
                if ow:
                    obj_on[oidx][ojdx] = []
                obj_on[oidx][ojdx].append(nans)
    print("... done; overwrote votes for " + str(len(overwrite_votes)) + " pairs")

    # Some basic stats.
    for prop, d in [["in", obj_in], ["on", obj_on]]:
        print(prop + " stats:")
        num_votes = [len(d[oidx][ojdx]) for oidx in d for ojdx in d[oidx]]
        total_by_num_votes = [num_votes.count(i) for i in range(1, 6)]
        print("\tNum votes totals:\t" + str(total_by_num_votes))
        print("\tNum votes percent:\t" + str(["%0.2f" % (n / float(sum(total_by_num_votes)))
                                              for n in total_by_num_votes]))
        sum_votes = [sum(d[oidx][ojdx]) for oidx in d for ojdx in d[oidx]]
        total_by_sum_votes = [sum_votes.count(i) for i in range(-5, 6)]
        print("\tSum votes percent:\t" + str([str(i) + ": %0.2f" % (n / float(len(sum_votes)))
                                              for i, n in zip(range(-5, 6), total_by_sum_votes)]))
        num_concensus = len([1 for oidx in d for ojdx in d[oidx]
                             if min(d[oidx][ojdx]) == max(d[oidx][ojdx])])
        num_offset = len([1 for oidx in d for ojdx in d[oidx]
                          if min(d[oidx][ojdx]) != max(d[oidx][ojdx]) and sum(d[oidx][ojdx]) != 0])
        num_even = len([1 for oidx in d for ojdx in d[oidx]
                        if sum(d[oidx][ojdx]) == 0])
        print("\tConsensus vote:\t" + str(num_concensus) + " (%0.2f" % (num_concensus / float(len(num_votes))) + ")")
        print("\tOffset vote:\t" + str(num_offset) + " (%0.2f" % (num_offset / float(len(num_votes))) + ")")
        print("\tEven vote:\t" + str(num_even) + " (%0.2f" % (num_even / float(len(num_votes))) + ")")

    # Read in language and vision JSON.
    print("Reading in language and vision data from '" + args.in_lang_vis + "'...")
    with open(args.in_lang_vis, 'r') as f:
        lvd = json.load(f)
    print("... done")

    # Output to JSON.
    print("Writing to outfile to '" + args.outfile + "'...")
    with open(args.outfile, 'w') as f:
        json.dump({"names": objs,
                   "res": [lvd["res"][lvd["names"].index(n)] for n in objs],
                   "imgs": [lvd["imgs"][lvd["names"].index(n)] for n in objs],
                   "votes": {"in": obj_in, "on": obj_on}},
                  f, indent=2)
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_run1', type=str, required=True,
                        help="input csv file for the first round of mturk")
    parser.add_argument('--infile_run2', type=str, required=True,
                        help=("input csv file for the second round of mturk for" +
                              " mismatched images from round 1"))
    parser.add_argument('--in_lang_vis', type=str, required=True,
                        help="input json with language and vision data")
    parser.add_argument('--outfile', type=str, required=True,
                        help="the file to write the json dataset to")
    main(parser.parse_args())
