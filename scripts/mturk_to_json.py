#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the output CSV and log dir from the MTurk run and stores processed dataset as json.
# dataset json format:
# {"names": [obj1, obj2, ...],
#   {prep:
#     {obj1+obj2: [v1, v2, ...]
#      ...}
#   }
# }

import argparse
import numpy as np
import pandas as pd
import json


def main(args):

    # Read in CSV from MTurk to get uids.
    print("Reading data from '" + args.infile + "'...")
    df = pd.read_csv(args.infile)
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

            ob1 = ob1.strip().replace("\t","")
            ob2 = ob2.strip().replace("\t","")
            ans = df['Answer.annotation' + ns + '-mental'][idx]
            nans = -1 if ans == 0 else (1 if ans == 1 else 0)
            if ob1 not in objs:
                objs.append(ob1)
            if ob2 not in objs:
                objs.append(ob2)
            k = '+'.join([ob1, ob2])
            if aidx < 10:
                if k not in obj_in:
                    obj_in[k] = []
                obj_in[k].append(nans)
            else:
                if k not in obj_on:
                    obj_on[k] = []
                obj_on[k].append(nans)
    print("... done")

    # Some basic stats.
    for prop, d in [["in", obj_in], ["on", obj_on]]:
        print(prop + " stats:")
        num_votes = [len(d[k]) for k in d]
        total_by_num_votes = [num_votes.count(i) for i in range(1, 5)]
        print("\tNum votes totals:\t" + str(total_by_num_votes))
        print("\tNum votes percent:\t" + str(["%0.2f" % (n / float(sum(total_by_num_votes)))
                                              for n in total_by_num_votes]))
        sum_votes = [sum(d[k]) for k in d]
        total_by_sum_votes = [sum_votes.count(i) for i in range(-4, 5)]
        print("\tSum votes percent:\t" + str([str(i) + ": %0.2f" % (n / float(len(sum_votes)))
                                              for i, n in zip(range(-4, 5), total_by_sum_votes)]))
        num_concensus = len([1 for k in d if min(d[k]) == max(d[k])])
        num_offset = len([1 for k in d if min(d[k]) != max(d[k]) and sum(d[k]) != 0])
        num_even = len([1 for k in d if sum(d[k]) == 0])
        print("\tConsensus vote:\t" + str(num_concensus) + " (%0.2f" % (num_concensus / float(len(d))) + ")")
        print("\tOffset vote:\t" + str(num_offset) + " (%0.2f" % (num_offset / float(len(d))) + ")")
        print("\tEven vote:\t" + str(num_even) + " (%0.2f" % (num_even / float(len(d))) + ")")

    # Output to JSON.
    print("Writing to outfile to '" + args.outfile + "'...")
    with open(args.outfile, 'w') as f:
        json.dump({"names": objs, "votes": {"in": obj_in, "on": obj_on}}, f)
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input csv file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="the file to write the json dataset to")
    main(parser.parse_args())
