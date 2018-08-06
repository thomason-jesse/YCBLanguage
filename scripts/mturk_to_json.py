#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the output CSV and log dir from the MTurk run and stores processed dataset as json.

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
            ans = df['Answer.annotation' + ns + '-mental'][idx]
            nans = -1 if ans == 0 else (1 if ans == 1 else 0)
            if ob1 not in objs:
                objs.append(ob1)
            if ob2 not in objs:
                objs.append(ob2)
            k = (ob1, ob2)
            if k not in obj_in:
                obj_in[k] = []
                obj_on[k] = []
            if aidx < 10:
                obj_in[k].append(nans)
            else:
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input csv file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="the file to write the json dataset to")
    main(parser.parse_args())
