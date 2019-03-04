#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Take in the pre-database CSV and output an input CSV for the referring expression synonyms MTurk task.

import argparse
import pandas as pd


def main(args):

    # Read in CSV from MTurk to get uids.
    print("Reading data from '" + args.infile + "'...")
    df = pd.read_csv(args.infile)
    names = []
    img_urls = []
    for idx in df.index:
        if type(df['image link'][idx]) is str and len(df['image link'][idx].strip()) > 0:
            name = df['Object Name'][idx].strip().replace("\t", "")
            url = df['image link'][idx].strip().replace("\t", "")
            assert name not in names
            assert url not in img_urls
            names.append(name)
            img_urls.append(url)
        else:
            print("WARNING: skipping '" + str(df['Object Name'][idx]) + "' with link '" +
                  str(df['image link'][idx]) + "'")

    # Write out task csv.
    print("Writing task to '" + args.outfile + "'...")
    with open(args.outfile, 'w') as f:
        f.write("name,image_url\n")
        f.write('\n'.join([','.join([names[idx], img_urls[idx]]) for idx in range(len(names))]))
    print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input csv file")
    parser.add_argument('--outfile', type=str, required=True,
                        help="the csv file to write out")
    main(parser.parse_args())
