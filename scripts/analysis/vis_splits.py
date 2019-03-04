
import argparse
import numpy as np
import json
import csv

def main(args):
    with open(args.infile, 'r') as f:
        splits = json.load(f)
        names = splits['names']
        spl = splits['folds'][str(args.split)][str(args.pred)]
    
    with open(args.split + "_" + args.pred +  '.csv', 'w') as f:

        # sorting
        ob1s_sorted = np.array(spl['ob1'])
        ob2s_sorted = np.array(spl['ob2'])
        inds = ob2s_sorted.argsort()
        ob1s_sorted = ob1s_sorted[inds]
        ob2s_sorted.sort()

        # indexes -> names 
        names = np.array(names)
        ob1s = names[ob1s_sorted]
        ob2s = names[ob2s_sorted]

        # write to csv
        for i,item in enumerate(ob2s):
          f.write( str(ob1s[i]) + ' + ' + str(ob2s[i]) + '\n')
        print("Total " + args.split + " '" +  args.pred + "': " + str(len(ob2s)))

    print("... done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input dataset json file")
    parser.add_argument('--split', type=str, required=True,
                        help="which split do you want to see? train, dev, or test?")
    parser.add_argument('--pred', type=str, required=True,
                        help="which predicate do you want? on or in?")
    main(parser.parse_args())
