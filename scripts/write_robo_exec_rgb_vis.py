#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in a csv with object pairs and writes a directory populated with the RGB images of those pairs after-trial
# RBD exectution images for each of the 5 trials.
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from extract_robot_features import read_robo_dir, robo_data_ob_name_converter
from tqdm import tqdm


def main(args):

    # Read robo dir.
    grade = 255
    d = read_robo_dir(args.features_indir)

    # Read in object pairs line by line.
    print("Writing RGB final trial images to dir for every pair in input CSV...")
    not_found = []
    with open(args.infile, 'r') as f:
        for (idx, line) in tqdm(enumerate(f.readlines())):
            if idx == 0:  # header
                continue
            if len(line.strip()) == 0:  # blank
                continue
            ok = line.split(',')[0]
            k = '(' + ok.replace(' +', ',') + ')'
            found = False
            for dk in d:
                ob1n, ob2n = dk[1:-2].split(',')
                ob1 = robo_data_ob_name_converter(ob1n.strip())
                ob2 = robo_data_ob_name_converter(ob2n.strip())
                dk_conv = '(' + ob1 + ', ' + ob2 + ')'
                if k == dk_conv:
                    found = True
                    kdir = os.path.join(args.outdir, "%d_%s" % (idx, ok.replace(' ', '+')))
                    if not os.path.isdir(kdir):
                        os.system("mkdir " + kdir)
                    for trial in d[dk]:
                        im = np.asarray(d[dk][trial]['t1_rgbmap'])
                        im /= np.max(im)
                        fn = os.path.join(kdir, "%d.png" % int(trial))
                        cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                                      for cidx in range(grade)])
                        bounds = range(grade)
                        norm = colors.BoundaryNorm(bounds, cmap.N)
                        fig, ax = plt.subplots()
                        pltim = [[[int(im[cidx, xidx, yidx] * grade) for cidx in range(3)]
                                  for yidx in range(im.shape[2])]
                                 for xidx in range(im.shape[1])]
                        ax.imshow(pltim, cmap=cmap, norm=norm)
                        for spine in plt.gca().spines.values():
                            spine.set_visible(False)
                        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off',
                                        labelbottom='off', labeltop='off')
                        plt.savefig(fn, bbox_inches='tight')
                        plt.close(fig)
            if not found:
                not_found.append(k)
    print("... done; keys not found in robo structures:\n\t" + '\n\t'.join(not_found))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--features_indir', type=str, required=True,
                        help="input dir containing input jsons from rosbag processing")
    parser.add_argument('--infile', type=str, required=True,
                        help="csv infile specifying pairs")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to write RGB trial images to")
    main(parser.parse_args())
