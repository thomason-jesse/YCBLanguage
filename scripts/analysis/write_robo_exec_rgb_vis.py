#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in a csv with object pairs and writes a directory populated with the RGB images of those pairs after-trial
# RBD exectution images for each of the 5 trials.
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import sys
from extract_robot_features import read_robo_dir, robo_data_ob_name_converter
from tqdm import tqdm


def main(args):

    # Read robo dir.
    grade = 255
    d = read_robo_dir(args.features_indir)

    # Read in object pairs line by line.
    print("Writing RGB final trial images to dir for every pair in input CSV...")
    not_found = []
    com = {0: 2, 1: 1, 2: 0}  # R->B, G->G, B->R
    with open(args.infile, 'r') as f:
        lines = f.read().split('\n')
        for (idx, line) in tqdm(enumerate(lines)):
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
                        for scan in ["0", "1"]:
                            im_rgb = np.asarray(d[dk][trial]['t%s_rgbmap' % scan], dtype=float)
                            im_d = np.asarray(d[dk][trial]['t%s_depthmap' % scan], dtype=float)
                            if im_rgb.shape[0] == 3:
                                cdim = 0
                                xdim = 1
                                ydim = 2
                            elif im_rgb.shape[2] == 3:
                                cdim = 2
                                xdim = 0
                                ydim = 1
                            else:
                                sys.exit("Unrecognized RGB shape " + str(im_rgb.shape))
                            im_rgb /= np.max(im_rgb) if cdim == 0 else 255.
                            im_d /= np.max(im_d)
                            fn_rgb = os.path.join(kdir, "%d.rgb.%s.png" % (int(trial), scan))
                            fn_d = os.path.join(kdir, "%d.d.%s.png" % (int(trial), scan))
                            cmap = colors.ListedColormap([[(cidx + 1) / float(grade) for _ in range(3)]
                                                          for cidx in range(grade)])
                            bounds = range(grade)
                            norm = colors.BoundaryNorm(bounds, cmap.N)

                            fig, ax = plt.subplots()
                            pltim = [[[int((im_rgb[cidx, xidx, yidx] if cdim == 0 else im_rgb[xidx, yidx, com[cidx]])
                                           * grade)
                                       for cidx in range(3)]
                                      for yidx in range(im_rgb.shape[ydim])]
                                     for xidx in range(im_rgb.shape[xdim])]
                            ax.imshow(pltim, cmap=cmap, norm=norm)
                            for spine in plt.gca().spines.values():
                                spine.set_visible(False)
                            plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off',
                                            labelbottom='off', labeltop='off')
                            plt.savefig(fn_rgb, bbox_inches='tight')
                            plt.close(fig)

                            fig, ax = plt.subplots()
                            pltim = [[[int((im_d[xidx, yidx] if cdim == 0 else im_d[xidx, yidx])
                                           * grade)
                                       for _ in range(3)]
                                      for yidx in range(im_d.shape[1])]
                                     for xidx in range(im_d.shape[0])]
                            ax.imshow(pltim, cmap=cmap, norm=norm)
                            for spine in plt.gca().spines.values():
                                spine.set_visible(False)
                            plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off',
                                            labelbottom='off', labeltop='off')
                            plt.savefig(fn_d, bbox_inches='tight')
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
