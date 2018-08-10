#!/usr/bin/env python3
__author__ = 'thomason-jesse'
# Takes in the output CSV from a YCB RE MTurk run and creates a json output and optional HTML visualization and summary.
# json format:
# {"names": [str1, str2, ...],
#  "res": [[re11, re12, ...], [re21, re22, ...] ...]
#  "imgs": [img_fp1, img_fp2, ...]
# }

import argparse
import pandas as pd
import json
import os
import string


def main(args):
    assert os.path.isdir(args.outdir)
    img_dir = os.path.join(args.outdir, "imgs")
    outfile = os.path.join(args.outdir, "lang_vis.json")
    if not os.path.isdir(img_dir):
        cmd = "mkdir " + img_dir
        os.system(cmd)

    # Read in CSV from MTurk and validate users.
    print("Reading data in from '" + args.infile + "'...")
    df = pd.read_csv(args.infile)
    imgs = {}
    descs = {}
    transtable = str.maketrans(' ', ' ', string.punctuation)
    for idx in df.index:

        if df['AssignmentStatus'][idx] == "Approved":
            wds = [df['Answer.field1'][idx].lower().strip(), df['Answer.field2'][idx].lower().strip(),
                   df['Answer.field3'][idx].lower().strip()]
            dtks = []
            v = True
            for d in wds:
                d = d.replace("'s", " 's")
                d = d.translate(transtable)

                # Custom cleans.
                tostrike = [" over there", "give me ", "could you ", "can you ", "that s sitting there",
                            "i would like ", " in the middle of the glass", " on the middle of the glass",
                            " on top of the glass", " on the glass", " on the table", " sitting", " there"]
                for tos in tostrike:
                    if tos in d:
                        dr = d.replace(tos, "")
                        print("WARNING: custom clean '" + d + "' -> '" + dr + "'")
                        d = dr

                # Custom splits.
                tosplit = [["sixsided", "six sided"], ["mediumsized", "medium sized"],
                           ["blockobject", "block object"], ["pearshaped", "pear shaped"],
                           ["avocadoshaped", "avocado shaped"], ["cubelooking", "cube looking"],
                           ["cubelike", "cube like"], ["pokingout", "poking out"],
                           ["marbleshaped", "marble shaped"], ["cubeshaped", "cube shaped"],
                           ["boxthing", "box thing"], ["pokingbits", "poking bits"], ["bridgelike", "bridge like"]]
                for sp, p in tosplit:
                    if sp in d:
                        dr = d.replace(sp, p)
                        print("WARNING: custom split '" + d + "' -> '" + dr + "'")
                        d = dr

                tks = [s for s in d.split() if len(s) > 0]
                dtks.append(tks)
            if dtks[0] == dtks[1] or dtks[1] == dtks[2] or dtks[0] == dtks[2]:
                print("Worker " + df["WorkerId"][idx] + " wrote same description twice; '" + ' '.join(dtks[0]) + "'")
                v = False

            # Store data for analysis.
            if v:
                imgs[df['Input.name'][idx]] = df['Input.image_url'][idx]
                if df['Input.name'][idx] not in descs:
                    descs[df['Input.name'][idx]] = []
                descs[df['Input.name'][idx]].extend(dtks)
    print("... done")

    # Calculate data statistics.
    print("Num valid, annotated items:\t" + str(len(descs)))
    sd = 0
    sdl = 0
    sut = 0
    wc = {}
    for n in descs:
        sd += len(descs[n])
        for d in descs[n]:
            sdl += len(d)
            sut += len(set(d))
            for tk in d:
                if tk not in wc:
                    wc[tk] = 0
                wc[tk] += 1
    ad = sd / float(len(descs))
    adl = sdl / float(sd)
    aut = sut / float(sd)
    topw = [(k, wc[k]) for k in sorted(wc, key=wc.get, reverse=True)]
    print("Avg REs per item:\t" + str(ad))
    print("Avg tokens per RE:\t" + str(adl))
    print("Avg unique tokens per RE:\t" + str(aut))
    print("Num unique words:\t" + str(len(wc)))
    print("Ten most frequent words:\t" + str(topw[:10]))
    print("Ten least frequent words:\t" + str(topw[-10:]))

    # Get images.
    print("Downloading images to local directory '" + img_dir + "'...")
    limgs = {}
    for n in descs:
        lfp = os.path.join(img_dir, n + ".jpeg")
        if not os.path.isfile(lfp):
            cmd = "curl -o " + lfp + " " + imgs[n]
            os.system(cmd)
        limgs[n] = lfp
    print("... done")

    # Write to outfile.
    print("Writing out to '" + outfile + "'...")
    with open(outfile, 'w') as f:
        json.dump({"names": list(descs.keys()),
                   "res": [[re for re in descs[k]] for k in descs.keys()],
                   "imgs": [limgs[k] for k in descs.keys()]},
                  f, indent=2)
    print("... done")

    # Write to HTML outfile.
    if args.html_outfile is not None:
        print("Writing output to '" + args.html_outfile + "'...")
        with open(args.html_outfile, 'w') as f:
            f.write("<p>Num valid, annotated items: " + str(len(descs)) + "<br/>")
            f.write("Avg REs per item: " + str(ad) + "<br/>")
            f.write("Avg tokens per RE: " + str(adl) + "<br/>")
            f.write("Avg unique tokens per RE: " + str(aut) + "<br/>")
            f.write("Num unique words: " + str(len(wc)) + "<br/>")
            f.write("Ten most frequent words: " + str(topw[:10]) + "<br/>")
            f.write("Ten least frequent words: " + str(topw[-10:]) + "</p>")

            f.write("<table border='1'><tr><th>Name</th><th>Pic</th><th>REs</th></tr>")
            for n in descs:
                f.write("<tr><td>" + n + "</td>")
                f.write("<td><img src=\"" + imgs[n] + "\" width=\"400px\"></td>")
                f.write("<td>" + "<br/><br/>".join([' '.join(d) for d in descs[n]]) + "</td>")
                f.write("</tr>\n")
            f.write("</table>")

            f.write("<table><tr><th>Word</th><th>Frequency</th></tr>")
            for w, k in topw:
                f.write("<tr><td>" + w + "</td><td>" + str(k) + "</td></tr>\n")
            f.write("</table>")
        print("... done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True,
                        help="input csv file")
    parser.add_argument('--outdir', type=str, required=True,
                        help="output directory for json file and images directory")
    parser.add_argument('--html_outfile', type=str, required=False,
                        help="output html file for visualization")
    main(parser.parse_args())
