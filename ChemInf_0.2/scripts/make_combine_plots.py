#!usr/bin/env python

__author__ = 'Andreas Johan Luttens'
__date__ = '2020-11-13'
__email__ = 'andreas.luttens@gmail.com'
__version__ = '0.1'

"""
Plot ConfPred statistics 

[Andreas Luttens, Uppsala University - 2020]
"""

import argparse
import matplotlib.pyplot as pt
import seaborn as sns
import numpy as np


def ParseArgs():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(epilog='GitHub URL')
    parser.add_argument('-i', '--inputfiles',
                        nargs='+',
                        help='files containing enrichment values',
                        required=True)
    parser.add_argument('-o', '--outputname',
                        help='name of image output file',
                        required=True)
    args = parser.parse_args()
    return args


def read_pred_summary(in_file):
    """Reads a amcp_pred_summary file as outputted
    from the amcp_summarize_pred.py script.
    """

    with open(in_file, 'r') as fin:
        header = fin.readline().strip().split()

    data = np.genfromtxt(in_file, delimiter='\t',
                         dtype=float, skip_header=1)

    return header, data


def cal_plot(x_arr, y_arr, prefix, factor, settings):
    """Writing out a plot with a name based on the prefix name
    and the factor.
    """

    # Create color palette
    base_colors = sns.color_palette("YlOrRd", len(settings))
    fig, ax = pt.subplots()
    pt.style.use('seaborn-whitegrid')

    if 'error_rate' in factor:
        pt.plot([0, 1], [0, 1], '--', c='gray')

    for idx, setting in enumerate(settings):
        pt.plot(x_arr[idx], y_arr[idx], linewidth=2,
                color=base_colors[idx], label=setting)

    pt.ylabel(factor, fontweight='bold')
    pt.xlabel('significance', fontweight='bold')
    pt.ylim((0, 1))
    pt.xlim((0, 1))
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower center', frameon=True,
                    ncol=1, bbox_to_anchor=(0.5, -0.4), fontsize=12)

    pt.savefig("%s.%s.png" % (prefix, factor), dpi=300, bbox_inches="tight")
    pt.clf()


def calibration_plots(library, header, prefix):
    """Calculating summary statistics and plotting them for a set of
    error levels.
    """

    settings = library.keys()
    x = np.copy(library[settings[0]][:, 0])

    for i in range(9, len(header)):
        factor = header[i]
        y_arr = []
        for setting in settings:
            y = library[setting][:, i]
            # print(y)
            y_masked = np.ma.masked_invalid(y)
            x_masked = np.ma.array(x, mask=y_masked.mask)
            y_arr.append(y_masked)

    x_arr = np.asarray([x_masked for _ in settings])
    y_arr = np.asarray(y_arr)

    cal_plot(x_arr, y_arr, prefix, factor, settings)


def main():
    """
    Driver function
    """

    # parse command line arguments
    args = ParseArgs()
    headers = set()
    library = {}

    # iterate over the inputfiles
    for inputfile in args.inputfiles:
        header, data = read_pred_summary(inputfile)
        headers.add(" ".join(header))
        library[inputfile] = data

    # check if all headers have same format
    assert len(headers) == 1, "Headers in inputfiles have different formats!"

    # create scatter plots
    calibration_plots(library, header, args.outputname)

if __name__ == "__main__":
    main()