
__author__  = 'Leonard Sparring'
__date__    = '2020-01-23'
__email__   = 'leo.sparring@gmail.com'

"""This script is used render calibration plots for cheminf predictions.
You input a cheminf prediction summary file with one header line. It
outputs png plots.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def argument_parser():
    """Specify what file will be read, and optionally, a
    prefix for the png files.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', required=True,
            help='Specify the input file.')
    parser.add_argument('-p', '--prefix',
            help='Optionally, specify a prefix for the plots.')
    args = parser.parse_args()

    return args


def read_pred_summary(in_file):
   """Reads a amcp_pred_summary file as outputted
   from the cheminf_summarize_pred.py script.
   """

   with open(in_file, 'r') as fin:
       header = fin.readline().strip().split()

   data = np.genfromtxt(in_file, delimiter='\t',
                        dtype=float, skip_header=1)

   return header, data


def cal_plot(x, y, prefix, factor):
    """Writing out a plot with a name based on the prefix name
    and the factor.
    """

    plt.style.use('seaborn-darkgrid')

    if 'error_rate' in factor:
        plt.plot([0,1], [0,1], '--', c='gray')
    plt.plot(x, y, linewidth=3)
    plt.ylabel(factor, fontweight='bold')
    plt.xlabel('significance', fontweight='bold')
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.savefig(f'{prefix}.{factor}.png', dpi=300)
    plt.clf()


def calibration_plots(header, data, prefix):
    """Calculating summary statistics and plotting them for a set of
    error levels.
    """

    x = np.copy(data[:, 0])

    for i in range(9, len(header)):
        factor = header[i]
        y = data[:,i]
        y_masked = np.ma.masked_invalid(y) 
        x_masked = np.ma.array(x, mask=y_masked.mask)
        cal_plot(x_masked, y_masked, prefix, factor)


def main():
    """Run main program."""
    
    args = argument_parser()

    if args.prefix:
        prefix = args.prefix
    else:
        prefix = args.in_file

    header, data = read_pred_summary(args.in_file)
    calibration_plots(header, data, prefix)


if __name__ == "__main__":
    main()

