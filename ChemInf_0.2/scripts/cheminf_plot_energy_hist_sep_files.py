
__author__  = 'Leonard Sparring'
__date__    = '2020-02-17'
__email__   = 'leo.sparring@gmail.com'

"""
"""

import argparse
import statistics

import matplotlib.pyplot as plt


def argument_parser():
    """Specify what file will be read, and optionally, a
    prefix for the png file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--in_file1', required=True,
            help='Specify the input file 1.')
    parser.add_argument('-i2', '--in_file2', required=True,
            help='Specify the input file 2.')
    parser.add_argument('-p', '--prefix',
            help='Optionally, specify a prefix for the plots.')
    parser.add_argument('-t', '--threshold', type=float,
            help='Optionally, add a vertical line at this x-value.')
    args = parser.parse_args()

    return args


def read_pred_file(in_file):
    """
    """

    scores = []

    with open(in_file, 'r') as fin:
        header = fin.readline().strip().split()
        next(fin)
        for line in fin:
            line_list = line.strip().split()
            score = float(line_list[1])
            scores.append(score)

    return scores


def hist_plot(scores_f1, scores_f2, prefix,
                                  threshold):
    """
    """

    plt.style.use('seaborn-white')
    
    nr_f1 = len(scores_f1)
    nr_f2 = len(scores_f2)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    print(statistics.mean(scores_f1))
    print(statistics.mean(scores_f2))

    ax.hist(scores_f1, color = 'blue',
                                          bins = int(180),
                                          label = 'Random, %d samples' % nr_f1,
                                          density=True,
                                          alpha = 0.60)
    ax.hist(scores_f2, color = 'red',
                                          bins = int(180),
                                          density=True,
                                          label = 'Selected molecules %d samples' % nr_f2,
                                          alpha = 0.60)
    if threshold:
        plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)

    plt.ylabel('Density', fontweight='bold')
    plt.xlabel('Score', fontweight='bold')
    ax.set_xlim([-70, 5]) #


    # Show the legend.
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', frameon=True,
                                     ncol=3, bbox_to_anchor=(0.5,-0.1))
    # Save the plot.
    fig.savefig(f'{prefix}.hist.png', bbox_extra_artists=(lgd, ),
                                            bbox_inches='tight', dpi=300)
    plt.clf()

def main():
    """Run main program."""
    
    args = argument_parser()

    if args.prefix:
        prefix = args.prefix
    else:
        prefix = args.in_file

    scores_f1 = read_pred_file(args.in_file1)
    scores_f2 = read_pred_file(args.in_file2)

    hist_plot(scores_f1, scores_f2, prefix, args.threshold)


if __name__ == "__main__":
    main()


