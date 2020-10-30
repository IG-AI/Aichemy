
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
    parser.add_argument('-i', '--in_file', required=True,
            help='Specify the input file.')
    parser.add_argument('-s', '--significance', required=True,
            type=float,
            help='Specify the significance level.')
    parser.add_argument('-p', '--prefix',
            help='Optionally, specify a prefix for the plots.')
    parser.add_argument('-t', '--threshold', type=float,
            help='Optionally, add a vertical line at this x-value.')
    args = parser.parse_args()

    return args


def read_pred_file(in_file, significance):
    """
    """

    pred_in_1_set_scores = []
    all_scores = []

    with open(in_file, 'r') as fin:
        header = fin.readline().strip().split()
        next(fin)
        for line in fin:
            line_list = line.strip().split()
            score = float(line_list[0].split('_')[1])
            p1 = float(line_list[3])
            p0 = float(line_list[2])
            real_class = line_list[1]
            if p1 > significance and p0 < significance and score != 10000:
                pred_in_1_set_scores.append(score)
                all_scores.append(score)
            elif score != 10000:
                all_scores.append(score)

    return all_scores, pred_in_1_set_scores


def hist_plot(all_scores, pred_in_1_set_scores, prefix,
                                  significance, threshold):
    """
    """

    plt.style.use('seaborn-white')
    
    nr_samples = len(all_scores)
    nr_pred1 = len(pred_in_1_set_scores)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    print(statistics.mean(pred_in_1_set_scores))
    print(statistics.mean(all_scores))

    ax.hist(all_scores, color = 'blue', edgecolor = 'black',
                                          bins = int(180),
                                          label = 'All, %d samples' % nr_samples,
                                          density=True)
    ax.hist(pred_in_1_set_scores, color = 'red', edgecolor = 'black',
                                          bins = int(180),
                                          density=True,
                                          label = 'Pred in {1}, %d samples' % nr_pred1,
                                          alpha = 0.60)
    if threshold:
        plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)

    plt.ylabel('Density', fontweight='bold')
    plt.xlabel('Score', fontweight='bold')
    ax.set_xlim([-62, 15]) #
    #plt.xlim((-65, 10)) #


    # Show the legend.
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', frameon=True,
                                     ncol=3, bbox_to_anchor=(0.5,-0.1))
    # Save the plot.
    fig.savefig(f'{prefix}.s{significance}.hist.png', bbox_extra_artists=(lgd, ),
                                            bbox_inches='tight', dpi=300)
    plt.clf()

def main():
    """Run main program."""
    
    args = argument_parser()

    if args.prefix:
        prefix = args.prefix
    else:
        prefix = args.in_file

    all_scores, pred_in_1_set_scores = read_pred_file(args.in_file,
                                                 args.significance)

    hist_plot(all_scores, pred_in_1_set_scores, prefix,
                        args.significance, args.threshold)


if __name__ == "__main__":
    main()


