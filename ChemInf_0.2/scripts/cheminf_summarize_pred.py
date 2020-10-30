
__author__  = 'Leonard Sparring'
__date__    = '2020-01-20'
__email__   = 'leo.sparring@gmail.com'

import re

"""This script is used to get summary statistics following an
cheminf prediction. You input an cheminf prediction output file with
two header lines. subsequent to this you can plot the statistics
with the cheminf_plot_summary.py script.
"""

import argparse
import numpy as np

ERR_LEV_STEPS = 50

def argument_parser():
    """Specify what file will be read,
    and the name of the file to be outputted from the script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', required=True,
            help='Specify the input file.')
    parser.add_argument('-o', '--out_file', required=True,
            help='Specify the output file.')
    parser.add_argument('-s', '--significance',
            help=('If you only want results for a single '
                 'significance level, give a value between '
                 '0 and 1.'),
            type=float)
    args = parser.parse_args()

    if args.significance:
        assert args.significance <= 1 and args.significance >= 0,\
            'The significance level must be a value between 0 and 1.'

    return args


def set_prediction(p0, p1, significance):
    """Determines the classification set of a sample."""

    if p0 > significance:
        if p1 > significance:
            set_prediction = '{0,1}'
        else:
            set_prediction = '{0}'
    elif p1 > significance:
            set_prediction = '{1}'
    else:
        set_prediction ='{}'

    return set_prediction


def read_pred_file(in_file, significance_list):
    """This function will calculate values for the confusion
    matrix and the set numbers based on a list of significance levels.
    """

    summary_array = np.zeros((ERR_LEV_STEPS-1, 8), dtype=int)
    
    with open(in_file, 'r') as fin:
        for line in fin:
            word_list = re.split(" +|\t", line)
            is_header = not any(word.isdigit() for word in word_list)
            if is_header:
                print(line)
                continue
            line_list = line.strip().split()
            real_class = line_list[1]
            p0 = float(line_list[2])
            p1 = float(line_list[3])
            set_preds = [ set_prediction(p0, p1, significance) for \
                           significance in significance_list ]

            # Confusion matrix and set counts.
            for i, set_pred in enumerate(set_preds):
                if set_pred == '{0}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 2] += 1 # true_neg
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 3] += 1 # false_neg
                elif set_pred == '{0,1}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 4] += 1 # both_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 5] += 1 # both_class1
                elif set_pred == '{}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 6] += 1 # null_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 7] += 1 # null_class1
                elif set_pred == '{1}':
                    if real_class == '1' or real_class == '1.0':
                        summary_array[i, 0] += 1 # true_pos
                    elif real_class == '0' or real_class == '0.0':
                        summary_array[i, 1] += 1 # false_pos

    return summary_array


def write_pred_summary_file(out_file, summary_array, significance_list):
    """Calculates standard ML metrics and CP metrics based on the
    set/class-counts. Writes a summary file.
    """

    with open(out_file, 'w+') as fout:
        fout.write('significance\ttrue_pos\tfalse_pos\ttrue_neg\tfalse_neg\t'
                   'both_class0\tboth_class1\tnull_class0\tnull_class1\t'
                   'error_rate\terror_rate_class1\terror_rate_class0\t'
                   'efficiency\tefficiency_class1\tefficiency_class0\t'
                   'precision\tsensitivity\tf1_score\tpred1:total'
                   '\tharmonic\n')

        total_samples = sum(summary_array[0, :]) 
        total_positives = (summary_array[0, 0] + summary_array[0, 7] + summary_array[0, 5] + summary_array[0, 3])
        total_negatives = (summary_array[0, 2] + summary_array[0, 4] + summary_array[0, 6] + summary_array[0, 1])

        for i, significance in enumerate(significance_list):
            precision = 'n/a'
            sensitivity = 'n/a'
            f1_score = 'n/a'
            pred1_ratio = 'n/a'
            harmonic = 'n/a'

            # (false_pos + false_neg + null_set) / total_samples
            error_rate = (summary_array[i, 1] + summary_array[i, 3] +\
                          summary_array[i, 6] + summary_array[i, 7]) /\
                          total_samples

            # (true_pos + both_class1) / total_positives.
            error_rate_class1 = (summary_array[i, 3] +\
                                 summary_array[i, 7]) /\
                                 total_positives

            # (true_neg + both_class0) / total_negatives.
            error_rate_class0 = (summary_array[i, 6] +\
                                 summary_array[i, 1]) /\
                                 total_negatives

            # (true_pos + false_pos + true_neg + false_neg) / total_samples
            efficiency = (summary_array[i, 0] + summary_array[i, 1] +\
                          summary_array[i, 2] + summary_array[i, 3]) /\
                          total_samples

            # (true_pos + false_neg) / total_samples
            efficiency_1 = (summary_array[i, 0] +\
                            summary_array[i, 3]) /\
                            total_positives

            # (true_pos g) / total_samples
            efficiency_0 = (summary_array[i, 1] +\
                            summary_array[i, 2]) /\
                            total_negatives

            if summary_array[i, 0] + summary_array[i, 1] != 0:
                # true_pos / (true_pos + false_pos)
                precision = summary_array[i, 0] /\
                           (summary_array[i, 0] + summary_array[i, 1])

            if total_positives != 0:
                # Normally:
                #  true_pos / (true_pos + false_neg), AKA recall.
                # In CP:
                #  true_pos / total_positives.
                sensitivity = summary_array[i, 0] / total_positives

            if precision != 'n/a' and sensitivity != 'n/a':
                f1_score =  (2 * precision * sensitivity) /\
                            (precision + sensitivity)

            if total_samples != 0 and total_samples != 'n/a':
                # (true_pos + false_pos) / total_samples
                pred1_ratio = (summary_array[i, 0] + summary_array[i, 1]) /\
                               total_samples

            if precision != 'n/a' and sensitivity != 'n/a' and\
                                          pred1_ratio != 'n/a':
                # Harmonic mean for sensitivity, precision and 1-pred1_ratio.
                harmonic = (3*(1-pred1_ratio)*precision*sensitivity)/\
                             ((1-pred1_ratio)*precision+\
                              (1-pred1_ratio)*sensitivity+\
                              (precision*sensitivity))

            fout.write(f'{significance}\t{summary_array[i,0]}\t'
                       f'{summary_array[i,1]}\t{summary_array[i,2]}\t'
                       f'{summary_array[i,3]}\t{summary_array[i,4]}\t'
                       f'{summary_array[i,5]}\t{summary_array[i,6]}\t'
                       f'{summary_array[i,7]}\t{error_rate}\t'
                       f'{error_rate_class1}\t{error_rate_class0}\t'
                       f'{efficiency}\t{efficiency_1}\t{efficiency_0}\t'
                       f'{precision}\t{sensitivity}\t'
                       f'{f1_score}\t{pred1_ratio}\t{harmonic}\n')


def main():
    """Run main program."""
    
    args = argument_parser()
    
    if not args.significance:
        significance_list = [ (1/ERR_LEV_STEPS) * i for i \
                              in range(1, ERR_LEV_STEPS) ]
    else:
        significance_list = [args.significance]

    summary_array = read_pred_file(args.in_file, significance_list)
    write_pred_summary_file(args.out_file, summary_array, significance_list)


if __name__ == "__main__":
    main()

