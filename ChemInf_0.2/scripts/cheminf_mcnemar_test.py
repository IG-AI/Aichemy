
__author__  = 'Leonard Sparring'
__date__    = '2020-07-09'
__email__   = 'leo.sparring@gmail.com'

import argparse

import numpy as np
import pandas as pd
import statsmodels.stats.contingency_tables as sm


def argument_parser():
    """Specify what file will be read, and optionally, a
    prefix for the png file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--in_file_1', required=True,
            help='Specify the input file.')
    parser.add_argument('-i2', '--in_file_2', required=True,
            help='Specify the input file.')
    parser.add_argument('-s1', '--significance_1', required=True,
            help='Specify the significance.',
            type=np.float64)
    parser.add_argument('-s2', '--significance_2', required=True,
            help='Specify the significance.',
            type=np.float64)
    args = parser.parse_args()

    assert args.significance_1 <= 1 and args.significance_1 >= 0,\
        'The significance level must be a value between 0 and 1.'
    assert args.significance_2 <= 1 and args.significance_2 >= 0,\
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


def gen_line(in_file):
    with open(in_file) as f:
        for line in f:
            yield line.strip()


def contingency_tables(gens, s1, s2):


    df_sensitivity = pd.DataFrame({'f1_tp': [0, 0], 'f1_fn': [0, 0]},
                                  index=['f2_tp', 'f2_fn'])
    df_specificity = pd.DataFrame({'f1_tn': [0, 0], 'f1_fp': [0, 0]},
                                  index=['f2_tn', 'f2_fp'])
    df_accuracy = pd.DataFrame({'f1_correct': [0, 0], 'f1_wrong': [0, 0]},
                               index=['f2_correct', 'f2_wrong'])

    line_nr = 0

    for f1_line, f2_line in zip(*gens):
        line_nr += 1
        f1_items = f1_line.split()
        f2_items = f2_line.split()
        if line_nr > 2:
            if f1_items[0] == f2_items[0] and int(f1_items[1]) == int(f2_items[1]):
                real_class = int(f1_items[1])
                set_pred_1 = set_prediction(float(f1_items[2]),
                                            float(f1_items[3]), s1)
                set_pred_2 = set_prediction(float(f2_items[2]),
                                            float(f2_items[3]), s2)
                if real_class == 1:
                    if set_pred_1 == '{1}':
                        if set_pred_2 == '{1}':
                            df_sensitivity.loc['f2_tp', 'f1_tp'] += 1
                            df_accuracy.loc['f2_correct', 'f1_correct'] += 1
                        elif set_pred_2 != '{1}':
                            df_sensitivity.loc['f2_fn', 'f1_tp'] += 1
                            df_accuracy.loc['f2_wrong', 'f1_correct'] += 1
                    elif set_pred_1 == '{0}':
                        if set_pred_2 == '{1}':
                            df_sensitivity.loc['f2_tp', 'f1_fn'] += 1
                            df_accuracy.loc['f2_correct', 'f1_wrong'] += 1
                        elif set_pred_2 != '{1}':
                            df_sensitivity.loc['f2_fn', 'f1_fn'] += 1
                            df_accuracy.loc['f2_wrong', 'f1_wrong'] += 1
                elif real_class == 0:
                    if set_pred_1 == '{1}':
                        if set_pred_2 == '{0}':
                            df_specificity.loc['f2_tn', 'f1_fp'] += 1
                            df_accuracy.loc['f2_correct', 'f1_wrong'] += 1
                        elif set_pred_2 != '{0}':
                            df_specificity.loc['f2_fp', 'f1_fp'] += 1
                            df_accuracy.loc['f2_wrong', 'f1_wrong'] += 1
                    elif set_pred_1 == '{0}':
                        if set_pred_2 == '{0}':
                            df_specificity.loc['f2_tn', 'f1_tn'] += 1
                            df_accuracy.loc['f2_correct', 'f1_correct'] += 1
                        elif set_pred_2 != '{0}':
                            df_specificity.loc['f2_fp', 'f1_tn'] += 1
                            df_accuracy.loc['f2_wrong', 'f1_correct'] += 1

            else:
                print('Mismatching IDs at line {line_nr}')

    return df_sensitivity, df_specificity, df_accuracy


def main():
    """Run main program."""
    
    args = argument_parser()

    in_files = [args.in_file_1, args.in_file_2]
    gens = [gen_line(n) for n in in_files]

    (df_sensitivity, df_specificity,
            df_accuracy) = contingency_tables(gens,
                                              args.significance_1,
                                              args.significance_2)

    print('Sensitivity:')
    print(sm.mcnemar(df_sensitivity.values.tolist(), exact=False, correction=True))
    print('\nSpecificity:')
    print(sm.mcnemar(df_specificity.values.tolist(), exact=False, correction=True))
    print('\nAccuracy:')
    print(sm.mcnemar(df_accuracy.values.tolist(), exact=False, correction=True))


# McNemar for accuracy, precision and sensitivity.


if __name__ == "__main__":
    main()


