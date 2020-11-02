import os

import matplotlib.pyplot as plt

import numpy as np


class ChemInfPostProc(object):
    def __init__(self, controller):
        self.path = controller.src_dir
        self.project_name = controller.project_name
        self.mode = controller.args.postproc_mode
        self.classifier = controller.args.classifier
        self.error_level = controller.config.error_level
        if controller.args.significance:
            self.significance = controller.args.significance
        else:
            self.significance = None
        if self.mode == 'auto':
            self.pred_files = controller.pred_files
            self.classifier_type = controller.classifier_type
            self.nr_classifiers = len(controller.classifier_type)
            self.src_dir = controller.src_dir
        else:
            self.out_file = controller.out_file
            self.pred_file = controller.pred_files[0]

    def run(self, out_files=None):
        if out_files is None:
            out_files = self.out_file
            if self.mode == 'summary':
                self.make_summary(out_files=out_files)
            elif self.mode == 'plot':
                self.make_plot()

    def run_auto(self):
        out_files = []
        for classifier in self.classifier_type:
            out_files.append(f"{self.src_dir}/data/predictions/{self.project_name}/"
                             f"{self.project_name}_{classifier}_summary.csv")

    def make_summary(self, files=None):
        def _make_summary(_pred_file, _out_file, _significance_list):
            summary_array = read_pred_file(_pred_file, _significance_list, self.error_level)
            write_pred_summary_file(_out_file, summary_array, _significance_list)

        if not self.significance:
            significance_list = [(1 / self.error_level) * i for i in range(1, self.error_level)]
        else:
            significance_list = [self.significance]

        if files is None:
            out_file = self.out_file
            pred_file = self.pred_file
            _make_summary(pred_file, out_file, significance_list)
        else:
            for pred_file, out_file in files:
                _make_summary(pred_file, out_file, significance_list)

    def make_plot(self, pred_files=None):
        def _make_plot(_pred_file, project_name):
            header, data = read_pred_summary(_pred_file)
            calibration_plots(header, data, project_name)

        if pred_files is None:
            pred_file = self.pred_file
            _make_plot(pred_file, self.project_name)
        else:
            for pred_file in pred_files:
                _make_plot(pred_file, self.project_name)

    def get(self, key):
        return getattr(self, key)


def write_pred_summary_file(out_file, summary_array, significance_list):
    """Calculates standard ML metrics and CP metrics based on the
    set/class-counts. Writes a summary file.
    """

    with open(out_file, 'w+') as fout:
        fout.write("significance\ttrue_pos\tfalse_pos\ttrue_neg\tfalse_neg\t"
                   "both_class0\tboth_class1\tnull_class0\tnull_class1\t"
                   "error_rate\terror_rate_class1\terror_rate_class0\t"
                   "efficiency\tefficiency_class1\tefficiency_class0\t"
                   "precision\tsensitivity\tf1_score\tpred1:total"
                   "\tharmonic\n")

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
            error_rate = (summary_array[i, 1] + summary_array[i, 3] +
                          summary_array[i, 6] + summary_array[i, 7]) / \
                         total_samples

            # (true_pos + both_class1) / total_positives.
            error_rate_class1 = (summary_array[i, 3] +
                                 summary_array[i, 7]) / \
                                total_positives

            # (true_neg + both_class0) / total_negatives.
            error_rate_class0 = (summary_array[i, 6] +
                                 summary_array[i, 1]) / \
                                total_negatives

            # (true_pos + false_pos + true_neg + false_neg) / total_samples
            efficiency = (summary_array[i, 0] + summary_array[i, 1] +
                          summary_array[i, 2] + summary_array[i, 3]) / \
                         total_samples

            # (true_pos + false_neg) / total_samples
            efficiency_1 = (summary_array[i, 0] +
                            summary_array[i, 3]) / \
                           total_positives

            # (true_pos g) / total_samples
            efficiency_0 = (summary_array[i, 1] +
                            summary_array[i, 2]) / \
                           total_negatives

            if summary_array[i, 0] + summary_array[i, 1] != 0:
                # true_pos / (true_pos + false_pos)
                precision = summary_array[i, 0] / \
                            (summary_array[i, 0] + summary_array[i, 1])

            if total_positives != 0:
                # Normally:
                #  true_pos / (true_pos + false_neg), AKA recall.
                # In CP:
                #  true_pos / total_positives.
                sensitivity = summary_array[i, 0] / total_positives

            if precision != 'n/a' and sensitivity != 'n/a':
                f1_score = (2 * precision * sensitivity) / \
                           (precision + sensitivity)

            if total_samples != 0 and total_samples != 'n/a':
                # (true_pos + false_pos) / total_samples
                pred1_ratio = (summary_array[i, 0] + summary_array[i, 1]) / \
                              total_samples

            if precision != 'n/a' and sensitivity != 'n/a' and \
                    pred1_ratio != 'n/a':
                # Harmonic mean for sensitivity, precision and 1-pred1_ratio.
                harmonic = (3 * (1 - pred1_ratio) * precision * sensitivity) / \
                           ((1 - pred1_ratio) * precision +
                            (1 - pred1_ratio) * sensitivity +
                            (precision * sensitivity))

            fout.write(f"{significance}\t{summary_array[i, 0]}\t"
                       f"{summary_array[i, 1]}\t{summary_array[i, 2]}\t"
                       f"{summary_array[i, 3]}\t{summary_array[i, 4]}\t"
                       f"{summary_array[i, 5]}\t{summary_array[i, 6]}\t"
                       f"{summary_array[i, 7]}\t{error_rate}\t"
                       f"{error_rate_class1}\t{error_rate_class0}\t"
                       f"{efficiency}\t{efficiency_1}\t{efficiency_0}\t"
                       f"{precision}\t{sensitivity}\t"
                       f"{f1_score}\t{pred1_ratio}\t{harmonic}\n")


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
        set_prediction = '{}'

    return set_prediction


def read_pred_file(in_file, significance_list, error_level):
    """This function will calculate values for the confusion
    matrix and the set numbers based on a list of significance levels.
    """

    summary_array = np.zeros((error_level - 1, 8), dtype=int)

    with open(in_file, 'r') as fin:
        for line in fin:
            line_list = line.strip().split()
            real_class = line_list[1]
            p0 = float(line_list[2])
            p1 = float(line_list[3])
            set_preds = [set_prediction(p0, p1, significance) for significance in significance_list]

            # Confusion matrix and set counts.
            for i, set_pred in enumerate(set_preds):
                if set_pred == '{0}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 2] += 1  # true_neg
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 3] += 1  # false_neg
                elif set_pred == '{0,1}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 4] += 1  # both_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 5] += 1  # both_class1
                elif set_pred == '{}':
                    if real_class == '0' or real_class == '0.0':
                        summary_array[i, 6] += 1  # null_class0
                    elif real_class == '1' or real_class == '1.0':
                        summary_array[i, 7] += 1  # null_class1
                elif set_pred == '{1}':
                    if real_class == '1' or real_class == '1.0':
                        summary_array[i, 0] += 1  # true_pos
                    elif real_class == '0' or real_class == '0.0':
                        summary_array[i, 1] += 1  # false_pos

    return summary_array


def read_pred_summary(in_file):
    """Reads a amcp_pred_summary file as outputted
    from the amcp_summarize_pred.py script.
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
        plt.plot([0, 1], [0, 1], '--', c='gray')
    plt.plot(x, y, linewidth=3)
    plt.ylabel(factor, fontweight='bold')
    plt.xlabel('significance', fontweight='bold')
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.savefig(f'{prefix}.{factor}.png', dpi=300)
    plt.clf()


def calibration_plots(header, data, prefix):
    """Calculating summary statistics and plotting them for a set of
    error levels.
    """

    x = np.copy(data[:, 0])

    for i in range(9, len(header)):
        factor = header[i]
        y = data[:, i]
        y_masked = np.ma.masked_invalid(y)
        x_masked = np.ma.array(x, mask=y_masked.mask)
        cal_plot(x_masked, y_masked, prefix, factor)
