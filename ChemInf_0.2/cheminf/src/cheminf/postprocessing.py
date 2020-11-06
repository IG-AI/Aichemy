import os

import matplotlib.pyplot as plt

import numpy as np
from abc import ABCMeta, abstractmethod

from ..cheminf.utils import ModeError


class ChemInfPostProc(object, metaclass=ABCMeta):
    def __init__(self, controller):
        self.name = controller.args.name
        self.classifier = controller.args.classifier
        self.error_level = controller.config.execute.error_level
        self.plot_del_sum = controller.config.execute.plot_del_sum
        if controller.args.significance:
            self.significance = controller.args.significance
        else:
            self.significance = None

    @abstractmethod
    def run(self):
        pass

    def make_summary(self, files):
        for file_list in files:
            infile = file_list[0]
            outfile = file_list[1]
            if not self.significance:
                significance_list = [(1 / self.error_level) * i for i in range(1, self.error_level)]
            else:
                significance_list = [self.significance]
            summary_array = read_pred_file(infile, significance_list, self.error_level)
            write_pred_summary_file(outfile, summary_array, significance_list)

    def make_plot(self, files, del_sum):
        for file_list in files:
            infile = file_list[0]
            outfile = file_list[1]
            if not os.path.exists(outfile):
                self.make_summary([file_list])
            header, data = read_pred_summary(outfile)
            calibration_plots(header, data, infile)
            if del_sum:
                os.remove(outfile)

    def get(self, key):
        return getattr(self, key)


class PostProcNormal(ChemInfPostProc):
    def __init__(self, controller):
        super(PostProcNormal, self).__init__(controller)
        self.mode = controller.args.postproc_mode
        self.outfile = controller.args.outfile
        self.infile = controller.args.infile

    def run(self):
        files = [[self.infile, self.outfile]]
        if self.mode == 'summary':
            self.make_summary(files)
        elif self.mode == 'plot':
            self.make_plot(files, self.plot_del_sum)
        else:
            raise ModeError("postproc", self.mode)


class PostProcAuto(ChemInfPostProc):
    def __init__(self, controller):
        super(PostProcAuto, self).__init__(controller)
        self.mode = controller.args.mode
        self.auto_plus_plot = (self.mode == 'auto' and controller.config.execute.auto_plus_plot)
        self.auto_plus_sum = (self.mode == 'auto' and controller.config.execute.auto_plus_sum)
        self.pred_files = controller.args.pred_files
        self.classifier_type = controller.classifier_types
        self.nr_classifiers = len(controller.classifier_types)
        self.src_dir = controller.src_dir

    def run(self):
        files = []
        if self.classifier == 'all':
            classifiers = self.classifier_type
        else:
            classifiers = [self.classifier]
        for classifier in classifiers:
            outfile = (f"{self.src_dir}/data/predictions/{self.name}/"
                       f"{self.name}_{classifier}_predict_summary.csv")
            files.append([self.pred_files[classifier], outfile])

        if self.auto_plus_sum:
            self.make_summary(files)
        if self.auto_plus_plot:
            if self.auto_plus_sum:
                del_sum = False

def write_pred_summary_file(outfile, summary_array, significance_list):
    """Calculates standard ML metrics and CP metrics based on the
    set/class-counts. Writes a summary file.
    """
    with open(outfile, 'w+') as fout:
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
            _set_prediction = '{0,1}'
        else:
            _set_prediction = '{0}'
    elif p1 > significance:
        _set_prediction = '{1}'
    else:
        _set_prediction = '{}'

    return _set_prediction


def read_pred_file(infile, significance_list, error_level):
    """This function will calculate values for the confusion
    matrix and the set numbers based on a list of significance levels.
    """

    summary_array = np.zeros((error_level - 1, 8), dtype=int)

    with open(infile, 'r') as fin:
        next(fin)
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


def read_pred_summary(infile):
    """Reads a amcp_pred_summary file as outputted
    from the amcp_summarize_pred.py script.
    """
    with open(infile, 'r') as fin:
        header = fin.readline().strip().split()

    data = np.genfromtxt(infile, delimiter='\t',
                         dtype=float, skip_header=1)

    return header, data


def cal_plot(x, y, name, factor):
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
    plt.savefig(f'{name}_{factor}.png', dpi=300)
    plt.clf()


def calibration_plots(header, data, file_path):
    """Calculating summary statistics and plotting them for a set of
    error levels.
    """

    x = np.copy(data[:, 0])

    name, _ = os.path.splitext(file_path)

    for i in range(9, len(header)):
        factor = header[i]
        y = data[:, i]
        y_masked = np.ma.masked_invalid(y)
        x_masked = np.ma.array(x, mask=y_masked.mask)
        cal_plot(x_masked, y_masked, name, factor)
