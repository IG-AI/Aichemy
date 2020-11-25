import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    def make_summary(self, infile, outfile):
        if not self.significance:
            significance_list = [(1 / self.error_level) * i for i in range(1, self.error_level)]
        else:
            significance_list = [self.significance]

        summary_array = read_pred_file(infile, significance_list, self.error_level)
        write_pred_summary_file(outfile, summary_array, significance_list)

    def make_plot(self, infiles, outfile):
        # parse command line arguments
        headers = set()
        library = {}

        # iterate over the input files
        for infile in infiles:
            header, data = read_pred_summary(infile)
            headers.add(" ".join(header))
            library[infile] = data

        # check if all headers have same format
        assert len(headers) == 1, "Headers in input files have different formats!"

        # create scatter plots
        calibration_plots(library, header, outfile)

    def get(self, key):
        return getattr(self, key)


class PostProcNormal(ChemInfPostProc):
    def __init__(self, controller):
        super(PostProcNormal, self).__init__(controller)
        self.mode = controller.args.postproc_mode
        self.infiles = controller.args.infiles
        self.outfile = controller.args.outfile

    def run(self):
        if self.mode == 'summary':
            self.make_summary(self.infiles, self.outfile)
        elif self.mode == 'plot':
            self.make_plot(self.infiles, self.outfile, self.plot_del_sum)
        else:
            raise ModeError("postproc", self.mode)


class PostProcAuto(ChemInfPostProc):
    def __init__(self, controller):
        super(PostProcAuto, self).__init__(controller)
        self.mode = controller.args.mode
        self.auto_plus_plot = (self.mode == 'auto' and controller.config.execute.auto_plus_plot)
        self.auto_plus_sum = (self.mode == 'auto' and controller.config.execute.auto_plus_sum)
        self.pred_files = controller.args.pred_files
        self.classifier_types = controller.classifier_types
        self.nr_classifiers = len(controller.classifier_types)
        self.src_dir = controller.src_dir

    def run(self):
        files = []
        if self.classifier == 'all':
            classifiers = self.classifier_types
        else:
            classifiers = [self.classifier]

        for classifier in classifiers:
            outfile = (f"{self.src_dir}/data/{self.name}/predictions/"
                       f"{self.name}_{classifier}_predictions_summary.csv")
            files.append([self.pred_files[classifier], outfile])

        for file_list in files:
            infile = file_list[0]
            outfile = file_list[1]
            print(infile)
            print(outfile)
            if self.auto_plus_plot:
                self.make_summary(infile, outfile)
                infile = outfile
                outfile_name, outfile_extension = os.path.splitext(outfile)
                output_plot = outfile_name.replace("_summary", '')
                self.make_plot([infile], output_plot)
                if not self.auto_plus_sum:
                    os.remove(outfile)
            elif self.auto_plus_sum:
                self.make_summary(infile, outfile)
            else:
                pass


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


def calibration_plots(library, header, prefix):
    """Calculating summary statistics and plotting them for a set of
    error levels.
    """

    settings = list(library.keys())
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

        _calibration_plots(x_arr, y_arr, prefix, factor, settings)


def _calibration_plots(x_arr, y_arr, prefix, factor, settings):
    """Writing out a plot with a name based on the prefix name
    and the factor.
    """
    # Create color palette
    base_colors = sns.color_palette("YlOrRd", len(settings))
    fig, ax = plt.subplots()
    plt.style.use('seaborn-whitegrid')

    if 'error_rate' in factor:
        plt.plot([0, 1], [0, 1], '--', c='gray')

    for idx, setting in enumerate(settings):
        plt.plot(x_arr[idx], y_arr[idx], linewidth=2,
                 color=base_colors[idx], label=setting)

    plt.ylabel(factor, fontweight='bold')
    plt.xlabel('significance', fontweight='bold')
    plt.ylim((0, 1))
    plt.xlim((0, 1))

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    filenames = [os.path.split(label)[1] for label in labels]
    lgd = ax.legend(handles, filenames, loc='lower center', frameon=True,
                    ncol=1, bbox_to_anchor=(0.5, -0.4), fontsize=12)

    plt.savefig("%s.%s.png" % (prefix, factor), dpi=300, bbox_inches="tight")
    plt.clf()
