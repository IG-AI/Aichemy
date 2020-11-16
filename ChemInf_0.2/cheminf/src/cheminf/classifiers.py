import random
from abc import ABC
from copy import copy

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from torch.nn import Module as NNModule

CLASSIFIER_TYPES = ['rndfor', 'nn']

class ChemInfClassifier(object):
    def __init__(self, classifier_type, config):
        self.type = classifier_type
        self.config = config
        self.architecture = self.init_classifier(classifier_type, config)

    @staticmethod
    def init_classifier(classifier_type, config):
        if classifier_type == 'rndfor':
            architecture = ClassifierRF(smooth=config.smooth)

        elif classifier_type == 'nn':
            architecture = ClassifierNN(dim_in=config.dim_in,
                                        dim_hidden=config.dim_hidden,
                                        dim_out=config.dim_out,
                                        dropout=config.dropout)

        else:
            raise ValueError("Invalid classifier type")

        return architecture

    def get(self):
        return self.architecture

    def new(self):
        return self.init_classifier(classifier_type=self.type, config=self.config)

    def copy(self):
        return copy(self.architecture)

    def reset(self):
        self.architecture.reset()
        return self.architecture


class ClassifierRF(RandomForestClassifier):
    """Inherits from RandomForestClassifier"""
    def __init__(self, smooth=True):
        super(ClassifierRF, self).__init__()
        global bisect_right
        global bisect_left
        _temp = __import__('bisect', globals(), locals(), ['bisect_right', 'bisect_left'])
        bisect_right = _temp.bisect_right
        bisect_left = _temp.bisect_left
        self.smooth = smooth

    def nonconformity_scores(self, data):
        """Here you define the nonconformity function for your classifier.
        """
        nonconformity_scores = (lambda x: 1 - x)(self.predict_proba(data).app)

        return nonconformity_scores

    def cali_nonconf_scores(self, calibration_data):
        """Determine the conformity scores of the calibration data.
        Get prediction probabilities for all classes and store them if
        they belong to the true class, in separate vectors.
        """
        calibration_alphas = self.nonconformity_scores(calibration_data.values[:, 1:])
        # Get number of classes
        nr_class = len(calibration_alphas[0])
        # Iterate over all classes and retrieve calibration scores
        for c in range(nr_class):
            calibration_alpha_c = np.array([calibration_alphas[i][c]
                                            for i in range(len(calibration_alphas))
                                            if calibration_data.values[i, 0] == c])

            # Sorting arrays in-place without a copy (lowest to highest).
            calibration_alpha_c.sort()
            yield calibration_alpha_c

    def get_CP_p_value(self, nonconf_score_c, calibration_alphas_c):
        """Returns the p-value of a sample as determined from
        the calibration set's conformity scores and the
        positive and negative conformity scores for a given sample.
        """
        # Getting indices for the conformity scores from the sorted lists.
        #  If the current alpha (config score) would be inserted into the
        #  sorted calibration config scores they would be located at the
        #  following position.
        size_cal_list = len(calibration_alphas_c)
        index_p_c = bisect_left(calibration_alphas_c, nonconf_score_c)
        n_equal_or_over = size_cal_list - index_p_c
        # The relative position is outputted as the conformal
        # prediction's p value.
        if not self.smooth:
            p_c = float(n_equal_or_over) / float(size_cal_list + 1)
        else:
            right_index = bisect_right(calibration_alphas_c, nonconf_score_c)
            n_equal = right_index - index_p_c
            n_over = n_equal_or_over - n_equal
            p_c = (n_over + (n_equal * random.random())) / (float(size_cal_list + 1))
        return p_c

    def reset(self):
        clone(self)


class ClassifierNN(NNModule, ABC):
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int, dropout: float):
        super(ClassifierNN, self).__init__()
        global nn
        _temp = __import__('torch', globals(), locals(), ['nn'])
        nn = _temp.nn
        self.layer_depth = len(dim_hidden) + 1
        dim_hidden.insert(0, dim_in)
        dim_hidden.append(dim_out)
        self.layer_dimensions = dim_hidden
        self.layer_dropout_proc = dropout

        fc = nn.ModuleList([])
        for i in range(self.layer_depth):
            fc.append(nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
        self.fc = fc
        self.dropout = nn.Dropout(p=self.layer_dropout_proc)

    def forward(self, x):
        for i in range(self.layer_depth - 1):
            x = self.dropout(nn.functional.relu(self.fc[i](x)))
        x = self.fc[self.layer_depth - 1](x)
        return x

    def reset(self):
        fc = nn.ModuleList([])
        for i in range(self.layer_depth):
            fc.append(nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
        self.fc = fc
        self.dropout = nn.Dropout(p=self.layer_dropout_proc)
        return self
