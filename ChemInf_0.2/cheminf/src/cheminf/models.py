import os
import sys

import cloudpickle
import numpy as np
from abc import ABCMeta, abstractmethod

from ..cheminf.classifiers import ChemInfClassifier
from ..cheminf.preprocessing import ChemInfPreProc, split_dataframe
from ..cheminf.utils import read_dataframe, split_array

class ChemInfModel(object, metaclass=ABCMeta):
    def __init__(self, controller, model_type):
        if controller.args.mode == 'auto':
            preproc = ChemInfPreProc(controller)
            paths = preproc.run()
            self.paths = paths
            self.auto_mode = True
        else:
            self.infile = controller.args.infile
            self.auto_mode = False

        self.type = model_type
        self.outfile = controller.args.pred_files[model_type]
        self.config = controller.config.classifier
        self.name = controller.args.name
        self.models_dir = controller.args.models_dir
        self.classifier = ChemInfClassifier(self.type, self.config)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def improve(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    def save_models(self, model=None, iteration=0):
        if model is None:
            model = self.classifier.architecture

        model_name = f"{self.models_dir}/{self.name}_{self.type}_m{iteration}.z"
        if os.path.isfile(model_name):
            os.remove(model_name)
        with open(model_name, mode='ab') as f:
            cloudpickle.dump(model, f)

    def save_scores(self, calibration_data, iteration=0, model=None):
        if model is None:
            model = self.classifier.architecture

        for c, alpha_c in enumerate(model.cali_nonconf_scores(calibration_data)):
            model_score = f"{self.models_dir}/{self.name}_{self.type}_calibration-α{c}_m{iteration}.z"
            if os.path.isfile(model_score):
                os.remove(model_score)
            with open(model_score, mode='ab') as f:
                cloudpickle.dump(alpha_c, f)

    def load_models(self):
        dir_files = os.listdir(self.models_dir)
        nr_models = sum([1 for f in dir_files if f.startswith(f"{self.name}_{self.type}_m")])
        models = []
        print(f"Loading models from {self.models_dir}")
        for i in range(nr_models):
            model_file = f"{self.models_dir}/{self.name}_{self.type}_m{i}.z"
            with open(model_file, 'rb') as f:
                models.append(cloudpickle.load(f))

        print("Loaded {nr_models} models.".format(nr_models=nr_models))
        return models

    def load_scores(self):
        # Number of classes is amount of model files, divided by model count
        #  minus 1, as 1 file is classification file
        nr_models = self.config.nr_models
        nr_model_files = sum([1 for f in os.listdir(self.models_dir) if f.endswith(".z")])
        nr_class = int((nr_model_files / nr_models) - 1)
        scores = [list() for _ in range(nr_class)]
        print(f"Loading scores from {self.models_dir}")

        for i in range(nr_models):
            for c in range(nr_class):
                score_file = f"{self.models_dir}/{self.name}_{self.type}_calibration-α{c}_m{i}.z"
                with open(score_file, 'rb') as f:
                    scores[c].append(cloudpickle.load(f))

        print(f"Loaded {nr_models} x {nr_class} scores.")
        return scores, nr_class

    def create_new(self):
        return self.classifier.new()

    def reset(self):
        return self.classifier.reset()

    def get(self, key):
        return getattr(self, key)

    def __get_dataframe(self, label):
        if self.auto_mode:
            file_path = self.paths[label]
        else:
            file_path = self.infile
        dataframe = read_dataframe(file_path)
        return dataframe


# Todo: Make it work in current framework, with dataframes
class ModelRNDFOR(ChemInfModel):
    def __init__(self, database):
        super(ModelRNDFOR, self).__init__(database, 'rndfor')
        if database.args.outfile2:
            self.outfile_train = database.args.outfile2

    def build(self, models=None):
        """Trains NR_MODELS models and saves them as compressed files
        in the MODELS_PATH directory along with the calibration
        conformity scores.
        """
        from ..cheminf.utils import shuffle_arrays_in_unison

        nr_models = self.config.nr_models
        prop_train_ratio = self.config.prop_train_ratio

        train_dataframe = self.__get_dataframe('train')

        train_id_dataframe, train_data_dataframe = split_dataframe(train_dataframe, index=2, axis=1)
        train_id = np.array([np.array(row.tolist()) for _, row in train_id_dataframe.iterrows()])
        train_data = np.array([np.array(row.tolist()) for _, row in train_data_dataframe.iterrows()])
        nr_of_training_samples = len(train_id)
        for model_iteration in range(nr_models):
            if models is None:
                model = self.reset()
            else:
                model = models[model_iteration]
            # Shuffling the sample_IDs and data of the training
            #  set in unison, without creating copies.
            shuffle_arrays_in_unison(train_id, train_data)
            # Splitting training data into proper train set and
            #  calibration set.
            prop_train_data, calibration_data = split_array(train_data,
                                                            array_size=nr_of_training_samples,
                                                            percent_to_first=prop_train_ratio)

            print(f"Now building model: {model_iteration}")
            model.fit(prop_train_data[:, 1:], prop_train_data[:, 0])
            # Saving models to disk.
            self.save_models(model_iteration)

            # Retrieving the calibration conformity scores.
            self.save_scores(calibration_data, model_iteration)

    def improve(self):
        models = self.load_models()
        self.build(models)

    def predict(self):
        """Reads the pickled models and calibration conformity scores.
        Initializes a fixed size numpy array. Reads in nrow lines of the
        input file and then predicts the samples in the batch with each
        ml_model. The median p-values are calculated and written out in the
        outfile. This is performed until all lines in the infile are predicted.
        """
        outfile_path, outfile = os.path.split(self.outfile)
        if not os.path.isdir(outfile_path):
            os.mkdir(outfile_path)

        test_dataframe = self.__get_dataframe('test')
        test_id, test_data = split_dataframe(test_dataframe, index=2, axis=1)
        ncol = test_data.shape(1)

        # Reading parameters
        nrow = self.config.pred_nrow  # To control memory.

        dir_files = os.listdir(self.models_dir)
        nr_of_models = sum([1 for f in dir_files if f.startswith(f"{self.name}_rndfor")])

        # Initializing list of pointers to model objects
        #  and calibration conformity score lists.
        models = self.load_models()
        calibration_alphas_c, nr_class = self.load_scores()

        with open(os.path.join(outfile_path, outfile), 'w+') as fout:
            sample_id = np.empty(nrow, dtype=np.dtype('U50'))
            predict_data = np.empty((nrow, ncol), dtype=int)

            # Three dimensional class array
            p_c_array = np.empty((nrow, nr_of_models, nr_class), dtype=float)
            print(f"Allocated memory for an {nrow} X {ncol} array.")

            for i, row in test_data.iterrows():
                # Reading in nrow samples.
                if i % nrow != nrow - 1:
                    # Inserting data from the infile into np.arrays.
                    (sample_id[i % nrow],
                     sample_data) = row.strip().split(None, 1)
                    predict_data[i % nrow] = sample_data.split()
                else:
                    (sample_id[i % nrow],
                     sample_data) = row.strip().split(None, 1)
                    predict_data[i % nrow] = sample_data.split()
                    # The array has now been filled.
                    for model_index, model in enumerate(models):
                        # Predicting and getting p_values for each model
                        #  and sample.
                        predict_alphas = model.nonconformity_scores(predict_data[:, 1:])

                        # Iterate over the classes
                        for c, predict_alpha_c in enumerate(zip(*predict_alphas)):
                            for sample_index in range(nrow):
                                p_c = model.get_CP_p_value(predict_alpha_c[sample_index],
                                                           calibration_alphas_c[c][model_index])

                                p_c_array[sample_index, model_index, c] = p_c

                        # Calculating median p for each sample in the array, class c
                        p_c_medians = np.median(p_c_array, axis=1)

                    # Writing out sample prediction.
                    print(f"Predicted samples: {i + 1}.")
                    for j in range(nrow):
                        p_c_string = "\t".join([str(p_c_medians[j, c]) for c in range(nr_class)])
                        fout.write(f"{sample_id[j]}\t"
                                   f"{predict_data[j, 0]}\t"
                                   f"{p_c_string}\n")

            # After all samples have been read, the array still contains some
            #  samples that have not been predicted, as the 'else' was never
            #  encountered. These samples are handled and written out below.
            if i % nrow != nrow - 1:
                for model_index, model in enumerate(models):
                    predict_alphas = model.nonconformity_scores(predict_data[:i % nrow + 1, 1:])

                    # Iterate over the classes
                    for c, predict_alpha_c in enumerate(zip(*predict_alphas)):

                        for sample_index in range(i % nrow + 1):
                            p_c = model.get_CP_p_value(predict_alpha_c[sample_index],
                                                       calibration_alphas_c[c][model_index])

                            p_c_array[sample_index, model_index, c] = p_c

                # Calculate remaining p-value medians.
                p_c_medians = np.median(p_c_array[:i % nrow + 1, :], axis=1)

                # Writing out final samples predictions.
                for i in range(i % nrow + 1):
                    p_c_string = "\t".join([str(p_c_medians[i, c]) for c in range(nr_class)])
                    fout.write(f"{sample_id[i]}\t"
                               f"{predict_data[i, 0]}\t"
                               f"{p_c_string}\n")

    # Todo: Make validate submode work in current framework
    def validate(self):
        """Cross validation using the K-fold method.
        """
        from ..cheminf.utils import read_array, shuffle_arrays_in_unison

        # Reading parameters
        nr_models = self.config.nr_models
        prop_train_ratio = self.config.prop_train_ratio
        val_folds = self.config.val_folds

        # Reading in file to use for validation.
        data_type = self.config.data_type
        val_id, val_data = read_array(self.infile, data_type)
        nr_of_val_samples = len(val_id)
        val_indices = np.array(range(nr_of_val_samples))
        nr_holdout_samples = int(nr_of_val_samples / val_folds)
        nr_class = len(np.unique(val_data[:, 0]))

        # Out file(s).
        fout_test = open(self.outfile, 'w+')
        class_string = "\t".join(['p(%d)' % c for c in range(nr_class)])
        fout_test.write(f"validation\ttest_samples\tvalidation_file:\"{self.infile}\"\n"
                        f"sampleID\treal_class\t{class_string}\n")

        if self.outfile_train:
            fout_train = open(self.outfile_train, 'w+')
            fout_train.write(f"validation\ttrain_samples\tvalidation_file:\"{self.infile}\"\n"
                             f"sampleID\treal_class\t{class_string}\n")

        for k in range(val_folds):
            print(f"\nBuilding and predicting for cross validation chunk {k}.")
            test_indices = val_indices[k * nr_holdout_samples:(k + 1) * nr_holdout_samples]

            training_indices = list(set(val_indices) - set(test_indices))

            test_data = val_data[test_indices]
            test_id = val_id[test_indices]
            nr_of_test_samples = len(test_id)

            training_data = val_data[training_indices]
            training_id = val_id[training_indices]
            nr_of_training_samples = len(training_id)
            print("Copied validation samples to training and test arrays")

            # Three dimensional class array. Initialize/reset.
            p_c_array = np.empty((nr_of_test_samples, nr_models, nr_class), dtype=float)
            if self.outfile_train:
                p_c_array_training = np.empty((nr_of_training_samples,
                                               nr_models, nr_class), dtype=float)
            print(f"Allocated memory for array.")

            for model_iteration in range(nr_models):
                model = self.reset()
                # Shuffling the sample_IDs and data of the training
                #  set in unison, without creating copies.
                shuffle_arrays_in_unison(training_id, training_data)
                # Splitting training data into proper train set and
                #  calibration set.
                prop_train_data, calibration_data = split_array(training_data,
                                                                array_size=nr_of_training_samples,
                                                                percent_to_first=prop_train_ratio)

                print(f"Now building model: {model_iteration}")
                model.fit(prop_train_data[:, 1:], prop_train_data[:, 0])

                # Initializing/resetting calibration alpha lists.
                calibration_alphas_c = []

                # Retrieving the calibration conformity scores.
                for c, alpha_c in enumerate(model.cali_nonconf_scores(calibration_data)):
                    calibration_alphas_c.append(alpha_c)

                test_alphas = model.nonconformity_scores(test_data[:, 1:])
                if self.outfile_train:
                    training_alphas = model.nonconformity_scores(training_data[:, 1:])

                # Iterate over the classes.
                for c, test_alpha_c in enumerate(zip(*test_alphas)):
                    for sample_index in range(nr_of_test_samples):
                        p_c = model.get_CP_p_value(test_alpha_c[sample_index],
                                                   calibration_alphas_c[c])

                        p_c_array[sample_index, model_iteration, c] = p_c

                # Iterate over the classes for the training samples.
                if self.outfile_train:
                    for c, training_alpha_c in enumerate(zip(*training_alphas)):
                        for sample_index in range(nr_of_training_samples):
                            p_c = model.get_CP_p_value(training_alpha_c[sample_index],
                                                       calibration_alphas_c[c])

                            p_c_array_training[sample_index, model_iteration, c] = p_c

            # Calculating median p for each sample in the array, class c
            p_c_medians = np.median(p_c_array, axis=1)

            # Writing out sample prediction.
            for i in range(nr_of_test_samples):
                p_c_string = "\t".join([str(p_c_medians[i, c]) for c in range(nr_class)])
                fout_test.write(f"{test_id[i]}\t"
                                f"{test_data[i, 0]}\t"
                                f"{p_c_string}\n")

            # Calculating median p for each sample in the array, class c
            if self.outfile_train:
                p_c_medians_training = np.median(p_c_array_training, axis=1)
                for i in range(nr_of_training_samples):
                    p_c_string = "\t".join([str(p_c_medians_training[i, c]) for c in range(nr_class)])
                    fout_train.write(f"{training_id[i]}\t"
                                     f"{training_data[i, 0]}\t"
                                     f"{p_c_string}\n")

        fout_test.close()
        if self.outfile_train:
            fout_train.close()


class ModelNN(ChemInfModel):
    def __init__(self, database):
        super(ModelNN, self).__init__(database, 'nn')

    def build(self, models=None):
        import torch
        from skorch import NeuralNetClassifier
        from skorch.callbacks import EarlyStopping, EpochScoring
        from skorch.dataset import Dataset
        from skorch.helper import predefined_split
        from torch import nn

        from libs.nonconformist.base import ClassifierAdapter
        from libs.nonconformist.icp import IcpClassifier
        from libs.nonconformist.nc import ClassifierNc, MarginErrFunc
        from libs.torchtools.optim import RangerLars

        train_dataframe = self.__get_dataframe('train')

        nr_models = self.config.nr_models
        val_ratio = self.config.val_ratio
        cal_ratio = self.config.cal_ratio
        batch_size = self.config.batch_size
        max_epochs = self.config.max_epochs
        early_stop_patience = self.config.early_stop_patience
        early_stop_threshold = self.config.early_stop_threshold

        X = np.array(train_dataframe.iloc[:, 2:]).astype(np.float32)
        y = np.array(train_dataframe['class']).astype(np.int64)

        for i in range(nr_models):
            if models is None:
                classifier = self.reset()
            else:
                classifier = models[i]

            # Setup proper training, calibration and validation sets

            # validation set created
            total_train = np.arange(len(train_dataframe.index))
            valid_set, train_set = split_array(total_train, val_ratio, shuffle=True)

            # calib set and proper training set created
            calib_set, proper_train_set = split_array(train_set, cal_ratio, shuffle=True)

            # Convert validation to skorch dataset
            valid_dataset = Dataset(X[valid_set], y[valid_set])

            # Calculate number of training examples for each class (for weights)
            nr_class0 = len([x for x in y[proper_train_set] if x == 0])
            nr_class1 = len([x for x in y[proper_train_set] if x == 1])

            # Setup for class weights
            class_weights = 1 / torch.FloatTensor([nr_class0, nr_class1])

            # Define the skorch classifier
            minus_ba = EpochScoring(minus_bacc,
                                    name='-BA',
                                    on_train=False,
                                    use_caching=False,
                                    lower_is_better=True)

            early_stop = EarlyStopping(patience=early_stop_patience,
                                       threshold=early_stop_threshold,
                                       threshold_mode='rel',
                                       lower_is_better=True)

            model = NeuralNetClassifier(classifier, batch_size=batch_size, max_epochs=max_epochs,
                                        train_split=predefined_split(valid_dataset),  # Use predefined validation set
                                        optimizer=RangerLars,
                                        optimizer__lr=0.001,
                                        optimizer__weight_decay=0.1,
                                        criterion=nn.CrossEntropyLoss,
                                        criterion__weight=class_weights,
                                        callbacks=[minus_ba, early_stop])

            mem_info = f"\nMEMORY INFORMATION FOR MODEL: {i}\n" \
                       f"------------------------------------------------------\n" \
                       f"Tensor size: {sys.getsizeof(model)}\n" \
                       f"Element size: {class_weights.element_size()}\n" \
                       f"Number of elements: {class_weights.nelement()}\n" \
                       f"Element memory size: {(class_weights.nelement() * class_weights.element_size())}" \
                       f"\nBatch size: {model.batch_size}" \
                       f"\n------------------------------------------------------\n"

            print(mem_info)

            print(f"Working on model {i}")
            icp = IcpClassifier(ClassifierNc(ClassifierAdapter(model), MarginErrFunc()))
            icp.fit(X[proper_train_set], y[proper_train_set])
            icp.calibrate(X[calib_set], y[calib_set])

            self.save_models(model=icp, iteration=i)

    def improve(self):
        models = self.load_models()
        self.build(models)

    def predict(self):
        import pandas as pd

        test_dataframe = self.__get_dataframe('test')
        nr_models = self.config.nr_models
        sig = self.config.pred_sig

        models = self.load_models()

        X = np.array(test_dataframe.iloc[:, 2:]).astype(np.float32)

        p_value_results = []
        for i in range(nr_models):
            icp = models[i]

            print(f"Predicting from model {i}")

            pred = icp.predict(X, significance=sig)
            pred = np.round(pred, 6).astype('str')

            p_value = {'P(0)': pred[:, 0].tolist(),
                       'P(1)': pred[:, 1].tolist()}

            p_value_results.append(pd.DataFrame(p_value).astype('float32'))

        p_value_dataframe = pd.concat(p_value_results, axis=1).groupby(level=0, axis=1).median()
        results_dataframe = pd.concat([test_dataframe['id'], test_dataframe['class'], p_value_dataframe], axis=1)

        results_dataframe.to_csv(self.outfile, sep='\t', mode='w+', index=False, header=True)

    # Todo: Implement validate submode for nn
    def validate(self):
        raise NotImplementedError("Validation for neural network models aren't implemented yet.")


def minus_bacc(net, X=None, y=None):
    from sklearn.metrics import balanced_accuracy_score
    y_true = y
    y_pred = net.predict(X)
    return -balanced_accuracy_score(y_true, y_pred)
