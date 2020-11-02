import os
import re
import time

import numpy as np
import pandas as pd

from ..cheminf.preprocessing import shuffle_dataframe
from ..cheminf.controller import ChemInfController


class ChemInfOperator(object):
    def __init__(self):
        self.controller = ChemInfController()
        self.mode = self.controller.args.mode
        self.classifier = self.controller.args.classifier
        if self.mode == 'postproc':
            from src.cheminf.preprocessing import ChemInfPreProc
            self.postproc = ChemInfPreProc(self.controller)
        elif self.mode == 'preproc':
            from src.cheminf.postprocessing import ChemInfPostProc
            self.preproc = ChemInfPostProc(self.controller)
        elif self.mode == 'auto':
            if self.controller.config.exec.auto_resample:
                from src.cheminf.postprocessing import ChemInfPostProc
                self.preproc = ChemInfPostProc(self.controller)
            else:
                self.preproc = None

            if hasattr(self.controller.config, 'auto_summery') or hasattr(self.controller.config, 'auto_plot'):
                from src.cheminf.preprocessing import ChemInfPreProc
                self.postproc = ChemInfPreProc(self.controller)
            else:
                self.postproc = None

        if self.mode in self.controller.model_modes or self.mode in self.controller.auto_modes:
            if self.controller.args.classifier == 'all':
                models = self.init_model(self.controller)
                self.rndfor_model = models[0]
                self.nn_model = models[1]
            else:
                if self.controller.args.classifier == 'rndfor':
                    self.rndfor_model = self.init_model(self.controller)
                    self.nn_model = None
                elif self.controller.args.classifier == 'nn':
                    self.nn_model = self.init_model(self.controller)
                    self.rndfor_model = None
                else:
                    raise ValueError("Not a supported classifier")

    @staticmethod
    def init_model(controller):
        if controller.args.classifier == 'all':
            from src.cheminf.models import ModelRNDFOR
            from src.cheminf.models import ModelNN
            models = [ModelRNDFOR(controller), ModelNN(controller)]
            return models
        elif controller.args.classifier == 'rndfor':
            from src.cheminf.models import ModelRNDFOR
            model = ModelRNDFOR(controller)
        elif controller.args.classifier == 'nn':
            from src.cheminf.models import ModelNN
            model = ModelNN(controller)
        else:
            raise ValueError("Wrong classifier input, has to be either 'random_forest' or 'neural_network'")
        return model

    def get_model(self, classifier=None):
        if classifier is None:
            classifier = self.classifier
        return getattr(self, f"{classifier}_model")

    def get_utils(self, utils=None):
        if utils is None:
            utils = self.mode
        return getattr(self, utils)

    def run(self):
        if hasattr(self, 'classifier'):
            if self.classifier == 'all':
                classifiers = self.controller.classifier_types
            else:
                classifiers = [self.classifier]
        else:
            classifiers = [None]

        if self.mode == 'auto':
            for classifier in classifiers:
                model = self.get_model(classifier)
                model.update_datasets()
                model.build()
                model.predict()
            self.postproc.run_auto()
        elif self.mode == 'preproc':
            self.postproc.run()
        elif self.mode is 'postproc' or self.mode in self.controller.auto_modes:
            self.postproc.run()
        elif self.mode in self.controller.model_modes:
            for classifier in classifiers:
                model = self.get_model(classifier)
                model_activate = model.get(self.mode)
                model_activate()
        else:
            raise ValueError("Invalid operation mode")


class Timer(object):
    def __init__(self, func):
        self.args = ChemInfController.get('args')
        self.func = func

    def __call__(self, *args):
        begin = time.time()

        self.func(*args)

        end = time.time()
        run_time = np.round(end - begin, decimals=2)
        print("\nTIME PROFILE\n-----------------------------------")
        print(f"Runtime for {self.func.__name__} was {run_time}s")
        if self.func.__name__ == "multicore":
            print(f"Runtime per core(s) with {self.func.__name__} was {(run_time * self.args.num_core)}s")


def read_parameters(in_file, params_dict):
    """ Reads a file with hyperparameters for the
    classifier and parameters for the CP script. Updates the params_dict.
    """

    with open(in_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            assert len(items) == 2, \
                "The parameter file should have one key value pair per line"
            params_dict[items[0]] = items[1]


def read_array(in_file, data_type):
    """Reads a white space separated file without a header.
    The first column contains the IDs of samples and the 2nd the class.
    All the following columns contain features (the independent variables)
    of the sample. The function first determines the size of the arrays to
    be created, then inserts data into them.
    """
    print(f"Reading from '{in_file}'.")

    # read (compressed) features
    file_name, extension = os.path.splitext(in_file)
    if extension == ".bz2":
        import bz2
        f = bz2.open(in_file, 'rb')
    elif extension == ".gz":
        import gzip
        f = gzip.open(in_file, 'rb')
    else:
        f = open(in_file, 'r')

    nrow = sum(1 for _ in f)
    f.seek(0)
    ncol = len(f.readline().split()) - 1  # Disregarding ID column.
    f.seek(0)
    # Initializing arrays.
    print(f"Initializing array of size {nrow} X {ncol}.")
    id = np.empty(nrow, dtype=np.dtype('U50'))
    print(f"Memory of ID_array(str): {(id.nbytes * 10 ** (-6))} MB.")
    if data_type == 'integer':
        data = np.empty((nrow, ncol), dtype=int)
    if data_type == 'float':
        data = np.empty((nrow, ncol), dtype=float)
    print(f"Memory of data_array: {(data.nbytes * 10 ** (-6))} MB.")

    for i, line in enumerate(f):
        if extension == ".bz2" or extension == ".gz":
            line = line.decode()
        sample_id, sample_data = line.strip().split(None, 1)
        id[i] = sample_id
        data[i] = sample_data.split()

    print(f"Read {nrow} samples in total.\n")

    # close file handle
    f.close()

    return id, data


def read_dataframe(file, chunksize=None, shuffle=False):
    print("\nReading from {file}".format(file=file))
    with open(file) as f:
        first_line = next(f)
        line_data = re.split(" |\t|[|]", first_line)
        num_cols = len(line_data)

    unnamed_columns = [str(i) for i in range(1, num_cols - 1)]
    columns_names = ['id', 'class'] + unnamed_columns

    columns_types = {'id': 'string', 'class': 'int8'}
    for i in range(1, (num_cols - 1)):
        columns_types[str(i)] = 'int8'

    if chunksize:
        dataframe = pd.read_csv(file,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                chunksize=chunksize,
                                iterator=True,
                                engine='c')

    else:
        dataframe = pd.read_csv(file,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                engine='c')

        if shuffle:
            dataframe = shuffle_dataframe(dataframe)

    return dataframe


def save_dataframe(dataframe, out_file):
    print(f"\nSave dataframe as csv to {out_file}")

    dataframe.to_csv(out_file,
                     index=False,
                     header=False,
                     sep='\t')


def remove_name_sep(string):
    return '_'.join(string.split('_')[:-1])


@Timer
def run_operator():
    cheminf = ChemInfOperator()
    cheminf.run()


if __name__ == "__main__":
    run_operator()

