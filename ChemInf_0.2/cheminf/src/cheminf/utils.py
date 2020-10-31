import argparse
import configparser
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cheminf.preprocessing import ChemInfPostProc
from cheminf.models import ModelRNDFOR, ModelNN

from src.cheminf.preprocessing import shuffle_dataframe


class ChemInfConfig(object):
    def __init__(self, classifier_type, config_file):
        config = configparser.SafeConfigParser()
        print(config_file)
        config.read(config_file)

        self.nr_models = int(config['all']['nr_models'])
        self.train_test_ratio = float(config['all']['train_test_ratio'])

        if classifier_type == 'rndfor':
            self.nr_trees = int(config['random_forest']['nr_of_trees'])
            self.n_jobs = int(config['random_forest']['n_jobs'])
            self.pred_nrow = float(config['random_forest']['pred_nrow']),
            self.val_folds = int(config['random_forest']['val_folds']),
            self.smooth = bool(config['random_forest']['smooth']),
            self.data_type = str(config['random_forest']['data_type'])

        elif classifier_type == 'nn':
            self.val_ratio = float(config['neural_network']['val_ratio'])
            self.cal_ratio = float(config['neural_network']['cal_ratio'])
            self.dim_in = int(config['neural_network']['dim_in'])
            self.dim_hidden = [int(x) for x in config['neural_network']['dim_hidden'].split(",")]
            self.dim_out = int(config['neural_network']['dim_out'])
            self.dropout = float(config['neural_network']['dropout'])
            self.batch_size = int(config['neural_network']['batch_size'])
            self.max_epochs = int(config['neural_network']['max_epochs'])
            self.early_stop_patience = int(config['neural_network']['early_stop_patience'])
            self.early_stop_threshold = float(config['neural_network']['early_stop_threshold'])

            try:
                self.pred_sig = int(config['neural_network']['pred_sig'])
            except ValueError:
                self.pred_sig = None

        else:
            raise ValueError("Config mode doesn't exist")


class ChemInfDataBase(object):
    path = None
    args = None
    config = None
    config_file = None
    initiated = False

    def __new__(cls, *args, **kwargs):
        if not cls.initiated:
            cls.initiated = True
            return super().__new__(cls)
        else:
            pass

    def __init__(self):
        source_path = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.dirname(source_path)
        self.args = self.argument_parser()
        if hasattr(self.args, 'config'):
            if self.args.config is not None:
                self.config_file = self.args.config
            else:
                self.config_file = "{path}/config/classifiers.ini".format(path=self.path)
        self.config = ChemInfConfig(self.args.classifier, self.config_file)
        self.config_file = self.args.config

    @classmethod
    def get_args(cls):
        return cls.args

    @classmethod
    def get_config(cls):
        return cls.config

    @classmethod
    def set_conf(cls, section, attribute, value):
        config_file = cls.config_file
        config = configparser.SafeConfigParser()
        config.read(config_file)
        config_update = {attribute: value}
        config.set(section, attribute, str(config_update))
        with open(config_file, "w+") as conf:
            config.write(conf)

        cls.config.attribute = value

    @classmethod
    def get_path(cls):
        return cls.path

    @staticmethod
    def argument_parser():
        """Specify what file will be read, what options to use
        and the name of the file to be outputted from the script.
        """

        parser = argparse.ArgumentParser(prog="cheminf")

        parser_command = parser.add_subparsers(help="Choose a mode to run cheminf in", dest='mode')

        parser_build = parser_command.add_parser('auto',
                                                 help="Automatic mode that preforms a complete data analysis of a "
                                                      "dataset. The dataset is split in train/test sets with a certain "
                                                      "percentage defined in the classifier config file. The train set"
                                                      "will be used to train model(s) and the test set will"
                                                      "be used to for prediction on the created models. This prediction"
                                                      " will then be used to make a summary of the model(s) performance"
                                                      "and graphs will be created based out if the result.")

        parser_build = parser_command.add_parser('build',
                                                 help="Builds a model with specific classifier")

        parser_improve = parser_command.add_parser('improve',
                                                   help="Improve a model with specific classifier")

        parser_predict = parser_command.add_parser('predict',
                                                   help="Predicts data with a classifier model")

        parser_validate = parser_command.add_parser('validate',
                                                    help="Preforms cross-validation on a classifier model")

        parser_postproc = parser_command.add_parser('pre-proc',
                                                    help="Preforms various post precessing operations")

        parser_preproc_mode = parser_postproc.add_subparsers(help="Choose a preprocessing mode to preform",
                                                             dest='preproc-mode')

        parser_postproc = parser_command.add_parser('post-proc',
                                                    help="Preforms various post precessing operations")

        parser_postproc_mode = parser_postproc.add_subparsers(help="Choose a post processing mode to preform",
                                                              dest='postproc-mode')

        parser_postproc_resample = parser_postproc_mode.add_parser('resample',
                                                                   help="Resample a datasets and saves it")

        parser_postproc_split = parser_postproc_mode.add_parser('split',
                                                                help="Splits a dataset and saves it")

        parser_postproc_trim = parser_postproc_mode.add_parser('trim',
                                                               help="Trims a datasets and saves it")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-cl', '--classifier',
                                   required=True,
                                   choices=['rndfor', 'nn'],
                                   help="Choose the underlying classifier")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate,
                          parser_postproc_resample, parser_postproc_split, parser_postproc_trim]:
            subparser.add_argument('-i', '--in_file',
                                   required=True,
                                   help="(For 'build, 'validate', 'predict') "
                                        "Specify the file containing the data.")

        date = datetime.now()
        current_date = date.strftime("%d-%m-%Y")
        for subparser in [parser_build, parser_improve, parser_predict, parser_validate,
                          parser_postproc_resample, parser_postproc_split, parser_postproc_trim]:
            subparser.add_argument('-n', '--name',
                                   default=current_date,
                                   help="Name of the current project, to which all out_puts "
                                        "will use as prefixes or in subdirectories with that name.")

        for subparser in [parser_predict, parser_validate, parser_postproc_resample,
                          parser_postproc_split, parser_postproc_trim]:
            subparser.add_argument('-o', '--out_file',
                                   help="Specify the output file with path. If it's not specified for \'predict\' "
                                        "then it will be sat to default (data/predictions)")

        for subparser in [parser_validate, parser_postproc_split]:
            subparser.add_argument('-o2', '--out_file2',
                                   help="Specify the second output file.")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-md', '--models_dir',
                                   help="Specify the path to the directory "
                                        "where the generated models should be stored for 'build' "
                                        "or 'predict'). Otherwise it will be the default model directory "
                                        "(data/models).")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate,
                          parser_postproc_resample, parser_postproc_split, parser_postproc_trim]:
            subparser.add_argument('-pc', '--percentage',
                                   help='Specify the split percentage.',
                                   type=float,
                                   default=0.5)

        for subparser in [parser_postproc_resample, parser_postproc_split, parser_postproc_trim]:
            subparser.add_argument('-sh', '--shuffle',
                                   default=False,
                                   action='store_true',
                                   help="Specify that the data should be shuffled")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-cfg', '--config',
                                   help="Specify file path to override the internal configuration file. "
                                        "The file needs to have the exact same format "
                                        "as the internal configuration file")

        for subparser in [parser_postproc_resample]:
            subparser.add_argument('-ch', '--chunksize',
                                   type=int,
                                   help="Specify size of chunks the files should be divided into.")

        for subparser in [parser_postproc_resample]:
            subparser.add_argument('-nc', '--num_core',
                                   default=1,
                                   type=int,
                                   help="Specify the amount of cores should be used.")

        args = parser.parse_args()

        if not Path(args.in_file).is_file():
            parser.error("The input file does not exist.")

        if hasattr(args, 'utils_mode'):
            if args.utils_mode == 'resample' and hasattr(args, 'num_core') and not hasattr(args, 'chunksize'):
                parser.error("Utils resample multicore mode has to be executed with chunking.")

            if (args.utils_mode == 'split' or args.utils_mode == 'trim') \
                    and (hasattr(args, 'num_core') or hasattr(args, 'chunksize')):
                parser.error(f"Utils {args.utils_mode} be can't executed in multicore mode or with chunking.")

        return args


class ChemInfOperator(object):
    def __init__(self):
        self.database = ChemInfDataBase()
        self.mode = self.database.args.mode
        if self.mode == 'utils':
            self.utils = ChemInfPostProc(self.database)
        else:
            self.model = self.init_model(self.database)

    @staticmethod
    def init_model(database):
        if database.args.classifier == 'rndfor':
            model = ModelRNDFOR(database)
        elif database.args.classifier == 'nn':
            model = ModelNN(database)
        else:
            raise ValueError("Wrong classifier input, has to be either 'random_forest' or 'neural_network'")

        return model


class Timer(object):
    def __init__(self, func):
        self.args = ChemInfDataBase.get_args()
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