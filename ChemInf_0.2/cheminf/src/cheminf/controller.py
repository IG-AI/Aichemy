import configparser
import argparse
import os
from datetime import datetime
import numpy as np

from src.cheminf.classifiers import CLASSIFIER_TYPES

MODEL_MODES = ['build', 'improve', 'predict', 'validate']
DATA_MODES = ['postproc', 'preproc']
AUTO_MODES = ['auto']


class ChemInfInput(object):
    def __init__(self, config_file):
        self.args = self.argument_parser()
        self.config = ChemInfConfig(self.args.mode, self.args.classifier, config_file, self.args.config)

    @staticmethod
    def argument_parser():
        """Specify what file will be read, what options to use
            and the name of the file to be outputted from the script.
            """

        parser = argparse.ArgumentParser(prog="cheminf")

        parser.add_argument('--version', action='version', version='ChemInf Version 0.2')

        parser_command = parser.add_subparsers(help="Choose a mode to run cheminf in", dest='mode')

        parser_auto = parser_command.add_parser('auto',
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

        parser_preproc = parser_command.add_parser('preproc',
                                                   help="Preforms various post-precessing operations")

        parser_preproc_mode = parser_preproc.add_subparsers(help="Choose a post processing mode to preform",
                                                            dest='preproc_mode')

        parser_preproc_resample = parser_preproc_mode.add_parser('resample',
                                                                 help="Resample a datasets and saves it")

        parser_preproc_split = parser_preproc_mode.add_parser('split',
                                                              help="Splits a dataset and saves it")

        parser_preproc_trim = parser_preproc_mode.add_parser('trim',
                                                             help="Trims a datasets and saves it")

        parser_postproc = parser_command.add_parser('postproc',
                                                    help="Preforms various post-precessing operations")

        parser_postproc_mode = parser_postproc.add_subparsers(help="Choose a preprocessing mode to preform",
                                                              dest='postproc_mode')

        parser_postproc_summary = parser_postproc_mode.add_parser('summary',
                                                                  help="Choose a preprocessing mode to preform")

        parser_postproc_plot = parser_postproc_mode.add_parser('plot',
                                                               help="Choose a preprocessing mode to preform")

        for subparser in [parser_auto, parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-cl', '--classifier',
                                   default='all',
                                   choices=['rndfor', 'nn', 'all'],
                                   help="Choose one or all classifiers for model operations")

        for subparser in [parser_auto, parser_build, parser_improve, parser_predict,
                          parser_validate, parser_preproc_resample, parser_preproc_split, parser_preproc_trim,
                          parser_postproc_summary, parser_postproc_plot]:
            subparser.add_argument('-i', '--in_file',
                                   required=True,
                                   help="(For 'build, 'validate', 'predict') "
                                        "Specify the file containing the data.")

        date = datetime.now()
        current_date = date.strftime("%d-%m-%Y")
        for subparser in [parser_auto, parser_build, parser_improve, parser_predict,
                          parser_validate, parser_preproc_resample, parser_preproc_split, parser_preproc_trim,
                          parser_postproc_summary,
                          parser_postproc_plot]:
            subparser.add_argument('-n', '--name',
                                   default=current_date,
                                   help="Name of the current project, to which all out_puts "
                                        "will use as prefixes or in subdirectories with that name.")

        for subparser in [parser_predict, parser_validate, parser_preproc_resample,
                          parser_preproc_split, parser_preproc_trim, parser_postproc_summary, parser_postproc_plot]:
            subparser.add_argument('-o', '--out_file',
                                   default=None,
                                   help="Specify the output file with path. If it's not specified for \'predict\' "
                                        "then it will be sat to default (data/predictions)")

        for subparser in [parser_validate, parser_preproc_split]:
            subparser.add_argument('-o2', '--out_file2',
                                   default=None,
                                   help="Specify the second output file.")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-md', '--models_dir',
                                   default=None,
                                   help="Specify the path to the directory "
                                        "where the generated models should be stored for 'build' "
                                        "or 'predict'). Otherwise it will be the default model directory "
                                        "(data/models).")

        for subparser in [parser_preproc_split, parser_preproc_trim]:
            subparser.add_argument('-pc', '--percentage',
                                   type=float,
                                   default=0.5,
                                   help='Specify the split percentage.')

        for subparser in [parser_preproc_split, parser_preproc_trim]:
            subparser.add_argument('-sh', '--shuffle',
                                   default=False,
                                   action='store_true',
                                   help="Specify that the data should be shuffled")

        for subparser in [parser_postproc_summary]:
            subparser.add_argument('-sg', '--significance',
                                   default=None,
                                   type=float,
                                   help="If you only want results for a single "
                                        "significance level, give a value between "
                                        "0 and 1.")

        for subparser in [parser_auto, parser_build, parser_improve, parser_predict,
                          parser_validate]:
            subparser.add_argument('-ocf', '--override_config',
                                   default=None,
                                   help="Specify a parameter in the configuration that will to be override ")

        for subparser in [parser_preproc_resample]:
            subparser.add_argument('-ch', '--chunksize',
                                   default=None,
                                   type=int,
                                   help="Specify size of chunks the files should be divided into.")

        for subparser in [parser_preproc_resample]:
            subparser.add_argument('-nc', '--nr_core',
                                   default=1,
                                   type=int,
                                   help="Specify the amount of cores should be used.")

        args = parser.parse_args()

        if hasattr(args, 'utils_mode'):
            if args.utils_mode == 'resample' and hasattr(args, 'num_core') and not hasattr(args, 'chunksize'):
                parser.error("Utils resample multicore mode has to be executed with chunking.")

            if (args.utils_mode == 'split' or args.utils_mode == 'trim') \
                    and (hasattr(args, 'num_core') or hasattr(args, 'chunksize')):
                parser.error(f"Utils {args.utils_mode} be can't executed in multicore mode or with chunking.")

        return args


class ChemInfConfig(object):
    def __init__(self, operator_mode, classifier_type, config_files, overreader):
        self.clf_conf_file = config_files[0]
        self.exec_conf_file = config_files[1]
        self.execute = ConfigExec(operator_mode, self.exec_conf_file)
        if operator_mode in AUTO_MODES or operator_mode in MODEL_MODES:
            self.classifier = ConfigClf(classifier_type, self.clf_conf_file)
        else:
            self.classifier = None

    def update_config(self, overrider):
        new_configs = overrider.split(';')
        for config in new_configs:
            key, value = config.split(',')
            try:
                value_type = type(getattr(self.classifier, key))
                value = exec(value_type(value))
                setattr(self.classifier, key, value)
            except:
                value_type = type(getattr(self.execute, key))
                value = exec(value_type(value))
                setattr(self.execute, key, value)


class ConfigClf(object):
    def __init__(self, classifier_type, config_file):
        config = configparser.SafeConfigParser()
        config.read(config_file)

        self.nr_models = int(config['all']['nr_models'])

        if classifier_type == 'rndfor' or classifier_type == 'all':
            self.nr_trees = int(config['random_forest']['nr_of_trees'])
            self.n_jobs = int(config['random_forest']['n_jobs'])
            self.pred_nrow = float(config['random_forest']['pred_nrow']),
            self.val_folds = int(config['random_forest']['val_folds']),
            self.smooth = bool(config['random_forest']['smooth']),
            self.data_type = str(config['random_forest']['data_type'])

        elif classifier_type == 'nn' or classifier_type == 'all':
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
    
        


class ConfigExec(object):
    def __init__(self, operator_mode, config_file):
        config = configparser.SafeConfigParser()
        config.read(config_file)

        if operator_mode == 'auto':
            self.auto_plus_resample = int(config['auto']['auto+resample'])
            self.auto_plus_summery = float(config['auto']['auto+summery'])
            self.auto_plus_summery = float(config['auto']['auto+plot'])
            self.train_test_ratio = float(config['auto']['train_test_ratio'])
        elif operator_mode == 'postproc':
            self.error_level = int(config['postproc']['error_level'])


class ChemInfController(ChemInfInput):
    model_modes = MODEL_MODES
    data_modes = DATA_MODES
    auto_modes = AUTO_MODES
    classifier_types = CLASSIFIER_TYPES
    src_dir = None
    models_dir = None
    in_file = None
    pred_files = {}
    out_file = None
    out_file2 = None
    project_name = None
    args = None
    config = None
    initiated = False

    def __new__(cls, *args, **kwargs):
        if not cls.initiated:
            cls.initiated = True
            return super().__new__(cls)
        else:
            pass

    def __init__(self):
        config_files = [f"{self.src_dir}/config/classifiers.ini", f"{self.src_dir}/config/execute.ini",
                        f"{self.src_dir}/config/classifiers.ini", f"{self.src_dir}/config/classifiers.ini"]
        super(ChemInfInput, self).__init__(config_files)
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.dirname(file_dir)
        self.project_name = self.args.name
        self.add_in_file()
        if self.args.mode in self.model_modes or self.args.mode in self.auto_modes:
            self.add_model_path()
        if self.args.mode == 'predict' or self.args.mode == 'auto':
            self.add_predict_path()
        if self.args.mode == 'preproc' or self.args.mode == 'postproc' or self.args.mode == 'auto':
            self.add_out_file()
        if self.args.mode == 'validate':
            if hasattr(self.args, 'out_file2'):
                self.out_file2 = self.args.out_file2
            else:
                self.out_file2 = None

    def add_in_file(self):
        if not os.path.exists(self.args.in_file):
            in_file = f"{self.src_dir}/data/{self.args.in_file}"
            if os.path.exists(in_file):
                self.in_file = in_file
            else:
                raise FileNotFoundError(f"Couldn't find the input file, both absolut path and file name in the data "
                                        f"directory in the ChemInf source directory ({self.src_dir}/data) has been "
                                        f"explored")
        else:
            self.in_file = self.args.in_file

    def add_out_file(self):
        in_file_name, in_file_extension = os.path.split(os.path.basename(self.in_file))
        if self.args.mode == 'preproc':
            if self.args.out_file:
                self.out_file = self.args.out_file
            else:
                if self.args.preproc_mode == 'trim':
                    self.out_file = f"{self.src_dir}/data/{in_file_name}_trimmed.{in_file_extension}"
                elif self.args.preproc_mode == 'resample':
                    self.out_file = f"{self.src_dir}/data/{in_file_name}_balanced.{in_file_extension}"
                elif self.args.preproc_mode == 'split':
                    self.out_file = f"{self.src_dir}/data/{in_file_name}_train.{in_file_extension}"
                    if self.args.out_file2:
                        self.out_file2 = self.args.out_file2
                    else:
                        self.out_file2 = f"{self.src_dir}/data/{in_file_name}_test.{in_file_extension}"

        elif self.args.mode == 'postproc':
            if self.args.out_file:
                self.out_file = self.args.out_file
            else:
                self.out_file = f"{self.src_dir}/data/predictions/{self.project_name}/" \
                                f"{self.project_name}_{self.args.classifier}_summary.csv"

    def add_model_path(self):
        if not self.args.models_dir:
            self.models_dir = f"{self.src_dir}/data/models/{self.project_name}"
        else:
            self.models_dir = self.args.models_dir
        try:
            os.mkdir(self.models_dir)
            print(f"Created the model directory: {self.models_dir}")
        except FileExistsError:
            pass

    def add_predict_path(self):
        def _add_predict_path(classifier_types):
            _pred_files = []
            for i, _classifier in enumerate(classifier_types):
                path = f"{self.src_dir}/data/predictions/{self.project_name}/" \
                       f"{self.project_name}_{_classifier}_predict.csv"
                _pred_files[_classifier] = path
            return _pred_files

        if self.args.classifier == 'all':
            self.pred_files = _add_predict_path(self.classifier_types)
        else:
            if self.args.out_file is None:
                self.pred_files = _add_predict_path([self.args.classifier])
            else:
                self.pred_files[self.args.classifier] = self.args.out_file

            for classifier in self.classifier_types:
                try:
                    pred_dir = os.path.dirname(self.pred_files[classifier])
                    pred_file_name = os.path.basename(self.pred_files[classifier])
                    os.mkdir(pred_dir)
                    print(f"Created the prediction directory for {pred_file_name}: {pred_dir}")
                except FileExistsError:
                    pass

    @classmethod
    def get(cls, key):
        return getattr(cls, key)
