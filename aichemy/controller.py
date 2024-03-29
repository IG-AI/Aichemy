import configparser
import argparse
import os
from datetime import datetime

from aichemy.classifiers import CLASSIFIER_TYPES
from aichemy.utils import ModeError

MODEL_MODES = ['build', 'improve', 'predict', 'validate']
DATA_MODES = ['postproc', 'preproc']
AUTO_MODES = ['auto']
SUBMODES = ['preproc_mode', 'postproc_mode']
ALL_MODES = MODEL_MODES + DATA_MODES + AUTO_MODES + SUBMODES

OCCASIONAL_FLAGS = ['outfile', 'outfile2', 'classifier', 'models_dir', 'percentage', 'shuffle', 'error_bars', 'significance',
                    'name', 'override_config', 'chunksize', 'nr_cores']
ALL_FLAGS = ['infile', 'name', 'override_config'] + OCCASIONAL_FLAGS

PYTORCH_OPTIMIZERS = ['Adam', 'AdamW', 'Adamax', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta']
TORCHTOOLS_OPTIMIZERS = ['RangerLars']


class AIchemyPref(object):
    def __init__(self, config_file):
        self.args = self.argument_parser()
        for flag in OCCASIONAL_FLAGS:
            if not hasattr(self.args, flag):
                setattr(self.args, flag, None)

        for mode in SUBMODES:
            if not hasattr(self.args, mode):
                setattr(self.args, mode, None)
        self.config = AIchemyConfig(self.args.mode, self.args.classifier, config_file, self.args.override_config)

    @staticmethod
    def argument_parser():
        """Specify what file will be read, what options to use
            and the name of the file to be outputted from the script.
        """

        parser = argparse.ArgumentParser(prog="aichemy")

        parser.add_argument('--version', action='version', version='AIchemy Version 0.2')

        parser_command = parser.add_subparsers(help="Choose a submode to run aichemy in", dest='mode')

        parser_auto = parser_command.add_parser('auto',
                                                help="Automatic submode that preforms a complete data analysis of a "
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

        parser_preproc_mode = parser_preproc.add_subparsers(help="Choose a post processing submode to preform",
                                                            dest='preproc_mode')

        parser_preproc_balancing = parser_preproc_mode.add_parser('balancing',
                                                                  help="Balancing a datasets and saves it")

        parser_preproc_sample = parser_preproc_mode.add_parser('sample',
                                                                 help="Balancing a datasets and saves it")

        parser_preproc_split = parser_preproc_mode.add_parser('split',
                                                              help="Splits a dataset and saves it")

        parser_preproc_trim = parser_preproc_mode.add_parser('trim',
                                                             help="Trims a datasets and saves it")

        parser_postproc = parser_command.add_parser('postproc',
                                                    help="Preforms various post-precessing operations")

        parser_postproc_mode = parser_postproc.add_subparsers(help="Choose a preprocessing submode to preform",
                                                              dest='postproc_mode')

        parser_postproc_structure_check = parser_postproc_mode.add_parser('structure-check',
                                                               help="Check the structure of a project directory")

        parser_postproc_summary = parser_postproc_mode.add_parser('summary',
                                                                  help="Creates a summary from a prediction file")

        parser_postproc_plot = parser_postproc_mode.add_parser('plot',
                                                               help="Creates plots from a summary file")

        all_parsers = [parser_auto, parser_build, parser_improve, parser_predict, parser_validate,
                       parser_preproc_balancing, parser_preproc_sample, parser_preproc_split,
                       parser_preproc_trim, parser_postproc_summary, parser_postproc_plot,
                       parser_postproc_structure_check]

        for subparser in [parser_auto, parser_build, parser_improve, parser_predict, parser_validate,
                          parser_postproc_structure_check]:
            subparser.add_argument('-cl', '--classifier',
                                   default='nn',
                                   choices=['rndfor', 'nn', 'all'],
                                   help="Choose one or all classifiers for model operations")

        for subparser in all_parsers:
            subparser.add_argument('-i', '--infiles',
                                   required=True,
                                   nargs='+',
                                   help="Specify the file containing the data. For postproc plot mode the infile could "
                                        "be combine with multiple infiles to add data to one plot")

        for subparser in [parser_predict, parser_validate, parser_preproc_balancing, parser_preproc_sample,
                          parser_preproc_split, parser_preproc_trim, parser_postproc_summary, parser_postproc_plot]:
            subparser.add_argument('-o', '--outfile',
                                   default=None,
                                   help="Specify the output file with path. If it's not specified for \'predict\' "
                                        "then it will be sat to default (data/predictions)")

        for subparser in [parser_validate, parser_preproc_split]:
            subparser.add_argument('-o2', '--outfile2',
                                   default=None,
                                   help="Specify the second output file.")

        date = datetime.now()
        current_date = date.strftime("%d-%m-%Y")
        for subparser in all_parsers:
            subparser.add_argument('-n', '--name',
                                   default=current_date,
                                   help="Name of the current project, to which all out_puts "
                                        "will use as prefixes or in subdirectories with that name.")

        for subparser in [parser_build, parser_improve, parser_predict, parser_validate]:
            subparser.add_argument('-md', '--models_dir',
                                   default=None,
                                   help="Specify the path to the directory "
                                        "where the generated models should be stored for 'build' "
                                        "or 'predict'). Otherwise it will be the default model directory "
                                        "(data/models).")

        for subparser in [parser_preproc_split, parser_preproc_trim, parser_preproc_balancing, parser_preproc_sample]:
            subparser.add_argument('-pc', '--percentage',
                                   type=float,
                                   default=0.5,
                                   help='Specify the split percentage.')

        for subparser in [parser_preproc_split, parser_preproc_trim]:
            subparser.add_argument('-sh', '--shuffle',
                                   default=False,
                                   action='store_true',
                                   help="Specify that the data should be shuffled")

        for subparser in [parser_postproc_plot]:
            subparser.add_argument('-eb', '--error_bars',
                                   default=False,
                                   action='store_true',
                                   help="Specify if the combine plots should be produced with multi-graphs or with "
                                        "error bars")

        for subparser in [parser_postproc_summary]:
            subparser.add_argument('-sg', '--significance',
                                   default=None,
                                   type=float,
                                   help="If you only want results for a single "
                                        "significance level, give a value between "
                                        "0 and 1.")

        for subparser in all_parsers:
            subparser.add_argument('-cf', '--override_config',
                                   default=None,
                                   help="Specify a parameter in the configuration that will to be override ")

        for subparser in [parser_auto, parser_preproc_balancing, parser_preproc_sample]:
            subparser.add_argument('-ch', '--chunksize',
                                   default=None,
                                   type=int,
                                   help="Specify the size of chunks the files should be divided into.")

        for subparser in [parser_auto, parser_preproc_balancing, parser_preproc_sample, parser_preproc_split,
                          parser_preproc_trim]:
            subparser.add_argument('-nc', '--nr_cores',
                                   default=1,
                                   type=int,
                                   help="Specify the amount of cores should be used.")

        args = parser.parse_args()

        if hasattr(args, 'preproc_mode'):
            if (args.preproc_mode == 'balancing' or args.preproc_mode == 'sample')\
                    and (args.nr_cores and not args.chunksize):
                parser.error("Multicore has to be executed with chunking in preproc mode.")

            if (args.preproc_mode == 'split' or args.preproc_mode == 'trim') \
                    and (args.nr_cores or args.chunksize):
                parser.error(f"Preproc {args.preproc_mode} be can't executed in multicore submode or with chunking.")

        return args


class AIchemyConfig(object):
    def __init__(self, operator_mode, classifier_type, config_files, overrider):
        self.clf_conf_file = config_files[0]
        self.exec_conf_file = config_files[1]
        self.execute = ConfigExec(operator_mode, self.exec_conf_file)

        if operator_mode in AUTO_MODES or operator_mode in MODEL_MODES:
            self.classifier = ConfigClf(classifier_type, self.clf_conf_file)
        else:
            self.classifier = None

        if overrider is not None:
            self.update_config(overrider)

    def update_config(self, overrider):
        new_configs = overrider.split(',')
        for config in new_configs:
            try:
                key, value = config.split(':')
            except ValueError:
                return

            try:
                attr_pos = 'execute'
                old_value = getattr(self.execute, key)
            except AttributeError:
                try:
                    attr_pos = 'classifier'
                    old_value = getattr(self.classifier, key)
                except AttributeError:
                    print(f"Config {key} can't be changed because current run doesn't use that configuration")
                    return 

            value_type = old_value.__class__.__name__

            if f"{value_type}" == 'bool':
                value = boolean(value)
            elif f"{value_type}" == 'list':
                value = config_to_list(value)
            else:
                if value_type != 'str':
                    try:
                        value = eval(f"{value_type}({value})")
                    except ValueError:
                        error_massage = f"The override configuration value ({value}) for the configuration {config} " \
                                        f"doesn't have equal type as the default value ({old_value})"
                        raise TypeError(error_massage)

            obj = eval(f"self.{attr_pos}")
            print(f"Changed configuration for '{key}' from '{old_value}' to '{value}'")
            setattr(obj, key, value)


class ConfigClf(object):
    def __init__(self, classifier_type, config_file):
        config = configparser.SafeConfigParser()
        config.read(config_file)

        self.nr_models = int(config['all']['nr_models'])

        if classifier_type == 'rndfor' or classifier_type == 'all':
            self.prop_train_ratio = float(config['random_forest']['prop_train_ratio'])
            self.nr_trees = int(config['random_forest']['nr_of_trees'])
            self.n_jobs = int(config['random_forest']['n_jobs'])
            self.pred_nrow = float(config['random_forest']['pred_nrow']),
            self.val_folds = int(config['random_forest']['val_folds']),
            self.smooth = boolean(config['random_forest']['smooth']),
            self.data_type = str(config['random_forest']['data_type'])

        if classifier_type == 'nn' or classifier_type == 'all':
            self.val_ratio = float(config['neural_network']['val_ratio'])
            self.cal_ratio = float(config['neural_network']['cal_ratio'])
            self.dim_in = int(config['neural_network']['dim_in'])
            self.dim_hidden = config_to_list(config['neural_network']['dim_hidden'])
            self.dim_out = int(config['neural_network']['dim_out'])
            self.dropout = float(config['neural_network']['dropout'])
            self.batch_size = int(config['neural_network']['batch_size'])
            self.max_epochs = int(config['neural_network']['max_epochs'])
            self.early_stop_patience = int(config['neural_network']['early_stop_patience'])
            self.early_stop_threshold = float(config['neural_network']['early_stop_threshold'])
            self.optimizer = config['neural_network']['optimizer']
            self.optimizer_learn_rate = float(config['neural_network']['optimizer_learn_rate'])
            self.optimizer_weight_decay = float(config['neural_network']['optimizer_weight_decay'])

            try:
                self.pred_sig = int(config['neural_network']['pred_sig'])
            except ValueError:
                self.pred_sig = None


class ConfigExec(object):
    def __init__(self, operator_mode, config_file):
        config = configparser.SafeConfigParser()
        config.read(config_file)

        if operator_mode == 'auto':
            self.auto_save_preproc = boolean(config['auto']['auto_save_preproc'])
            self.auto_plus_balancing = boolean(config['auto']['auto_plus_balancing'])
            self.auto_plus_sample = boolean(config['auto']['auto_plus_sample'])
            self.auto_plus_sum = boolean(config['auto']['auto_plus_sum'])
            self.auto_plus_plot = boolean(config['auto']['auto_plus_plot'])
            self.train_test_ratio = float(config['auto']['train_test_ratio'])

        if operator_mode == 'postproc' or operator_mode == 'auto':
            self.error_level = int(config['postproc']['error_level'])
            self.plot_del_sum = boolean(config['postproc']['plot_del_sum'])

        if operator_mode == 'preproc' or operator_mode == 'auto':
            self.sample_ratio = float(config['preproc']['sample_ratio'])
            self.balancing_ratio = float(config['preproc']['balancing_ratio'])


class AIchemyController(AIchemyPref):
    model_modes = MODEL_MODES
    data_modes = DATA_MODES
    auto_modes = AUTO_MODES
    classifier_types = CLASSIFIER_TYPES
    scr = None
    project_dir = None
    predictions_dir = None
    args = None
    config = None
    session = None

    def __new__(cls, *args, **kwargs):
        if cls.session is None:
            cls.session = super().__new__(cls)

        return cls.session

    def __init__(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(file_dir)
        self.src_dir = os.path.dirname(package_dir)

        config_files = [f"{self.src_dir}/config/classifiers.ini", f"{self.src_dir}/config/execute.ini"]
        super(AIchemyController, self).__init__(config_files)

        self.update_infiles()
        self.add_project_dir()

        if self.args.mode in self.model_modes or self.args.mode in self.auto_modes:
            self.add_model_path()

        if self.args.mode in self.model_modes or self.args.mode == 'auto':
            self.add_predict_path()

        if self.args.mode == 'preproc' or self.args.mode == 'postproc' or self.args.mode == 'auto':
            self.update_outfile()

    def update_infiles(self):
        nr_infiles = len(self.args.infiles)
        if nr_infiles == 1:
            infile = self.args.infiles[0]
            if not os.path.exists(infile):
                if self.args.mode == 'postproc':
                    infile = f"{self.src_dir}/data/{self.args.name}/predictions/{infile}"
                else:
                    infile = f"{self.src_dir}/data/{infile}"

            if os.path.isfile(infile):
                self.args.infile = infile
            else:
                raise FileNotFoundError(f"Couldn't find the input file, both absolut path and file name in the "
                                        f"data directory in the AIchemy source directory ({self.src_dir}/data) has "
                                        f"been explored")

        elif self.args.postproc_mode == 'plot':
            if all([not os.path.exists(infile) for infile in self.args.infiles]):
                raise ValueError("Multiple inputs needs to be provided with absolute or relative paths.")

            elif any([not os.path.exists(infile) for infile in self.args.infiles]):
                none_existing = []
                for infile in self.args.infiles:
                    if not os.path.exists(infile):
                        none_existing.append(infile)
                if len(none_existing) > 1:
                    plural_string = "s don't"
                else:
                    plural_string = " doesn't"
                error_massage = f"The following file{plural_string} exist: {' '.join(none_existing)}"
                raise ValueError(error_massage)

        else:
            if self.args.pre_mode:
                submode = f" with submode {self.args.pre_mode}"
            elif self.args.post_mode:
                submode = f" with submode {self.args.post_mode}"
            else:
                submode = ""
            error_massage = f"Mode {self.args.mode}{submode} doesn't support multiple input files"
            raise ValueError(error_massage)

    def update_outfile(self):
        if self.args.outfile is None or self.args.outfile2 is None:
            infile_name, infile_extension = os.path.splitext(os.path.basename(self.args.infiles[0]))
            if hasattr(self.args, 'infile'):
                if self.args.mode == 'preproc':
                    if self.args.preproc_mode == 'trim':
                        self.args.outfile = f"{self.project_dir}/{infile_name}_trimmed{infile_extension}"

                    elif self.args.preproc_mode == 'balancing':
                        self.args.outfile = f"{self.project_dir}/{infile_name}_balanced{infile_extension}"

                    elif self.args.preproc_mode == 'sample':
                        self.args.outfile = f"{self.project_dir}/{infile_name}_sampled{infile_extension}"

                    elif self.args.preproc_mode == 'split':
                        if self.args.outfile is None:
                            self.args.outfile = f"{self.project_dir}/{infile_name}_train{infile_extension}"
                        if self.args.outfile2 is None:
                            self.args.outfile2 = f"{self.project_dir}/{infile_name}_test{infile_extension}"
                    else:
                        ModeError(self.args.mode, self.args.preproc_mode)

                elif self.args.postproc_mode == 'summary':
                    self.args.outfile = f"{self.project_dir}/predictions/{infile_name}_summary{infile_extension}"

            else:
                pass

    def add_project_dir(self):

        project_path, project_name = os.path.split(self.args.name)
        if not project_path:
            self.project_dir = f"{self.src_dir}/data/{project_name}"
        else:
            self.project_dir = f"{self.src_dir}/data/{self.args.name}"
        try:
            os.makedirs(self.project_dir)
        except OSError:
            pass
        else:
            print(f"Created the project directory: {self.project_dir}")
        finally:
            self.args.name = project_name

    def add_model_path(self):
        if not self.args.models_dir:
            self.args.models_dir = f"{self.project_dir}/models"

        try:
            os.mkdir(self.args.models_dir)
            print(f"Created the model directory: {self.args.models_dir}")
        except FileExistsError:
            pass

    def add_predict_path(self):
        def _add_predict_path(classifier_types):
            _pred_files = {}
            if self.args.outfile is None:
                for i, _classifier in enumerate(classifier_types):
                    path = f"{self.predictions_dir}/{self.args.name}_{_classifier}_predictions.csv"
                    _pred_files[_classifier] = path
            else:
                outfile_path, outfile = os.path.split(self.args.outfile)
                outfile_name, outfile_extension = os.path.splitext(outfile)
                for i, _classifier in enumerate(classifier_types):
                    path = f"{outfile_path}{outfile_name}_{_classifier}{outfile_extension}"
                    _pred_files[_classifier] = path
                    self.update_outfile()
            return _pred_files

        self.predictions_dir = f"{self.project_dir}/predictions"
        try:
            os.mkdir(self.predictions_dir)
            print(f"Created the model directory: {self.predictions_dir}")
        except FileExistsError:
            pass

        if self.args.classifier == 'all':
            if not self.args.mode == 'build':
                self.args.pred_files = _add_predict_path(self.classifier_types)
            else:
                self.args.pred_files = None
        else:
            self.args.pred_files = _add_predict_path([self.args.classifier])
            for classifier in self.classifier_types:
                try:
                    pred_dir = os.path.dirname(self.args.pred_files[classifier])
                    pred_file_name = os.path.basename(self.args.pred_files[classifier])
                    os.mkdir(pred_dir)
                    print(f"Created the prediction directory for {pred_file_name}: {pred_dir}")
                except (KeyError, FileExistsError) as e:
                    pass

    @classmethod
    def get(cls, key):
        return getattr(cls, key)


def config_to_list(config):
    return [int(x) for x in config.split("|")]


def boolean(_bool):
    return f'{_bool}'.lower() in ("yes", "true", "t", "1")
