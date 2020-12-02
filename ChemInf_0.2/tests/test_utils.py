import os
import random
from typing import Union, List

from cheminf.src.cheminf.controller import ChemInfController, DATA_MODES, ALL_MODES, MODEL_MODES, AUTO_MODES, \
    PREPROC_SUBMODES, POSTPROC_SUBMODES
from src.cheminf.utils import rmdir

TEST_CASES = 8
NR_RND_FLAGS = 7


class MockController(ChemInfController):
    def __init__(self, mode: str, classifier_type: str, percentage: Union[float, None], significance: Union[float, None]
                 , *randoms: List[int]):
        self.name = f"test-{mode}-{random.randint(1,1000)}"
        self.mode = mode
        self.classifier_type = classifier_type
        self.args = self.make_mock_argv('easy', percentage, significance, *randoms)
        super(MockController, self).__init__()

    def make_mock_argv(self, difficulty, percentage: Union[float, None], significance: Union[float, None],
                       *randoms: List[int]):
        random.seed(random.random())
        if difficulty == 'easy':
            argv = self._get_easy_argv(percentage, significance, randoms)
        elif difficulty == 'hard':
            argv = self._get_hard_argv(percentage, significance, randoms)
        else:
            raise ValueError("Wrong kind of difficulty level")
        return argv

    def create_namespace(self, flags):
        if self.mode in PREPROC_SUBMODES:
            arg_list = [f"preproc", f"{self.mode}"]
        elif self.mode in POSTPROC_SUBMODES:
            arg_list = [f"postproc", f"{self.mode}"]
        else:
            arg_list = [f"{self.mode}"]

        for key, value in flags.items():
            arg_list.append(f"--{str(key)}")
            arg_list.append(value)

        return arg_list

    def _get_hard_argv(self, percentage, significance, *randoms):
        flags = {f"{key}": value for key, value in locals().items() if key not in ['randoms', 'self']}
        data_dir = "cheminf/data"
        flags["name"] = self.name
        flags["classifier"] = self.classifier_type
        os.mkdir(f"cheminf/data/{self.name}")

        if self.mode in DATA_MODES or self.mode == 'predict' or self.mode == 'validate':
            if (randoms[0] % 2) == 1:
                flags["outfile"] = f"{data_dir}/{self.mode}/outfile.csv"

        if self.mode == 'split' or self.mode == 'validate':
            flags["outfile2"] = f"{data_dir}/{self.mode}/outfile2.csv"

        if self.mode in ALL_MODES:
            flags["infiles"] = [f"{data_dir}/d2_trailset_balanced.csv",
                                  "d2_trailset_balanced.csv",
                                  f"{data_dir}/test_features.csv",
                                  "test_features.csv",
                                  f"{data_dir}/test_features.csv {data_dir}/test_features2.csv",
                                  f"test_features.csv test_features2.csv",
                                  f"{data_dir}/test_features.csv test_features2.csv",
                                  ""][randoms[1]]

            flags["override_config"] = [f"val_ratio:0.12",
                                          f"optimizer:Adam",
                                          f"dim_hidden:1000|4000|2000|2000",
                                          f"auto_plus_sum:False",
                                          f"val_ratio:0.12,cal_ratio:2.0",
                                          f"val_ratio:0.12, cal_ratio:2.0",
                                          f"cal_ratio:2.0",
                                          f""][randoms[2]]

        if self.mode in MODEL_MODES or self.mode in AUTO_MODES:
            if (randoms[3] % 2) == 1:
                flags["models_dir"] = f"{data_dir}/{self.mode}/models"

        if self.mode == 'split' or self.mode == 'trim':
            flags["shuffle"] = [[True, False][i % 2] for i in range(randoms[4])]

        if self.mode in AUTO_MODES or self.mode == 'balancing' or self.mode == 'sample':
            flags["chunksize"] = [0, 1, 1000, 10000, 1000000, 1000000000000000, -10000, None][randoms[5]]
            flags["nr_cores"] = range(8)[randoms[6]]

        if self.mode not in DATA_MODES:
            flags.pop("percentage")

        if self.mode != "summary":
            flags.pop("significance")

        return self.create_namespace(flags)

    def _get_easy_argv(self, percentage, significance, *randoms):
        flags = {f"{key}": value for key, value in locals().items() if key not in ['randoms', 'self']}
        data_dir = "cheminf/data"
        flags["name"] = self.name
        flags["classifier"] = self.classifier_type
        os.mkdir(f"cheminf/data/{self.name}")

        if self.mode in PREPROC_SUBMODES or self.mode in POSTPROC_SUBMODES or self.mode == ['predict', 'validate']:
            flags["outfile"] = random.choice([f"{data_dir}/{self.mode}/outfile.csv", None])

        if self.mode == 'split' or self.mode == 'validate':
            flags["outfile2"] = random.choice([f"{data_dir}/{self.mode}/outfile2.csv", None])

        if self.mode in ALL_MODES:
            flags["infiles"] = f"test_features.csv"

            flags["override_config"] = random.choice([f"val_ratio:0.12", None])

        if self.mode in MODEL_MODES or self.mode in AUTO_MODES:
            flags["models_dir"] = random.choice([f"{data_dir}/{self.name}/models", None])

        if self.mode in AUTO_MODES or self.mode == 'balancing' or self.mode == 'sample':
            flags["chunksize"] = random.choice([1000, None])
            flags["nr_cores"] = random.choice([1, 2])

        if self.mode not in DATA_MODES:
            flags.pop("percentage")

        if self.mode != "summary":
            flags.pop("significance")

        return self.create_namespace(flags)

    def __del__(self):
        rmdir(f"cheminf/data/{self.name}")
