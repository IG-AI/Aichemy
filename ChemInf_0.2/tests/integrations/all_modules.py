import time

from cheminf.src.cheminf.controller import ALL_MODES
from cheminf.src.cheminf.operator import run_operator
from cheminf.src.cheminf.classifiers import CLASSIFIER_TYPES
from tests.test_utils import MockController


def test_all(percentage: float, significance: float,
             index1: int, index2: int, index3: int, index4: int, index5: int, index6: int, index7: int):
    case_indexes = [value for value in locals().values() if isinstance(value, int)]
    for mode in ALL_MODES:
        for classifier_type in CLASSIFIER_TYPES:
            if classifier_type == 'nn':
                try:
                    controller = MockController(mode, classifier_type, percentage, significance, case_indexes)
                    run_operator(controller)
                except Exception as e:
                    print(f"Exception: {e.__class__.__name__}\n{e.__cause__}")
                    time.sleep(1)


"""
flags["--models_dir"] = [f"{data_dir}/models",
                         f"{data_dir}/models/",
                         f"{data_dir}/models/test/test",
                         f"models",
                         f"models/",
                         f"models.csv",
                         f"."
                         f""][randoms[1]]
outfile = []
for i in range(1, 2):
    if i == 1:
        i = ""
                 
    exec(f"outfile{i} = "
         f"[{data_dir}/test/outfile{i}.csv, "
         f"outfile{i}.csv, "
         f"{None}, "
         f"../{data_dir}/test/outfile{i}.csv, "
         f"outfileÅÄÖ{i}.csv, ,"
         f"outfile!$€{i}.csv, "
         f"test/outfile{i}.csv/][{randoms[2]}]")

if self.mode in DATA_MODES or self.mode == 'predict' or self.mode == 'validate':
    flags["--outfile"] = outfile[0]

if self.mode == 'split' or self.mode == 'validate':
    flags["--outfile2"] = outfile[1]
"""