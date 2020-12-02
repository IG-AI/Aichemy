from cheminf.src.cheminf.controller import ALL_MODES
from cheminf.src.cheminf.classifiers import CLASSIFIER_TYPES
from cheminf.src.cheminf.model import ChemInfModel
from tests.test_utils import MockController


def test_model(percentage: float, significance: float,
               index1: int, index2: int, index3: int, index4: int, index5: int, index6: int, index7: int):
    case_indexes = [value for value in locals().values() if isinstance(value, int)]
    for mode in ALL_MODES:
        for classifier_type in CLASSIFIER_TYPES:
            if classifier_type == 'nn':
                controller = MockController(mode, classifier_type, percentage, significance, case_indexes)
                model = ChemInfModel.initialize(controller)
