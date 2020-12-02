__version__ = 0.1
__author__ = "Daniel Ågtrand"
__email__ = "d.e.agstrand@gmail.com"
__url__ = "https://github.com/leosparring/ChemInf/tree/master/ChemInf_0.2"
__platform__ = 'Linux'
__date__ = "27/11-2020"

import random

from tests.integrations.all_modules import  test_all
from tests.test_utils import TEST_CASES, NR_RND_FLAGS


if __name__ == '__main__':
    for _ in range(1000):
        case_indexes = [random.randrange(1, TEST_CASES) for _ in range(NR_RND_FLAGS)]
        random.seed(random.random())
        percentage = random.random()
        significance = random.random()
        test_all(percentage, significance, *case_indexes)

