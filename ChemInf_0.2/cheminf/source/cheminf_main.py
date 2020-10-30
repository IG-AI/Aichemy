__author__ = "Leonard Sparring, Daniel Ågtrand"
__date__ = "2019-11-22"
__email__ = "leo.sparring@gmail.com, d.e.agstrand@gmail.com"
__version__ = "0.2"

"""This script is based on the cheminf_loop2_20.py aggregated
mondrian conformal predictor written by Ulf Norinder.

# 22-02-2020 AJL added multiclass protocol.
# 24-03-2020 AJL added (de)compressed data handling.
# 08-04-2020 LS revision of decompression and modularity implementation.
# 06-07-2020 LS implementation of k-fold cross validation.
"""
from source.cheminf_lib import Timer, ChemInfOperator


@Timer
def main():
    """Run the main program."""
    cheminf = ChemInfOperator()

    if cheminf.mode == 'validate':
        cheminf.model.validate()
    elif cheminf.mode == 'build':
        cheminf.model.build()
    elif cheminf.mode == 'predict':
        cheminf.model.predict()
    elif cheminf.mode == 'improve':
        cheminf.model.improve()
    elif cheminf.mode == 'utils':
        cheminf.utils.run()


if __name__ == '__main__':
    main()
