__name__ = 'cheminf'
__version__ = 0.2
__author__ = "Daniel Ågtrand, Leonard Sparring"
__email__ = "d.e.agstrand@gmail.com, leo.sparring@gmail.com"
__date__ = "30/10-2020"
__credits__ = "This frameworks random forest classifier is based on the cheminf_loop2_20.py aggregated mondrian " \
              "conformal predictor written by Ulf Norinder. The neural network classifier is based on " \
              "01_DNN_CP_tr_te_20.py written by Ulf Norinder and Jin Zhang. The external libraries nonconformist is " \
              "written by Henrik Linusson that and torchtools is written by Pablo Pernías"
"""
22/02-2020, AJL: 
    Added multiclass protocol.
24/03-2020, AJL: 
    Added (de)compressed data handling.
08/04-2020, LS:  
    Revision of decompression and modularity implementation.
06/07-2020, LS: 
    Implementation of k-fold cross validation.
28/09-2020 - 30/10-2020, DA: 
    Completely revamped the code with a object originated approached, extended the available classifiers with a 
    dynamic neural network, added data utils, completely changed to how the user interact with the framework to make it
    more user-friendly and added a setup script for easier installation of the framework.    
"""
import source.cheminf_main

if __name__ == "__main__":
    source.cheminf_main.main()
