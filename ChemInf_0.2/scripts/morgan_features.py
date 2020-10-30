
__author__  = 'Leonard Sparring'
__date__    = '2019-12-21'
__email__   = 'leo.sparring@gmail.com'
__version__ = '0.1'

"""This script will generate morgan fingerprints based on a .smi file.
It expects the first field to be a SMILE-pattern and the following field
to be an identifier. The outputfile is formatted with the ID as the
first field and the fingerprint bits as the following fields, tab-sep.
"""

import argparse
from timeit import default_timer as timer

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem


###--- FUNCTIONS ---###

def argument_parser():
    """Specify what file will be read, what options to use
    and the name of the file to be outputted from the script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file',
            help='Specify the input file.',
            required=True)
    parser.add_argument('-o', '--out_file',
            help='Specify the output file.',
            required=True)
    parser.add_argument('-r', '--radius',
            help='Specify the fingerprint radius.',
            type=int, required=True)
    args = parser.parse_args()

    return args


def read_and_write(args):
    """This function opens both the in file and the out file and
    calculates fingerprints line by line to preserve memory.
    """
    counter = 0
    error_index = []
    with open(args.in_file, 'r') as fin, open(args.out_file, 'w+') as fout:
        for line in fin:
            line_list = line.strip().split(None,1)
            counter += 1
            try:
                fingerprint = morgan_from_smi(line_list[0], args.radius)
                arr = np.zeros((1,), dtype=str)
                DataStructs.ConvertToNumpyArray(fingerprint, arr)
                # Writing out the sample ID.
                fout.write(f'{line_list[1]} ')
                fout.write(' '.join(arr))
                fout.write('\n')
            except Exception as msg:
                print(msg)
                error_index.append(counter)

    print(f'Could not get features for lines: {error_index}')


def morgan_from_smi(smi, radius):
    """The function calculates the morgan fingerprint for one
    molecule based on its SMILE-pattern.
    """
    molecule = Chem.MolFromSmiles(smi)
    fingerprint = Chem.GetMorganFingerprintAsBitVect(molecule,
                                         radius, nBits = 1024)

    return fingerprint


def main():
    """Run the main program."""

    script_start_time = timer()

    args = argument_parser()

    read_and_write(args)

    print(f'Script ran for {(timer() - script_start_time):.2f} s.')

if __name__ == '__main__':
    main()
