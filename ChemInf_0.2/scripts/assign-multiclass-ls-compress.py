__author__  = 'Leonard Sparring'
__date__    = '2019-11-12'
__email__   = 'leo.sparring@gmail.com'


"""This script was developed for the rdkit chemical predictor
->cheminf pipeline based on Ulf Norinders conformal prediction
code. It takes in a file that is output from the rdkit_pc_smi.py
and spits out a file that is compatible with the amcp_loop2_20.py
script.

# 23-02-2020 AJL added multiclass protocol.
"""

import argparse
import os

def argument_parser():
    """Specify what file will be read,
    and the name of the file to be outputted from the script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file',
            help='Specify the input file (.smi).')
    parser.add_argument('-o', '--out_file',
            help='Specify the output file.')
    parser.add_argument('-c', '--cutoffs',
            nargs='+', type=float,
            help='Specify cutoff values for classification.')
    args = parser.parse_args()

    return args


def read_morgan(in_file, out_file, cutoffs):
    """Takes in a file that is outputted from 
    script.
    """

    # Sort the cutoff values
    cutoffs.sort()

    # Retrieve the number of classes
    nr_class = len(cutoffs)

    # compressed data handling based on file extensions
    in_name, in_extension = os.path.splitext(in_file)
    out_name, out_extension = os.path.splitext(out_file)

    if in_extension == ".bz2":
        import bz2
        fin = bz2.open(in_file, 'rb')
    elif in_extension == ".gz":
        import gzip
        fin = gzip.open(in_file, 'rb')
    else:
        fin = open(in_file, 'r')
    
    if out_extension == ".bz2":
        import bz2
        fout = bz2.open(out_file, 'wb')
    elif out_extension == ".gz":
        import gzip
        fout = gzip.open(out_file, 'wb')
    else:
        fout = open(out_file, 'w+')

    if (in_extension == ".bz2" or in_extension == ".gz") and\
       (out_extension == ".bz2" or out_extension == ".gz"):
        for line in fin:
            line = line.decode()
            line_list = line.strip().split(None,1)
            score = float(line_list[0].split('_')[1])
            comparison = cutoffs.copy()
            comparison.append(score)
            comparison.sort(reverse=True)
            index = comparison.index(score)
            fout.write(line_list[0].encode())
            fout.write(f'\t{index}\t'.encode())
            fout.write(f'{line_list[1]}\n'.encode())

    elif (in_extension != ".bz2" and in_extension != ".gz") and\
         (out_extension == ".bz2" or out_extension == ".gz"):
        for line in fin:
            line_list = line.strip().split(None,1)
            score = float(line_list[0].split('_')[1])
            comparison = cutoffs.copy()
            comparison.append(score)
            comparison.sort(reverse=True)
            index = comparison.index(score)
            fout.write(line_list[0].encode())
            fout.write(f'\t{index}\t'.encode())
            fout.write(f'{line_list[1]}\n'.encode())

    elif (in_extension != ".bz2" and in_extension != ".gz") and\
         (out_extension != ".bz2" and out_extension != ".gz"):
        for line in fin:
            line_list = line.strip().split(None,1)
            score = float(line_list[0].split('_')[1])
            comparison = cutoffs.copy()
            comparison.append(score)
            comparison.sort(reverse=True)
            index = comparison.index(score)
            fout.write(line_list[0])
            fout.write(f'\t{index}\t')
            fout.write(f'{line_list[1]}\n')

    elif (in_extension == ".bz2" or in_extension == ".gz") and\
       (out_extension != ".bz2" and out_extension != ".gz"):
        for line in fin:
            line = line.decode()
            line_list = line.strip().split(None,1)
            score = float(line_list[0].split('_')[1])
            comparison = cutoffs.copy()
            comparison.append(score)
            comparison.sort(reverse=True)
            index = comparison.index(score)
            fout.write(line_list[0])
            fout.write(f'\t{index}\t')
            fout.write(f'{line_list[1]}\n')

    fin.close()
    fout.close()


def main():
    """Parsing arguments, reads a file, writes another file.
    """

    args = argument_parser()

    read_morgan(args.in_file, args.out_file, args.cutoffs)


if __name__ == '__main__':
    main()
