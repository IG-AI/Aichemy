import argparse

import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', required=True,
                        help='Specify the input file.')
    parser.add_argument('-o', '--out_file', required=True,
                        help='Specify the outfile.')
    args = parser.parse_args()

    return args


def reformat_file(input, output):
    with open(input, 'r') as fin:
        reader = fin.read()
        header = "ID\tset\tPredClass0\tPredClass1\tdum0\tClass\tmodel"
        models_results = reader.split(header)
        models_results = [i for i in models_results if i]
        nrow = models_results[0].count("\n") - 1
        array_list = np.empty((nrow, len(models_results), 2), dtype=float)
        sampleID = []
        real_class = []
        for i, model in enumerate(models_results):
            lines = model.split("\n")
            lines = [i for i in lines if i]
            for j, line in enumerate(lines):
                cell = line.split("\t")
                sampleID.append(cell[0])
                real_class.append(cell[5])
                for k in range(2):
                    array_list[j, i, k] = cell[k + 2]

        median_array = np.median(array_list, axis=1).round(10)

    with open(output, 'w+') as fout:
        fout.write("Prediction made from: " + input + '\nsampleID\treal_class\tp(0)\tp(1)\n')
        for i in range(nrow):
            fout.write(f'{sampleID[i]}\t'
                       f'{real_class[i]}\t'
                       f'{median_array[i, 0]}\t'
                       f'{median_array[i, 1]}\n')


if __name__ == '__main__':
    args = argument_parser()

    reformat_file(args.in_file, args.out_file)
