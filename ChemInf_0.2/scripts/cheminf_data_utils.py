import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.utils import resample

global mp


class AMCPBase:
    def __init__(self, mode):
        self.args = self._argument_parser()
        self.mode = mode

    @staticmethod
    def _argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_file', required=True,
                            help='Specify the input file.')
        parser.add_argument('-o', '--out_file', required=True,
                            help='Specify the outfile.')
        parser.add_argument('-o2', '--out_file2',
                            help='Specify the second outfile.')
        parser.add_argument('-p', '--percentage',
                            help='Specify the split percentage.',
                            type=float,
                            default=0.5)
        parser.add_argument('-c', '--chunksize',
                            type=int,
                            help='Specify size of chunks the files should be divided into.')
        parser.add_argument('-s', '--shuffle',
                            default=False,
                            action='store_true',
                            help='Specify that the data should be shuffled')
        parser.add_argument('-nc', '--num_core',
                            default=1,
                            type=int,
                            help='Specify the amount of cores should be used.')
        parser.add_argument('-m', '--mode', required=True,
                            choices=['split', 'trim', "resample"],
                            help='Specify what operation that should be preformed on the data.\n'
                                 'If the data should be:\n'
                                 '1. Splitted (-m split)\n'
                                 '2. Trim (-m trim)\n'
                                 '3. Resampled (-m resampled)\n')

        args = parser.parse_args()

        return args


class Timer(AMCPBase):
    def __init__(self, func):
        super().__init__("timer")
        self.func = func

    def __call__(self, *args):
        begin = time.time()

        self.func(*args)

        end = time.time()
        run_time = np.round(end - begin, decimals=2)
        print("\nTIME PROFILE\n-----------------------------------")
        print("Runtime for {func} was {time}s".format(func=self.func.__name__, time=run_time))
        if self.func.__name__ == "multicore":
            print("Runtime per core(s) with {func} was {time}s"
                  .format(func=self.func.__name__, time=(run_time * self.args.num_core)))


class AMCPUtils(AMCPBase):
    def __init__(self):
        super().__init__("utils")

    def run(self):
        def single_thread(args):
            if self.args.mode == "resample":
                dataframe = resample_file(args.in_file, args.chunksize)
                save_dataframe(dataframe, args.out_file)

            else:
                if args.mode == "split":
                    dataframe_large, dataframe_small = cut_file(args.in_file, args.percentage, args.shuffle, split=True)
                    save_dataframe(dataframe_large, args.out_file)
                    save_dataframe(dataframe_small, args.out_file2)

                elif args.mode == "trim":
                    dataframe = cut_file(args.in_file, args.percentage, args.shuffle)
                    save_dataframe(dataframe, args.out_file)

        def multicore(args):
            if args.mode not in ['split', 'trim']:
                mp = __import__("multiprocessing", globals(), locals())
                print("Available cores: {cpu}".format(cpu=mp.cpu_count()))
                ctx = mp.get_context('spawn')
                with ctx.Pool(args.num_core) as pool:
                    dataframe = pd.DataFrame()
                    if args.mode == "resample":
                        chunks = read_dataframe(args.in_file, chunksize=args.chunksize)
                        print("Starting multicore resampling with {cores} cores".format(cores=args.num_core))
                        pool_stack = pool.imap_unordered(resample_dataframe, chunks)
                        for i, pool_dataframe in enumerate(pool_stack):
                            print("\nWorking on chunk {iteration}"
                                  "\n-----------------------------------".format(iteration=i))
                            dataframe = pd.concat([dataframe, pool_dataframe])


                    pool.close()
                    pool.join()

                    dataframe = shuffle_dataframe(dataframe)
                    save_dataframe(dataframe, args.out_file)

            else:
                raise ValueError("{mode} doesn't have multicore support".format(mode=self.args.mode))

        if self.args.num_core > 1:
            if self.args.mode in ['split', 'trim']:
                raise ValueError("{mode} don't have multicore support".format(mode=self.args.mode))
            multicore(self.args)
        else:
            single_thread(self.args)


def resample_file(file, chunksize):
    chunks = read_dataframe(file, chunksize=chunksize)
    dataframe = pd.DataFrame()

    print("\nStart creating resampled dataframe from {file} in chunks with size {chunksize} rows"
          .format(file=file, chunksize=chunksize))

    for i, chunk in enumerate(chunks):
        print("\nWorking on chunk {iteration}"
              "\n-----------------------------------".format(iteration=i))
        data = resample_dataframe(chunk)
        dataframe = pd.concat([dataframe, data])

    print("\nShuffles the dataset")
    return shuffle_dataframe(dataframe)


def resample_dataframe(dataframe):
    dataframe_class0 = dataframe[dataframe['class'] == 0]
    dataframe_class1 = dataframe[dataframe['class'] == 1]

    data_division = dataframe['class'].value_counts()

    dataframe_class0_resample = resample(dataframe_class0,
                                         replace=False,
                                         n_samples=data_division[1],
                                         random_state=123)

    dataframe_resample = pd.concat([dataframe_class0_resample, dataframe_class1])

    return dataframe_resample


def shuffle_dataframe(dataframe):
    return dataframe.sample(frac=1, random_state=123, axis=0)


def cut_file(file, percentage, shuffle, split=False):
    dataframe = read_dataframe(file, shuffle=shuffle)

    index = int(np.round(len(dataframe) * percentage))
    dataframe1 = dataframe.iloc[:index, :]
    if split:
        dataframe2 = dataframe.iloc[index:, :]
        return dataframe1, dataframe2
    else:
        return dataframe1


def read_dataframe(file, chunksize=None, shuffle=False):
    print("\nReading from {file}".format(file=file))

    with open(file) as f:
        first_line = next(f)
        line_data = re.split(" |\t|[|]", first_line)
        num_cols = len(line_data)

    unnamed_columns = [str(i) for i in range(1, num_cols - 1)]
    columns_names = ['id', 'class'] + unnamed_columns

    columns_types = {'id': 'string', 'class': 'int8'}
    for i in range(1, (num_cols - 1)):
        columns_types[str(i)] = 'int8'

    if chunksize:
        dataframe = pd.read_csv(file,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                chunksize=chunksize,
                                iterator=True,
                                engine='c')

    else:
        dataframe = pd.read_csv(file,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                engine='c')

        if shuffle:
            dataframe = shuffle_dataframe(dataframe)

    return dataframe


def save_dataframe(dataframe, out_file):
    print("\nSave dataframe as csv to {out_file}".format(out_file=out_file))

    dataframe.to_csv(out_file,
                     index=False,
                     header=False,
                     sep='\t')


def read_parameters(in_file, params_dict):
    """ Reads a file with hyperparameters for the
    classifier and parameters for the CP script. Updates the params_dict.
    """

    with open(in_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            assert len(items) == 2, \
                'The parameter file should have one key value pair per line'
            params_dict[items[0]] = items[1]


def read_array(in_file, data_type):
    """Reads a white space separated file without a header.
    The first column contains the IDs of samples and the 2nd the class.
    All the following columns contain features (the independent variables)
    of the sample. The function first determines the size of the arrays to
    be created, then inserts data into them.
    """

    print(f'Reading from \'{in_file}\'.')

    # read (compressed) features
    file_name, extension = os.path.splitext(in_file)
    if extension == ".bz2":
        import bz2
        f = bz2.open(in_file, 'rb')
    elif extension == ".gz":
        import gzip
        f = gzip.open(in_file, 'rb')
    else:
        f = open(in_file, 'r')

    nrow = sum(1 for line in f)
    f.seek(0)
    ncol = len(f.readline().split()) - 1  # Disregarding ID column.
    f.seek(0)
    # Initializing arrays.
    print(f'Initializing array of size {nrow} X {ncol}.')
    IDs = np.empty(nrow, dtype=np.dtype('U50'))
    print(f'Memory of ID_array(str): {(IDs.nbytes * 10 ** (-6))} MB.')
    if data_type == 'integer':
        data = np.empty((nrow, ncol), dtype=int)
    if data_type == 'float':
        data = np.empty((nrow, ncol), dtype=float)
    print(f'Memory of data_array: {(data.nbytes * 10 ** (-6))} MB.')

    for i, line in enumerate(f):
        if extension == ".bz2" or extension == ".gz":
            line = line.decode()
        sample_id, sample_data = line.strip().split(None, 1)
        IDs[i] = sample_id
        data[i] = sample_data.split()

    print(f'Read {nrow} samples in total.\n')

    # close file handle
    f.close()

    return IDs, data


def shuffle_arrays_in_unison(array_a, array_b, seed=None):
    """This function shuffles two numpy arrays so that the indices
    between them will still correspond to one another.
    """
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(array_a)
    np.random.set_state(rng_state)
    np.random.shuffle(array_b)


def split_array(array, array_size, percent_to_first):
    """This splits the array by percent of rows
    and returns numpy slices (no copies).
    """

    split_index = int(percent_to_first * array_size)

    if array.ndim == 2:
        array_1 = array[:split_index, :]
        array_2 = array[split_index:, :]
    elif array.ndim == 1:
        array_1 = array[:split_index]
        array_2 = array[split_index:]
    else:
        raise Exception('Split_array is only implemented ' \
                        'for 1 and 2 dimensional arrays.')

    return array_1, array_2


if __name__ == '__main__':
    utils = AMCPUtils()
    utils.run()
