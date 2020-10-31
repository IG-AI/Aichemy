import os
import re

import numpy as np
import pandas as pd
from sklearn.utils import resample


class ChemInfUtils(object):
    def __init__(self, database):
        self.args = database.args

    def run(self):
        def single_thread(args):
            if self.args.mode == 'resample':
                dataframe = resample_file(args.in_file, args.chunksize)
                save_dataframe(dataframe, args.out_file)

            else:
                if args.utils_mode == 'split':
                    dataframe_large, dataframe_small = cut_file(args.in_file, args.percentage, args.shuffle,
                                                                split=True)
                    save_dataframe(dataframe_large, args.out_file)
                    save_dataframe(dataframe_small, args.out_file2)

                elif args.utils_mode == 'trim':
                    dataframe = cut_file(args.in_file, args.percentage, args.shuffle)
                    save_dataframe(dataframe, args.out_file)

        def multicore(args):
            if args.utils_mode not in ['split', 'trim']:
                global mp
                mp = __import__('multiprocessing', globals(), locals())
                print(f"Available cores: {mp.cpu_count()}")
                ctx = mp.get_context('spawn')
                with ctx.Pool(args.num_core) as pool:
                    dataframe = pd.DataFrame()
                    if args.mode == "resample":
                        chunks = read_dataframe(args.in_file, chunksize=args.chunksize)
                        print(f"Starting multicore resampling with {args.num_core} cores")
                        pool_stack = pool.imap_unordered(resample_dataframe, chunks)
                        for i, pool_dataframe in enumerate(pool_stack):
                            print(f"\nWorking on chunk {i}\n-----------------------------------")
                            dataframe = pd.concat([dataframe, pool_dataframe])

                    pool.close()
                    pool.join()

                    dataframe = shuffle_dataframe(dataframe)
                    save_dataframe(dataframe, args.out_file)

            else:
                raise ValueError(f"{self.args.mode} doesn't have multicore support")

        if self.args.num_core > 1:
            if self.args.utils_mode in ['split', 'trim']:
                raise ValueError(f"{self.args.mode} don't have multicore support")
            multicore(self.args)
        else:
            single_thread(self.args)


def resample_file(file, chunksize):
    chunks = read_dataframe(file, chunksize=chunksize)
    dataframe = pd.DataFrame()

    print(f"\nStart creating resampled dataframe from {file} in chunks with size {chunksize} rows")

    for i, chunk in enumerate(chunks):
        print(f"\nWorking on chunk {i}\n-----------------------------------")
        dataframe_chunk = resample_dataframe(chunk)
        dataframe = pd.concat([dataframe, dataframe_chunk])

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
    print(f"\nSave dataframe as csv to {out_file}")

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
                "The parameter file should have one key value pair per line"
            params_dict[items[0]] = items[1]


def read_array(in_file, data_type):
    """Reads a white space separated file without a header.
    The first column contains the IDs of samples and the 2nd the class.
    All the following columns contain features (the independent variables)
    of the sample. The function first determines the size of the arrays to
    be created, then inserts data into them.
    """
    print(f"Reading from '{in_file}'.")

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

    nrow = sum(1 for _ in f)
    f.seek(0)
    ncol = len(f.readline().split()) - 1  # Disregarding ID column.
    f.seek(0)
    # Initializing arrays.
    print(f"Initializing array of size {nrow} X {ncol}.")
    id = np.empty(nrow, dtype=np.dtype('U50'))
    print(f"Memory of ID_array(str): {(id.nbytes * 10 ** (-6))} MB.")
    if data_type == 'integer':
        data = np.empty((nrow, ncol), dtype=int)
    if data_type == 'float':
        data = np.empty((nrow, ncol), dtype=float)
    print(f"Memory of data_array: {(data.nbytes * 10 ** (-6))} MB.")

    for i, line in enumerate(f):
        if extension == ".bz2" or extension == ".gz":
            line = line.decode()
        sample_id, sample_data = line.strip().split(None, 1)
        id[i] = sample_id
        data[i] = sample_data.split()

    print(f"Read {nrow} samples in total.\n")

    # close file handle
    f.close()

    return id, data


def shuffle_arrays_in_unison(array_a, array_b, seed=None):
    """This function shuffles two numpy arrays so that the indices
    between them will still correspond to one another.
    """
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(array_a)
    np.random.set_state(rng_state)
    np.random.shuffle(array_b)


def split_array(array, percent_to_first, array_size=None, shuffle=False):
    """This splits the array by percent of rows
    and returns numpy slices (no copies).
    """
    if shuffle:
        np.random.seed(123)
        np.random.shuffle(array)

    if array_size is None:
        array_size = array.size

    split_index = int(percent_to_first * array_size)

    if array.ndim == 2:
        array_1 = array[:split_index, :]
        array_2 = array[split_index:, :]
    elif array.ndim == 1:
        array_1 = array[:split_index]
        array_2 = array[split_index:]
    else:
        raise Exception("Split_array is only implemented for 1 and 2 dimensional arrays.")

    return array_1, array_2
