import os
import re
import time

import numpy as np
import pandas as pd
from random import randrange
from functools import wraps


class Timer(object):
    def __init__(self, tag):
        if tag[0].isupper():
            runtime_tag = "Runtime"
        else:
            runtime_tag = "runtime"
        self.__prefix = f"\n{tag} {runtime_tag}\n-----------------------------------\n"
        self.__start_time = None

    def __repr__(self):
        runtime = self.__get_runtime()
        if runtime:
            f"{runtime['hours']:d}:{runtime['min']:02d}:{runtime['sec']:02d}.{runtime['centisec']:d}"
        else:
            return "0:00:00.0"

    def __str__(self):
        return f"{self.__prefix}" \
               f"{self.__repr__()}"

    def __get_runtime(self):
        if self.__start_time:
            current_time = np.round(time.time() - self.__start_time, decimals=2)
            runtime_sec, runtime_centisec = divmod(current_time, 1)
            runtime_min, runtime_sec = divmod(runtime_sec, 60)
            runtime_h, runtime_min = divmod(runtime_min, 60)
            return {'h': runtime_h, 'min': runtime_min, 'sec': runtime_sec, 'centisec': runtime_centisec}
        else:
            return None

    def start(self):
        self.__start_time = time.time()

    def lap(self):
        return f"{self.__prefix}The current runtime is {self.__repr__()}"

    def stop(self):
        return f"{self.__prefix}The runtime was {self.__repr__()}"

    def reset(self):
        self.__start_time = time.time()


# USAGE EXAMPLE:
# @TimerWrapper
# function(input)
class WrapperTimer(Timer):
    def __init__(self, func):
        super(WrapperTimer, self).__init__(func)
        self.func = func

    def __call__(self, *args):
        self.start()
        self.func(*args)
        self.end()


# USAGE EXAMPLE:
# @mutually_exclusive_wrapper(do, second_do, third_do)
# function(input, output, do, second_do, third_do)
def mutually_exclusive_wrapper(exclusive, *exclusives):
    exclusives = (exclusive,) + exclusives

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if sum(arg in exclusives for arg in kwargs) != 1:
                raise TypeError('You must specify exactly one of {}'.format(', '.join(exclusives)))
            return func(*args, **kwargs)

        return inner

    return wrapper


def read_parameters(infile, params_dict):
    """ Reads a file with hyperparameters for the
    classifier and parameters for the CP script. Updates the params_dict.
    """
    with open(infile, 'r') as f:
        for line in f:
            items = line.strip().split()
            assert len(items) == 2, \
                "The parameter file should have one key value pair per line"
            params_dict[items[0]] = items[1]


def read_array(infile, data_type):
    """Reads a white space separated file without a header.
    The first column contains the IDs of samples and the 2nd the class.
    All the following columns contain features (the independent variables)
    of the sample. The function first determines the size of the arrays to
    be created, then inserts data into them.
    """
    print(f"Reading from '{infile}'.")

    # read (compressed) features
    file_name, extension = os.path.splitext(infile)
    if extension == ".bz2":
        import bz2
        f = bz2.open(infile, 'rb')
    elif extension == ".gz":
        import gzip
        f = gzip.open(infile, 'rb')
    else:
        f = open(infile, 'r')

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


def read_dataframe(infile, chunksize=None, shuffle=False):
    print("\nReading from {file}".format(file=infile))

    with open(infile) as fin:
        first_line = next(fin)
        line_data = re.split(" |\t|[|]", first_line)
        num_cols = len(line_data)

    unnamed_columns = [str(i) for i in range(1, num_cols - 1)]
    columns_names = ['id', 'class'] + unnamed_columns

    columns_types = {'id': 'string', 'class': 'int8'}
    for i in range(1, (num_cols - 1)):
        columns_types[str(i)] = 'int8'

    if chunksize:
        dataframe = pd.read_csv(infile,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                chunksize=chunksize,
                                iterator=True,
                                engine='c')

    else:
        dataframe = pd.read_csv(infile,
                                sep='\s+',
                                skip_blank_lines=True,
                                header=None,
                                names=columns_names,
                                dtype=columns_types,
                                engine='c')

        if shuffle:
            dataframe = shuffle_dataframe(dataframe)

    return dataframe


def save_dataframe(dataframe, outfile):
    print(f"\nSave dataframe as csv to {outfile}")

    dataframe.to_csv(outfile,
                     index=False,
                     header=False,
                     sep='\t')


def shuffle_dataframe(dataframe):
    return dataframe.sample(frac=1, random_state=randrange(100, 999), axis=0)


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


class ModeError(Exception):
    def __init__(self, mode, submode):
        message = f"Submode '{submode}' isn't supported with {mode} in this code block"
        super(Exception, self).__init__(message)


class UnsupportedClassifierError(Exception):
    def __init__(self, classifier):
        message = f"Unsupported classifier: {classifier}"
        super(Exception, self).__init__(message)


class NoMultiCoreSupportError(Exception):
    def __init__(self, mode):
        message = f"{mode.capitalize()} doesn't have multicore support"
        super(Exception, self).__init__(message)


class MutuallyExclusiveError(Exception):
    def __init__(self, arg1, arg2):
        message = f"You must specify exactly one of {arg1} and {arg2}"
        super(Exception, self).__init__(message)
