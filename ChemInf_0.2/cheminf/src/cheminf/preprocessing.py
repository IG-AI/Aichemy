import os

import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.cheminf.operator import read_dataframe, save_dataframe, remove_name_sep


class ChemInfPreProc(object):
    def __init__(self, controller, in_file=None):
        if hasattr(controller.args, 'percentage'):
            self.percentage = controller.args.percentage
        if hasattr(controller.args, 'shuffle'):
            self.shuffle = controller.args.shuffle
        self.project_name = controller.project_name
        self.in_file = controller.args.in_file
        self.mode = controller.args.postproc_mode
        self.nr_core = controller.args.nr_core
        self.chunksize = controller.args.chunksize
        self.out_file = controller.args.out_file
        self.out_file2 = None
        self.src_dir = controller.src_dir
        self.auto_plus_resample = controller.config.execute.auto_plus_resample

    def run(self):
        if self.nr_core > 1:
            if self.mode in ['split', 'trim']:
                raise ValueError(f"{self.mode} don't have multicore support")
            self._multicore()
        else:
            self._single_thread()

    def run_auto(self):
        if not self.mode == 'auto':
            raise ValueError("preproc.run_auto() can only be executed in auto mode")
        if self.auto_plus_resample:
            out_file = self._make_auto_file('balanced')
            self._run_auto_resample(out_file)
            in_file = remove_name_sep(out_file)
            train_file_path = self._make_auto_file('train', file=in_file)
            test_file_path = self._make_auto_file('test', file=in_file)
        else:
            in_file = self.in_file
            train_file_path = self._make_auto_file('train')
            test_file_path = self._make_auto_file('test')

        self.mode = 'split'
        self._single_thread(in_file=in_file, out_file=train_file_path, out_file2=test_file_path)

        return {'train': remove_name_sep(train_file_path), 'test': remove_name_sep(test_file_path)}

    def _run_auto_resample(self, out_file):
        self.mode = 'resample'
        self._single_thread(out_file=out_file)

    def _make_auto_file(self, suffix, file=None):
        if file is None:
            file = self.in_file
        file_dir = os.path.dirname(file)
        file_name, file_extension = os.path.splitext(file_dir)
        if self.auto_plus_resample:
            return f"{self.src_dir}/data/{file_name}_{suffix}.{file_extension}"
        else:
            return f"{self.src_dir}/data/{file_name}_{suffix}.{file_extension}"

    def _single_thread(self, in_file=None, out_file=None, out_file2=None, save=True):
        if in_file is None:
            in_file = self.in_file

        if out_file is None:
            out_file = self.out_file

        if self.mode == 'resample':
            dataframe = resample_file(in_file, self.chunksize)
            if save:
                save_dataframe(dataframe, out_file)
            else:
                return dataframe
        else:
            if self.mode == 'split':
                if out_file2 is None:
                    out_file2 = self.out_file2
                dataframe_large, dataframe_small = cut_file(in_file, self.percentage, self.shuffle, split=True)
                if save:
                    save_dataframe(dataframe_large, out_file)
                    save_dataframe(dataframe_small, out_file2)
                return dataframe_large, dataframe_small

            elif self.mode == 'trim':
                dataframe = cut_file(in_file, self.percentage, self.shuffle)
                if save:
                    save_dataframe(dataframe, out_file)
                else:
                    return dataframe

    def _multicore(self):
        if self.mode not in ['split', 'trim']:
            global mp
            mp = __import__('multiprocessing', globals(), locals())
            print(f"Available cores: {mp.cpu_count()}")
            ctx = mp.get_context('spawn')
            with ctx.Pool(args.nr_core) as pool:
                dataframe = pd.DataFrame()
                if self.mode == "resample":
                    chunks = read_dataframe(self.in_file, chunksize=self.chunksize)
                    print(f"Starting multicore resampling with {self.nr_core} cores")
                    pool_stack = pool.imap_unordered(resample_dataframe, chunks)
                    for i, pool_dataframe in enumerate(pool_stack):
                        print(f"\nWorking on chunk {i}\n-----------------------------------")
                        dataframe = pd.concat([dataframe, pool_dataframe])

                pool.close()
                pool.join()

                dataframe = shuffle_dataframe(dataframe)
                save_dataframe(dataframe, self.out_file)
        else:
            raise ValueError(f"{self.mode} doesn't have multicore support")

    def get(self, key):
        return getattr(self, key)

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
        raise ("Split_array is only implemented for 1 and 2 dimensional arrays.")

    return array_1, array_2
