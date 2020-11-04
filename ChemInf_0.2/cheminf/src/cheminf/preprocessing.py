import os

import numpy as np
import pandas as pd
from sklearn.utils import resample

from ..cheminf.utils import read_dataframe, save_dataframe, shuffle_dataframe

MULTICORE_SUPPORT = ['resample']

class ChemInfPreProc(object):
    def __init__(self, controller):
        self.shuffle = controller.args.shuffle
        self.project_name = controller.args.name
        self.infile = controller.args.infile
        if controller.args.mode == 'preproc':
            self.mode = controller.args.preproc_mode
            self.percentage = controller.args.percentage
        else:
            self.mode = 'auto'
            self.percentage = controller.config.execute.train_test_ratio
            self.auto_plus_resample = controller.config.execute.auto_plus_resample
        self.nr_core = controller.args.nr_core
        self.chunksize = controller.args.chunksize
        self.outfile = controller.args.outfile
        self.outfile2 = controller.args.outfile2
        self.src_dir = controller.src_dir

    def run(self):
        if self.nr_core:
            if self.nr_core > 1:
                if self.mode in MULTICORE_SUPPORT:
                    self._multicore()
                else:
                    raise ValueError(f"{self.mode} doesn't have multicore support")
            else:
                self._single_thread()
        else:
            if self.mode == 'auto':
                if self.auto_plus_resample:
                    dataframe = self._run_auto_resample(save=False)
                else:
                    dataframe = None
                self._run_auto_split(dataframe)
            else:
                raise ValueError(f"{self.mode} can't be run with preproc")

    def _run_auto_resample(self, infile=None, save=True):
        if infile is None:
            infile = self.infile
        if save:
            file_path = self._make_auto_outfile('balanced')
            self.mode = 'resample'
            self._single_thread(infile, outfile=file_path)
            self.mode = 'auto'
            return _remove_name_sep(file_path)
        else:
            self.mode = 'resample'
            dataframe = self._single_thread(infile, save=False)
            self.mode = 'auto'
            return dataframe

    def _run_auto_split(self, infile=None, save=True):
        if infile is None:
            infile = self.infile

        if save:
            train_file_path = self._make_auto_outfile('train')
            test_file_path = self._make_auto_outfile('test')
            print(train_file_path)
            print(test_file_path)
            self.mode = 'split'
            self._single_thread(infile, outfile=train_file_path, outfile2=test_file_path)
            self.mode = 'auto'
            return {'train': train_file_path, 'test': test_file_path}
        else:
            train_dataframe, test_dataframe = self._single_thread(infile, save=False)
            self.mode = 'auto'
            return train_dataframe, test_dataframe

    def _make_auto_outfile(self, suffix, file=None):
        if file is None:
            file = self.infile
        file_dir = os.path.basename(file)
        file_name, file_extension = os.path.splitext(file_dir)
        if self.auto_plus_resample:
            return f"{self.src_dir}/data/{file_name}_{suffix}{file_extension}"
        else:
            return f"{self.src_dir}/data/{file_name}_{suffix}{file_extension}"

    def _single_thread(self, infile=None, outfile=None, outfile2=None, save=True):
        if infile is None:
            infile = self.infile

        if outfile is None:
            outfile = self.outfile

        if self.mode == 'resample':
            dataframe = self.resample(infile=infile)
            if save:
                save_dataframe(dataframe, outfile)
            else:
                return dataframe

        else:
            if self.mode == 'split':
                if outfile2 is None:
                    outfile2 = self.outfile2
                dataframe_large, dataframe_small = self.cut(split=True)
                if save:
                    save_dataframe(dataframe_large, outfile)
                    save_dataframe(dataframe_small, outfile2)
                return dataframe_large, dataframe_small
            elif self.mode == 'trim':
                dataframe = self.cut(infile)
                if save:
                    save_dataframe(dataframe, outfile)
                else:
                    return dataframe

    def _multicore(self):
        if self.mode not in ['split', 'trim']:
            global mp
            mp = __import__('multiprocessing', globals(), locals())
            print(f"Available cores: {mp.cpu_count()}")
            ctx = mp.get_context('spawn')
            with ctx.Pool(self.nr_core) as pool:
                dataframe = pd.DataFrame()
                if self.mode == "resample":
                    chunks = read_dataframe(self.infile, chunksize=self.chunksize)
                    print(f"Starting multicore resampling with {self.nr_core} cores")
                    pool_stack = pool.imap_unordered(resample_dataframe, chunks)
                    for i, pool_dataframe in enumerate(pool_stack):
                        print(f"\nWorking on chunk {i}\n-----------------------------------")
                        dataframe = pd.concat([dataframe, pool_dataframe])

                pool.close()
                pool.join()

                dataframe = shuffle_dataframe(dataframe)
                save_dataframe(dataframe, self.outfile)
        else:
            raise ValueError(f"{self.mode} doesn't have multicore support")

    def resample(self, infile=None):
        if infile is None:
            infile = self.infile
        chunks = read_dataframe(infile, chunksize=self.chunksize)
        dataframe = pd.DataFrame()

        if self.chunksize:
            print(f"\nStart creating resampled dataframe from {infile} in chunks with size {self.chunksize} rows")
            for i, chunk in enumerate(chunks):
                print(f"\nWorking on chunk {i}\n-----------------------------------")
                dataframe_chunk = resample_dataframe(chunk)
                dataframe = pd.concat([dataframe, dataframe_chunk])

        else:
            print(f"\nStart creating resampled dataframe from {infile}")
            dataframe = resample_dataframe(chunks)

        print("\nShuffles the dataset")
        return shuffle_dataframe(dataframe)

    # Todo: Implement multicore support for trim and split mode
    def cut(self, infile=None, dataframe=None, split=False):
        if dataframe is None:
            if infile is None:
                infile = self.infile
            dataframe = read_dataframe(infile, shuffle=self.shuffle)

        index = int(np.round(len(dataframe) * self.percentage))
        dataframe1 = dataframe.iloc[:index, :]
        if split:
            dataframe2 = dataframe.iloc[index:, :]
            return dataframe1, dataframe2
        else:
            return dataframe1

    def get(self, key):
        return getattr(self, key)


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
