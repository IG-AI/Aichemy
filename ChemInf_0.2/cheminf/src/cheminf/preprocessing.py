import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from random import randrange
from abc import ABCMeta, abstractmethod

from ..cheminf.utils import read_dataframe, save_dataframe, shuffle_dataframe, mutually_exclusive, \
    ModeError, NoMultiCoreSupport

MULTICORE_SUPPORT = ['balancing']

class ChemInfPreProc(object, metaclass=ABCMeta):
    def __init__(self, controller):
        self.shuffle = controller.args.shuffle
        self.name = controller.args.name
        self.infile = controller.args.infile
        self.percentage = controller.args.percentage
        self.mode = controller.args.mode
        if controller.args.nr_core:
            self.nr_core = controller.args.nr_core
        else:
            self.nr_core = 1
        self.chunksize = controller.args.chunksize
        self.outfile = controller.args.outfile
        self.outfile2 = controller.args.outfile2
        self.src_dir = controller.src_dir

    @abstractmethod
    def run(self):
        pass

    def _single_thread(self, submode, dataframe=None, outfile=None, outfile2=None, save=True):
        if dataframe is None:
            infile = self.infile
            dataframe = read_dataframe(infile, self.chunksize)
        if outfile is None:
            outfile = self.outfile

        if submode == 'split':
            if outfile2 is None:
                outfile2 = self.outfile2
            dataframe_large, dataframe_small = self.split(submode, dataframe)
            if save:
                save_dataframe(dataframe_large, outfile)
                save_dataframe(dataframe_small, outfile2)
                return outfile, outfile2
            else:
                return dataframe_large, dataframe_small

        elif submode == 'trim' or submode == 'balancing' or submode == 'resample':
            utils = getattr(self, submode)
            dataframe = utils(dataframe)
        else:
            raise ModeError(self.mode, submode)

        if save:
            save_dataframe(dataframe, outfile)
            return outfile
        else:
            return dataframe

    # Todo: Increase multicore-support
    def _multicore(self, submode):
        if submode == "balancing":
            global mp
            mp = __import__('multiprocessing', globals(), locals())
            print(f"Available cores: {mp.cpu_count()}")
            ctx = mp.get_context('spawn')
            with ctx.Pool(self.nr_core) as pool:
                dataframe = pd.DataFrame()
                chunks = read_dataframe(self.infile, chunksize=self.chunksize)
                print(f"Starting multicore resampling with {self.nr_core} cores")
                pool_stack = pool.imap_unordered(balancing_dataframe, chunks)
                for i, pool_dataframe in enumerate(pool_stack):
                    print(f"\nWorking on chunk {i}\n-----------------------------------")
                    dataframe = pd.concat([dataframe, pool_dataframe])

                pool.close()
                pool.join()

                dataframe = shuffle_dataframe(dataframe)
                save_dataframe(dataframe, self.outfile)
        else:
            raise NoMultiCoreSupport(submode)

    def balancing(self, dataframe_chunks=None):
        return self._sample('balancing', dataframe_chunks)

    def resample(self, dataframe_chunks=None):
        return self._sample('resample', dataframe_chunks)

    def _sample(self, submode, dataframe_chunks=None):
        if submode == 'balancing' or submode == 'resample':
            if dataframe_chunks is None:
                infile = self.infile
                dataframe_chunks = read_dataframe(infile, chunksize=self.chunksize)
            dataframe_results = pd.DataFrame()

            print1 = f"\nStart creating balanced dataframe"
            if self.chunksize:
                print2 = f" in chunks with size {self.chunksize} rows"
            else:
                print2 = ""
            print(print1 + print2)

            for i, chunk in enumerate(dataframe_chunks):
                if self.chunksize:
                    print(f"\nWorking on chunk {i}\n-----------------------------------")

                if submode == 'balancing':
                    dataframe_chunk = balancing_dataframe(chunk, percentage=self.percentage)
                elif submode == 'resample':
                    dataframe_chunk = resample_dataframe(chunk, percentage=self.percentage)

                dataframe_results = pd.concat([dataframe_results, dataframe_chunk])

            print("\nShuffles the dataset")
            return shuffle_dataframe(dataframe_results)
        else:
            raise ModeError(self.mode, submode)

    def split(self, dataframe=None, index=None):
        return self._cut('split', dataframe, index)

    def trim(self, dataframe=None, index=None):
        return self._cut('split', dataframe, index)

    # Todo: Implement chunking-support for trim and split submode
    def _cut(self, submode, dataframe=None, index=None):
        if submode != 'balancing' and submode != 'resample':
            if dataframe is None:
                dataframe = read_dataframe(self.infile, shuffle=self.shuffle)

            if index is None:
                if submode == 'split':
                    return split_dataframe(dataframe, percentage=self.percentage)
                elif submode == 'trim':
                    return trim_dataframe(dataframe, percentage=self.percentage)
            else:
                if submode == 'split':
                    return split_dataframe(dataframe, index=index)
                elif submode == 'trim':
                    return trim_dataframe(dataframe, index=index)
        else:
            raise ModeError(self.mode, submode)

    def get(self, key):
        return getattr(self, key)


class PreProcNormal(ChemInfPreProc):
    def __init__(self, controller):
        super(PreProcNormal, self).__init__(controller)
        self.submode = controller.args.preproc_mode

    def run(self):
        if self.nr_core > 1:
            if self.mode in MULTICORE_SUPPORT:
                self._multicore(self.submode)
            else:
                raise NoMultiCoreSupport(self.submode)
        else:
            print(self.submode)
            self._single_thread(self.submode)


class PreProcAuto(ChemInfPreProc):
    def __init__(self, controller):
        super(PreProcAuto, self).__init__(controller)
        self.train_test_ratio = controller.config.execute.train_test_ratio
        self.sample_ratio = controller.config.execute.sample_ratio
        self.auto_plus_balancing = (self.mode == 'auto' and controller.config.execute.auto_plus_balancing)
        self.auto_plus_resample = (self.mode == 'auto' and controller.config.execute.auto_plus_resample)

    def run(self):
        if self.auto_plus_balancing and self.auto_plus_resample:
            dataframe = self._run_auto_mode('balancing', save=False)
            dataframe = self._run_auto_mode('resample', dataframe, save=False)
        elif self.auto_plus_balancing:
            dataframe = self._run_auto_mode('balancing', save=False)
        elif self.auto_plus_resample:
            dataframe = self._run_auto_mode('resample', save=False)
        else:
            dataframe = None
        return self._run_auto_mode('split', dataframe)

    def _run_auto_mode(self, submode, dataframe=None, save=True):
        if submode == 'split':
            train_file_path = self.__make_auto_outfile('train')
            test_file_path = self.__make_auto_outfile('test')
            self.percentage = self.train_test_ratio
            if save:
                self._single_thread(submode, dataframe, outfile=train_file_path, outfile2=test_file_path)
                self.percentage = None
                return {'train': train_file_path, 'test': test_file_path}
            else:
                train_dataframe, test_dataframe = self._single_thread(submode, dataframe, save=False)
                self.percentage = None
                return train_dataframe, test_dataframe

        elif submode == 'resample' or submode == 'balancing':
            if submode == 'resample':
                self.percentage = self.sample_ratio
            file_path = self.__make_auto_outfile(submode)
            if save:
                dataframe_path = self._single_thread('balancing', dataframe, outfile=file_path)
                self.percentage = None
                return dataframe_path
            else:
                dataframe = self._single_thread('balancing', dataframe, save=False)
                self.percentage = None
                return dataframe
        else:
            ModeError(self.mode, submode)

    def __make_auto_outfile(self, suffix, file=None):
        if file is None:
            file = self.infile

        file_dir = os.path.basename(file)
        file_name, file_extension = os.path.splitext(file_dir)
        if self.auto_plus_balancing:
            return f"{self.src_dir}/data/{file_name}_{suffix}{file_extension}"
        else:
            return f"{self.src_dir}/data/{file_name}_{suffix}{file_extension}"


def balancing_dataframe(dataframe, percentage=1):
    dataframe_class0 = dataframe[dataframe['class'] == 0]
    dataframe_class1 = dataframe[dataframe['class'] == 1]

    dataframe_class1_resampled = resample_dataframe(dataframe_class1, percentage)

    data_div = dataframe_class1_resampled['class'].value_counts()
    nr_samples = int(np.round(data_div[1], decimals=0))
    print(data_div[1])
    print(type(data_div[1]))
    dataframe_class0_balancing = resample(dataframe_class0,
                                          replace=False,
                                          n_samples=nr_samples,
                                          random_state=randrange(100, 999))

    return pd.concat([dataframe_class0_balancing, dataframe_class1_resampled])


def resample_dataframe(dataframe, percentage=1):
    return dataframe.sample(frac=percentage, random_state=randrange(100, 999), axis=0)


def split_dataframe(dataframe, percentage=None, index=None, axis=0):
    mutually_exclusive(percentage, index)
    if percentage:
        index = int(np.round(len(dataframe) * percentage))

    if axis == 0:
        dataframe1 = dataframe.iloc[:index, :]
        dataframe2 = dataframe.iloc[index:, :]
        return dataframe1, dataframe2
    elif axis == 1:
        dataframe1 = dataframe.iloc[:, :index]
        dataframe2 = dataframe.iloc[:, index:]
        return dataframe1, dataframe2
    else:
        raise ValueError("Unsupported axis parameter")


def trim_dataframe(dataframe, percentage=None, index=None, axis=0):
    mutually_exclusive(percentage, index)
    if percentage:
        index = int(np.round(len(dataframe) * percentage))

    if axis == 0:
        dataframe1 = dataframe.iloc[:index, :]
        return dataframe1
    elif axis == 1:
        dataframe1 = dataframe.iloc[:, :index]
        return dataframe1
    else:
        raise ValueError("Unsupported axis parameter")
