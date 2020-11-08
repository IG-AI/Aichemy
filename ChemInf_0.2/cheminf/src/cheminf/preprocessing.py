import os
import numpy as np
import pandas as pd

from pandas.io.parsers import TextFileReader as Chunks
from sklearn.utils import resample
from random import randrange
from abc import ABCMeta, abstractmethod

from ..cheminf.utils import read_dataframe, save_dataframe, shuffle_dataframe, MutuallyExclusive, \
    ModeError, NoMultiCoreSupport

MULTICORE_SUPPORT = ['balancing', 'resample']

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

    def _single_core(self, submode, dataframes=None, outfile=None, outfile2=None, save=True):
        if dataframes is None:
            infile = self.infile
            dataframes = read_dataframe(infile, self.chunksize)
        if outfile is None:
            outfile = self.outfile

        if submode == 'split':
            if outfile2 is None:
                outfile2 = self.outfile2
            dataframe_large, dataframe_small = self.split(dataframes)
            if save:
                save_dataframe(dataframe_large, outfile)
                save_dataframe(dataframe_small, outfile2)
                return outfile, outfile2
            else:
                return dataframe_large, dataframe_small

        elif submode == 'trim' or submode == 'balancing' or submode == 'resample':
            utils = getattr(self, submode)
            dataframe = utils(dataframes)
        else:
            raise ModeError(self.mode, submode)

        if save:
            save_dataframe(dataframe, outfile)
            return outfile
        else:
            return dataframe

    # Todo: Make fail-safe when chunksize is smaller than the rows in the input dataframe
    def _multicore(self, submode, dataframes=None, outfile=None, outfile2=None, save=True):
        if submode in MULTICORE_SUPPORT:
            if dataframes is None:
                infile = self.infile
                dataframes = read_dataframe(infile, self.chunksize)
            if outfile is None:
                outfile = self.outfile

            global mp
            mp = __import__('multiprocessing', globals(), locals())
            print(f"Available cores: {mp.cpu_count()}")

            ctx = mp.get_context('spawn')
            with ctx.Pool(self.nr_core) as pool:
                dataframe = pd.DataFrame()
                print(f"Starting multicore resampling with {self.nr_core} cores")
                utils = getattr(self, submode)
                pool_stack = pool.imap_unordered(utils, dataframes)
                for i, pool_dataframe in enumerate(pool_stack):
                    print(f"\nWorking on chunk {i}\n-----------------------------------")
                    dataframe = pd.concat([dataframe, pool_dataframe])

                pool.close()
                pool.join()

                dataframe = shuffle_dataframe(dataframe)
                save_dataframe(dataframe, self.outfile)
        else:
            raise NoMultiCoreSupport(submode)

    def balancing(self, dataframes=None):
        return self._sample('balancing', dataframes)

    def resample(self, dataframes=None):
        return self._sample('resample', dataframes)

    def _sample(self, submode, dataframes=None):
        if submode == 'balancing' or submode == 'resample':
            if dataframes is None:
                infile = self.infile
                dataframes = read_dataframe(infile, chunksize=self.chunksize)
            dataframe_results = pd.DataFrame()

            if self.nr_core == 1:
                print1 = f"\nStart {submode} dataframe"
                if self.chunksize:
                    print2 = f" in chunks with size {self.chunksize} rows"
                else:
                    print2 = ""
                print(print1 + print2)

            # Checks that ChemInf operates in single core and that the dataframe is chunked
            if self.nr_core == 1 and isinstance(dataframes, Chunks):
                for i, chunk in enumerate(dataframes):
                    if self.chunksize:
                        print(f"\nWorking on chunk {i}\n-----------------------------------")
                    if submode == 'balancing':
                        dataframe_chunk = balancing_dataframe(chunk, percentage=self.percentage)
                    elif submode == 'resample':
                        dataframe_chunk = resample_dataframe(chunk, percentage=self.percentage)

                    dataframe_results = pd.concat([dataframe_results, dataframe_chunk])

            elif self.nr_core > 1 or isinstance(dataframes, pd.DataFrame):
                if submode == 'balancing':
                    dataframe_results = balancing_dataframe(dataframes, percentage=self.percentage)
                elif submode == 'resample':
                    dataframe_results = resample_dataframe(dataframes, percentage=self.percentage)

            else:
                raise ValueError("Unrecognized instance of 'dataframes'")

            print("\nShuffles the dataset")
            return shuffle_dataframe(dataframe_results)
        else:
            raise ModeError(self.mode, submode)

    def split(self, dataframe=None, index=None):
        return self._cut('split', dataframe, index)

    def trim(self, dataframe=None, index=None):
        return self._cut('split', dataframe, index)

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
            if self.submode in MULTICORE_SUPPORT:
                self._multicore(self.submode)
            else:
                raise NoMultiCoreSupport(self.submode)
        else:
            self._single_core(self.submode)


# Todo: Implement multicore support fro preproc-auto mode
class PreProcAuto(ChemInfPreProc):
    def __init__(self, controller):
        super(PreProcAuto, self).__init__(controller)
        self.train_test_ratio = controller.config.execute.train_test_ratio
        self.resample_ratio = controller.config.execute.resample_ratio
        self.auto_save_preproc = controller.config.execute.auto_save_preproc
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

        if self.auto_save_preproc:
            data = self._run_auto_mode('split', dataframe)
        else:
            data = self._run_auto_mode('split', dataframe, save=False)
        return data

    def _run_auto_mode(self, submode, dataframe=None, save=True):
        if submode == 'split':
            train_file_path = self._make_auto_outfile('train')
            test_file_path = self._make_auto_outfile('test')
            self.percentage = self.train_test_ratio
            if save:
                self._single_core(submode, dataframe, outfile=train_file_path, outfile2=test_file_path)
                self.percentage = None
                return {'train': train_file_path, 'test': test_file_path}
            else:
                # Todo: Fix bug with piping of dataframe to models
                raise NotImplementedError("Not saving preproc results not yet implemented")
                """
                train_dataframe, test_dataframe = self._single_core(submode, dataframe, save=False)
                self.percentage = None
                return {'train': train_dataframe, 'test': test_dataframe}
                """

        elif submode == 'resample' or submode == 'balancing':
            if submode == 'resample':
                self.percentage = self.resample_ratio
            elif submode == 'balancing':
                self.percentage = 1
            file_path = self._make_auto_outfile(submode)
            if save:
                dataframe_path = self._single_core(submode, dataframe, outfile=file_path)
                self.percentage = None
                return dataframe_path
            else:
                dataframe = self._single_core(submode, dataframe, save=False)
                self.percentage = None
                return dataframe
        else:
            ModeError(self.mode, submode)

    def _make_auto_outfile(self, suffix, file=None):
        if file is None:
            file = self.infile

        file_dir = os.path.basename(file)
        file_name, file_extension = os.path.splitext(file_dir)
        return f"{self.src_dir}/data/{self.name}/{file_name}_{suffix}{file_extension}"


def balancing_dataframe(dataframe, percentage=1):
    dataframe_class0 = dataframe[dataframe['class'] == 0]
    dataframe_class1 = dataframe[dataframe['class'] == 1]

    dataframe_class1_resampled = resample_dataframe(dataframe_class1, 1)

    data_div = dataframe_class1_resampled['class'].value_counts()
    nr_samples = int(np.round(data_div[1]*percentage, decimals=0))

    dataframe_class0_balancing = resample(dataframe_class0,
                                          replace=False,
                                          n_samples=nr_samples,
                                          random_state=randrange(100, 999))

    return pd.concat([dataframe_class0_balancing, dataframe_class1_resampled])


def resample_dataframe(dataframe, percentage=1):
    return dataframe.sample(frac=percentage, random_state=randrange(100, 999), axis=0)


def split_dataframe(dataframe, percentage=None, index=None, axis=0):
    if (percentage and index) or (percentage is None and index is None):
        raise MutuallyExclusive('percentage', 'index')
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
    if (percentage and index) or (percentage is None and index is None):
        raise MutuallyExclusive('percentage', 'index')
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
