import os
import numpy as np
import pandas as pd

from pandas.io.parsers import TextFileReader as Chunks
from sklearn.utils import resample
from random import randrange
from abc import ABCMeta, abstractmethod

from ..cheminf.utils import read_dataframe, save_dataframe, shuffle_dataframe, MutuallyExclusiveError, \
    ModeError, NoMultiCoreSupportError

MULTICORE_SUPPORT = ['balancing', 'sample', 'auto']

# Todo: Make the provided outfile will be used correctly
class ChemInfPreProc(object, metaclass=ABCMeta):
    def __init__(self, controller):
        self.shuffle = controller.args.shuffle
        self.name = controller.args.name
        self.infile = controller.args.infile
        self.percentage = controller.args.percentage
        self.mode = controller.args.mode
        if controller.args.nr_cores:
            self.nr_cores = controller.args.nr_cores
        else:
            self.nr_cores = 1
        self.chunksize = controller.args.chunksize
        self.outfile = controller.args.outfile
        self.outfile2 = controller.args.outfile2
        self.src_dir = controller.src_dir

    @abstractmethod
    def run(self):
        pass

    def _single_core(self, submode, dataframes=None, outfile=None, outfile2=None, save=True):
        if dataframes is None:
            self._check_chunksize()
            dataframes = read_dataframe(self.infile, self.chunksize)
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

        elif submode == 'trim' or submode == 'balancing' or submode == 'sample':
            utils = getattr(self, submode)
            dataframe = utils(dataframes)
        else:
            raise ModeError(self.mode, submode)

        if save:
            save_dataframe(dataframe, outfile)
            return outfile
        else:
            return dataframe

    def _multicore(self, submode, dataframes=None, outfile=None, outfile2=None, save=True):
        import multiprocessing as mp
        if submode in MULTICORE_SUPPORT:
            if dataframes is None:
                self._check_chunksize()
                dataframes = read_dataframe(self.infile, self.chunksize)
            if outfile is None:
                outfile = self.outfile

            dataframe = pd.DataFrame()
            preproc = getattr(self, submode)
            ctx = mp.get_context('spawn')
            print(f"Starting multicore {submode} with {self.nr_cores} cores")
            with ctx.Pool(self.nr_cores) as pool:
                pool_stack = pool.imap_unordered(preproc, dataframes)
                for i, pool_dataframe in enumerate(pool_stack):
                    print(f"\nCombining pool dataframe {i}\n-----------------------------------")
                    dataframe = pd.concat([dataframe, pool_dataframe])
                pool.close()
                pool.join()

            dataframe = shuffle_dataframe(dataframe)
            if save:
                save_dataframe(dataframe, outfile)
                return outfile
            else:
                return dataframe

        else:
            raise NoMultiCoreSupportError(submode)

    def _check_chunksize(self):
        if self.chunksize:
            with open(self.infile) as fin:
                for i in range(self.chunksize):
                    try:
                        next(fin)
                    except StopIteration:
                        print(f"The applied chunk size ({self.chunksize}) is bigger then the input file. Therefore the "
                              f"chunking will disabled")
                        self.chunksize = None
                        break

    def balancing(self, dataframes=None):
        try:
            data = self._sample_or_balancing('balancing', dataframes)
        except Exception as e:
            print(e)
            data = None
        return data

    def sample(self, dataframes=None):
        return self._sample_or_balancing('sample', dataframes)

    def _sample_or_balancing(self, submode, dataframes=None):
        if submode == 'balancing' or submode == 'sample':
            if dataframes is None:
                infile = self.infile
                dataframes = read_dataframe(infile, chunksize=self.chunksize)

            if self.nr_cores == 1:
                print1 = f"\nStart {submode} dataframe"
                if self.chunksize:
                    print2 = f" in chunks with size {self.chunksize} rows"
                else:
                    print2 = ""
                print(print1 + print2)

            dataframe_results = pd.DataFrame()
            if self.nr_cores == 1 and isinstance(dataframes, Chunks):
                for i, chunk in enumerate(dataframes):
                    print(f"\nWorking on chunk {i}\n-----------------------------------")
                    if submode == 'balancing':
                        dataframe_chunk = balancing_dataframe(chunk, percentage=self.percentage)
                    elif submode == 'sample':
                        dataframe_chunk = sample_dataframe(chunk, percentage=self.percentage)
                    else:
                        raise ModeError(self.mode, submode)

                    dataframe_results = pd.concat([dataframe_results, dataframe_chunk])

            elif (self.nr_cores > 1) or isinstance(dataframes, pd.DataFrame):
                if submode == 'balancing':
                    dataframe_results = balancing_dataframe(dataframes, percentage=self.percentage)
                elif submode == 'sample':
                    dataframe_results = sample_dataframe(dataframes, percentage=self.percentage)
                else:
                    raise ModeError(self.mode, submode)
            else:
                raise ValueError("Unrecognized instance of 'dataframes'")

            if not dataframe_results.empty:
                return shuffle_dataframe(dataframe_results)
            else:
                return dataframe_results

        else:
            raise ModeError(self.mode, submode)

    def split(self, dataframe=None, index=None):
        return self._split_or_trim('split', dataframe, index)

    def trim(self, dataframe=None, index=None):
        return self._split_or_trim('trim', dataframe, index)

    def _split_or_trim(self, submode, dataframe=None, index=None):
        if submode == 'split' or submode != 'trim':
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
        if self.nr_cores > 1:
            if self.submode in MULTICORE_SUPPORT:
                self._multicore(self.submode)
            else:
                raise NoMultiCoreSupportError(self.submode)
        else:
            self._single_core(self.submode)


class PreProcAuto(ChemInfPreProc):
    def __init__(self, controller):
        super(PreProcAuto, self).__init__(controller)
        self.train_test_ratio = controller.config.execute.train_test_ratio
        self.sample_ratio = controller.config.execute.sample_ratio
        self.balancing_ratio = controller.config.execute.balancing_ratio
        self.auto_save_preproc = controller.config.execute.auto_save_preproc
        self.auto_plus_balancing = (self.mode == 'auto' and controller.config.execute.auto_plus_balancing)
        self.auto_plus_sample = (self.mode == 'auto' and controller.config.execute.auto_plus_sample)

    def run(self):
        if self.auto_plus_balancing and self.auto_plus_sample:
            dataframe = self._run_auto_mode('balancing', save=False)
            dataframe = self._run_auto_mode('sample', dataframe, save=False)
        elif self.auto_plus_balancing:
            dataframe = self._run_auto_mode('balancing', save=False)
        elif self.auto_plus_sample:
            dataframe = self._run_auto_mode('sample', save=False)
        else:
            dataframe = None

        data = self._run_auto_mode('split', dataframe, save=self.auto_save_preproc)
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
                raise NotImplementedError("'Not saving preproc results' not yet implemented")
                """
                train_dataframe, test_dataframe = self._single_core(submode, dataframe, save=False)
                self.percentage = None
                return {'train': train_dataframe, 'test': test_dataframe}
                """

        elif submode == 'sample' or submode == 'balancing':
            if submode == 'sample':
                self.percentage = self.sample_ratio
            elif submode == 'balancing':
                self.percentage = self.balancing_ratio
            file_path = self._make_auto_outfile(submode)
            
            if self.nr_cores == 1:
                dataframe_data = self._single_core(submode, dataframe, outfile=file_path, save=save)
            elif self.nr_cores > 1:
                dataframe_data = self._multicore(submode, dataframe, outfile=file_path, save=save)
            else:
                error_message = f"Unsupported value for --nr_cores: {self.nr_cores}"
                raise ValueError(error_message)
            
            self.percentage = None
            return dataframe_data

        else:
            raise ModeError(self.mode, submode)

    def _make_auto_outfile(self, suffix, file=None):
        if file is None:
            file = self.infile

        file_dir = os.path.basename(file)
        file_name, file_extension = os.path.splitext(file_dir)
        return f"{self.src_dir}/data/{self.name}/{file_name}_{suffix}{file_extension}"


# Todo: Make the amount of classes dynamic
def balancing_dataframe(dataframe, percentage=1):
    dataframe_class0 = dataframe[dataframe['class'] == 0]
    dataframe_class1 = dataframe[dataframe['class'] == 1]

    dataframe_class1_sampled = sample_dataframe(dataframe_class1, 1)

    data_div = dataframe_class1_sampled['class'].value_counts()
    nr_samples = int(np.round(data_div[1]*percentage, decimals=0))

    if nr_samples > 1:
        dataframe_class0_balancing = resample(dataframe_class0,
                                              replace=False,
                                              n_samples=nr_samples,
                                              random_state=randrange(100, 999))

        return pd.concat([dataframe_class0_balancing, dataframe_class1_sampled])

    else:
        return pd.DataFrame()


def sample_dataframe(dataframe, percentage=1):
    return dataframe.sample(frac=percentage, random_state=randrange(100, 999), axis=0)


def split_dataframe(dataframe, percentage=None, index=None, axis=0):
    if (percentage and index) or (percentage is None and index is None):
        raise MutuallyExclusiveError('percentage', 'index')
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
        raise MutuallyExclusiveError('percentage', 'index')
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
