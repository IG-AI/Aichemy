import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.cheminf.utils import read_dataframe, save_dataframe


class ChemInfPostProc(object):
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
