import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc


def calc_class_weights(labels):
    class_weights = {}
    for c in range(max(labels)+1):
        class_weights.update({c: len(labels) / float(np.count_nonzero(labels == c))})
    return class_weights


def flatten_list(a_list):
    return [item for sublist in a_list for item in sublist]


class PrintTime:
    def __init__(self, name, verbose=True):
        """

        How to use:

            with PrintTime("The test"):
                time.sleep(1)

            > "working on The test"
            > "time for The test: 1.00 sec"

        :param name: str
        :param verbose: bool
        """
        self.name = name
        self.start = None
        self.end = None
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            self.start = time.time()
            print("\nworking on {} ...".format(self.name))

    def __exit__(self, type, value, traceback):
        if self.verbose:
            self.end = time.time()
            print("time for {}: {} sec\n".format(self.name, format(self.end - self.start, '.2f')))


def load_pkl_from_storage(path_func, verbose=True):
    def dfunc(make_func):
        def wfunc(*args, **kwargs):
            data_path = path_func(*args, **kwargs)

            if os.path.isfile(data_path):

                if verbose:
                    print("load {} data from {}".format(make_func.__name__, data_path))

                with open(data_path, "rb") as f:
                    gc.disable()  # this improve the required loading time!
                    data = pickle.load(f)
                    gc.enable()
            else:
                data = make_func(*args, **kwargs)

                if verbose:
                    print("write {} data to {}".format(make_func.__name__, data_path))

                with open(data_path, "wb") as f:
                    pickle.dump(data, f, 2)
                    # Protocol version 0 itr_features_balanceds the original ASCII protocol and is backwards compatible with earlier versions of Python.
                    # Protocol version 1 is the old binary format which is also compatible with earlier versions of Python.
                    # Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes.

            return data
        return wfunc
    return dfunc