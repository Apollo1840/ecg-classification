import time
import numpy as np
import matplotlib.pyplot as plt


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