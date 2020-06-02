import unittest

import os
import numpy as np
from ..cross_validation import *
from ..constant import *
from pprint import pprint

# todo: set validation value instead of print


class TestCrossValMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.labels = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                       1, 0, 0, 1, 0, 0, 2, 2, 2, 2,
                       2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       3, 3]

        self.labels_sm = [0, 1, 1, 2, 2, 2]
        self.features_sm = [[0, 0, 0],
                            [1, 1, 1],
                            [1, 1, 1],
                            [2, 2, 2],
                            [2, 2, 2],
                            [2, 2, 2]]
        self.patient_num_beats_sm = [1, 2, 3]

    def test_cross_val_index_by_patient(self):
        print("## test1 ##")
        k_folds_indices = cross_val_index_by_patient(self.patient_num_beats_sm)
        print("cross val index by patient: ")
        print([list(i) for i in k_folds_indices])
        print()

    def test_cross_val_index_by_beat(self):
        print("## test2 ##")
        indices_in_cls = []
        for c in range(max(self.labels) + 1):
            indices_in_cls.append([i for i, label in enumerate(self.labels) if label == c])
        # pprint(indices_in_cls)

        k_folds_indices = cross_val_index_by_beat(self.labels, 3)
        print("cross val index by beat: ")
        pprint(k_folds_indices)

        print("\n--small version--\n")

        indices_in_cls = []
        for c in range(max(self.labels_sm) + 1):
            indices_in_cls.append([i for i, label in enumerate(self.labels_sm) if label == c])
        # pprint(indices_in_cls)

        k_folds_indices = cross_val_index_by_beat(self.labels_sm, 3)
        print("cross val index by beat (small): ")
        pprint(k_folds_indices)
        print()

    def test_cv(self):
        print("## test3 ##")
        features = np.array(self.features_sm)
        labels = np.array(self.labels_sm)

        k = 3
        k_folds_indices = cross_val_index_by_beat(labels, k)

        print("test cv with cross_val index by beat: ")
        for kk in range(k):
            print("fold ", kk)
            indices_val = np.array(k_folds_indices[kk])
            indices_trn = np.array(flatten_list([k_folds_indices[i] for i in range(k) if i != kk]))

            print("indices train: ")
            print(indices_trn)
            tr_features = features[indices_trn]
            tr_labels = labels[indices_trn]

            val_features = features[indices_val]
            val_labels = labels[indices_val]

            print("train: ")
            print(tr_features)
            print(tr_labels)
            print("test: ")
            print(val_features)
            print(val_labels)
            print()

        print()


if __name__ == '__main__':
    unittest.main()
