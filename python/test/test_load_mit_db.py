import unittest

import os
import numpy as np
from ..load_MITBIH import *
from ..constant import *


class TestECGMethods(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_load_signal(self):
        my_db = load_signal(record_ids=DS_bank["reduced"]["DS1"][:1], ws=(90, 90), do_preprocess=True)
        print(len(my_db.beat))  # len of records 1
        print(len(my_db.beat[0]))  # number of beats
        print(len(my_db.beat[0][0]))  # number of leads  2
        print(len(my_db.beat[0][0][0]))  # len of a beat  180

        # TODO: use assert to varify

    def test_labels(self):
        # labels = np.array(sum(my_db.class_ID, [])).flatten()
        pass

    def test_parse_annotations(self):
        fRecords, fAnnotations = parse_data_dir(DATA_DIR, DS_bank["reduced"]["DS1"])
        filename = os.path.join(DATA_DIR, fRecords[0])
        MLII, V1 = load_ecg_from_csv(filename)
        filename = os.path.join(DATA_DIR, fAnnotations[0])
        annotations = load_ann_from_txt(filename)
        beat_indices, labels, r_peaks, r_peaks_original, is_r_valid = parse_annotations(
            annotations,
            MLII,
            (90, 90),
            size_rr_max=20)

        print(len(r_peaks))
        print(len(beat_indices))
        print(len(beat_indices[0]))  # 3

        # todo: plot the beat


if __name__ == '__main__':
    unittest.main()
