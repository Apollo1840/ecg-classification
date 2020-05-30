#!/usr/bin/env python

"""
load_MITBIH.py

Download .csv files and annotations from:
    kaggle.com/mondejar/mitbih-database

VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import os
import csv
import gc
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.stats
import pywt
import time
import operator
from tqdm import tqdm

from mit_db import mit_db, RR_intervals
from features_ECG import *

from numpy.polynomial.hermite import hermfit, hermval

DATA_DIR = '/home/congyu/dataset/ECG/mitbihcsv/'
MITBIH_CLASSES = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']  # , 'P', '/', 'f', 'u']
AAMI = {
    "N": ['N', 'L', 'R'],
    "SVEB": ['A', 'a', 'J', 'S', 'e', 'j'],
    "VEB": ['V', 'E'],
    "F": ['F'],
    # "Q": ['P', '/', 'f', 'u'],
}

DS_bank = {
    "normal": {
        "DS1": [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                223, 230],
        "DS2": [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                233, 234]
    },
    "reduced": {
        "DS1": [101, 106, 108, 109, 112, 115, 118, 119, 201, 203,
                205, 207, 208, 209, 215, 220, 223, 230],
        "DS2": [105, 111, 113, 121, 200, 202, 210, 212, 213, 214,
                219, 221, 222, 228, 231, 232, 233, 234]
    }
}

AAMI_CLASSES = sorted(AAMI.keys())


def load_mit_db(
        DS,
        winL,
        winR,
        do_preprocess,
        maxRR,
        use_RR,
        norm_RR,
        compute_morph,
        db_path,
        reduced_DS,
        leads_flag,
        is_save=True,
):
    """
    Load the data with the configuration and features selected

    :param DS: Str, "DS1" or "DS2"
    :param winL: int
    :param winR: int
    :param do_preprocess: Bool
    :param maxRR: Bool
    :param use_RR: Bool
    :param norm_RR: Bool
    :param compute_morph: List[str] or Set[str],
        can be ['resample_10', 'raw', 'u-lbp', 'lbp', 'hbf5', 'wvlt', 'wvlt+pca', 'HOS', 'myMorph']
    :param db_path: str
    :param reduced_DS: Bool,
            load DS1, DS2 patients division (Chazal) or reduced version,
            i.e., only patients in common that contains both MLII and V1
    :param leads_flag: [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :param is_save: save loaded as pickle or not
    :return: features, labels, patient_num_beats
    """

    features_labels_name = name_ml_data(DS,
                                        winL,
                                        winR,
                                        do_preprocess,
                                        maxRR,
                                        use_RR, norm_RR,
                                        compute_morph,
                                        db_path,
                                        reduced_DS,
                                        leads_flag)

    if os.path.isfile(features_labels_name):
        print("Loading pickle: " + features_labels_name + "...")
        f = open(features_labels_name, 'rb')
        # disable garbage collector
        gc.disable()  # this improve the required loading time!
        features, labels, patient_num_beats = pickle.load(f)
        gc.enable()
        f.close()

    else:
        my_db = load_mitbih_db(DS, db_path, (winL, winR), reduced_DS, do_preprocess, False)

        # first labels
        labels = np.array(sum(my_db.class_ID, [])).flatten()

        # then features
        features = np.array([], dtype=float)

        # before actual get the features, there are something to prepare:
        # prepare for use_RR and norm_RR, if it is needed
        RR = []
        if use_RR or norm_RR:
            if maxRR:
                r_poses = my_db.R_pos
            else:
                r_poses = my_db.orig_R_pos
            RR = calc_RR_intervals(r_poses, my_db.valid_R)

        #########################################################################
        # Compute RR intervals features

        if use_RR:
            features_rr = get_features_rr(RR)
            features = np.column_stack((features, features_rr)) if features.size else features_rr

        if norm_RR:
            features_rr_norm = get_features_rr_norm(RR)
            features = np.column_stack((features, features_rr_norm)) if features.size else features_rr_norm

        ##########################################################################
        # Compute morphological features

        if 'raw' in compute_morph:
            print("Raw ...")
            start = time.time()

            features_raw = get_features_raw(my_db.beat, leads_flag)
            features = np.column_stack((features, features_raw)) if features.size else features_raw

            end = time.time()
            print("Time raw: " + str(format(end - start, '.2f')) + " sec")

        if 'resample_10' in compute_morph:
            print("Resample_10 ...")
            start = time.time()

            features_raw = get_features_resample_10(my_db.beat, leads_flag)
            features = np.column_stack((features, features_raw)) if features.size else features_raw

            end = time.time()
            print("Time resample: " + str(format(end - start, '.2f')) + " sec")

        if 'u-lbp' in compute_morph:
            print("u-lbp ...")

            features_u_lbp = get_features_u_lbp(my_db.beat, leads_flag)
            features = np.column_stack((features, features_u_lbp)) if features.size else features_u_lbp

            print(features.shape)

        if 'lbp' in compute_morph:
            print("lbp ...")

            features_lbp = get_features_lbp(my_db.beat, leads_flag)
            features = np.column_stack((features, features_lbp)) if features.size else features_lbp

            print(features.shape)

        if 'hbf5' in compute_morph:
            print("hbf ...")

            features_temp = get_features_hbf5(my_db.beat, leads_flag)
            features = np.column_stack((features, features_temp)) if features.size else features_temp

            print(features.shape)

        # Wavelets
        if 'wvlt' in compute_morph:
            print("Wavelets ...")

            features_temp = get_features_wvlt(my_db.beat, leads_flag)
            features = np.column_stack((features, features_temp)) if features.size else features_temp

        # Wavelets
        if 'wvlt+pca' in compute_morph:
            features_temp = get_features_wvlt_pca(my_db.beat, leads_flag, DS, family="db1", level=3, pca_k=7)
            features = np.column_stack((features, features_temp)) if features.size else features_temp

        # HOS
        if 'HOS' in compute_morph:
            print("HOS ...")
            features_temp = get_featurs_hos(my_db.beat, leads_flag)
            features = np.column_stack((features, features_temp)) if features.size else features_temp
            print(features.shape)

        # My morphological descriptor
        if 'myMorph' in compute_morph:
            print("My Descriptor ...")
            features_temp = get_features_mymorph(my_db.beat, leads_flag, winL, winR)
            features = np.column_stack((features, features_temp)) if features.size else features_temp

        # This array contains the number of beats for each patient (for cross_val)
        patient_num_beats = np.array([], dtype=np.int32)
        for p in range(len(my_db.beat)):
            patient_num_beats = np.append(patient_num_beats, len(my_db.beat[p]))

        # Set labels array!
        print('writing pickle: ' + features_labels_name + '...')
        if not os.path.exists(os.path.dirname(features_labels_name)):
            os.makedirs(os.path.dirname(features_labels_name))

        with open(features_labels_name, 'wb') as f:
            pickle.dump([features, labels, patient_num_beats], f, 2)

    return features, labels, patient_num_beats


def load_mitbih_db(DS, db_path, ws, is_reduce, do_preprocess=False, is_save=True):
    """

    load mitbih db as my_db

    :param DS: Str, "DS1" or "DS2"
    :param db_path: str
    :param ws: Tuple[int], (winL, winR), window size
    :param do_preprocess: Bool
    :param is_reduce: Bool
    :param is_save: Bool
    :return: my_db object
    """

    print("Loading MIT BIH arr (" + DS + ") ...")

    # ML-II + V1
    if is_reduce:
        DS1 = DS_bank["reduced"]["DS1"]
        DS2 = DS_bank["reduced"]["DS2"]
    # ML-II
    else:
        DS1 = DS_bank["normal"]["DS1"]
        DS2 = DS_bank["normal"]["DS2"]

    winL, winR = ws
    mit_pickle_name = name_my_db(db_path, is_reduce, do_preprocess, winL, winR, DS)

    # If the data with that configuration has been already computed Load pickle
    if os.path.isfile(mit_pickle_name):
        with open(mit_pickle_name, 'rb') as f:
            # disable garbage collector
            gc.disable()  # this improve the required loading time!
            my_db = pickle.load(f)
            gc.enable()

    else:  # Load data and compute de RR features
        if DS == 'DS1':
            my_db = load_signal(DS1, ws, do_preprocess)
        else:
            my_db = load_signal(DS2, ws, do_preprocess)

        if is_save:
            print("Saving signal processed data ...")
            with open(mit_pickle_name, 'wb') as f:
                pickle.dump(my_db, f, 2)
                # Protocol version 0 itr_features_balanceds the original ASCII protocol and is backwards compatible with earlier versions of Python.
                # Protocol version 1 is the old binary format which is also compatible with earlier versions of Python.
                # Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes.

    return my_db


def name_my_db(db_path, reduced_DS, do_preprocess, winL, winR, DS):
    """

    :param db_path:
    :param reduced_DS:
    :param do_preprocess:
    :param winL:
    :param winR:
    :param DS:
    :return: str
    """

    mit_pickle_name = db_path + 'python_mit'
    if reduced_DS:
        mit_pickle_name = mit_pickle_name + '_reduced_'

    if do_preprocess:
        mit_pickle_name = mit_pickle_name + '_rm_bsline'

    mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR)
    mit_pickle_name = mit_pickle_name + '_' + DS + '.pkl'

    return mit_pickle_name


def name_ml_data(DS, winL, winR, do_preprocess, maxRR, use_RR, norm_RR,
                 compute_morph, db_path, reduced_DS, leads_flag):
    """


    :param DS:
    :param winL:
    :param winR:
    :param do_preprocess:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param db_path: str
    :param reduced_DS:
    :param leads_flag:
    :return: Str, eg. "..._v1.p"
    """

    features_labels_name = db_path + 'features/' + 'w_' + str(winL) + '_' + str(winR) + '_' + DS

    if do_preprocess:
        features_labels_name += '_rm_bsline'

    if maxRR:
        features_labels_name += '_maxRR'

    if use_RR:
        features_labels_name += '_RR'

    if norm_RR:
        features_labels_name += '_norm_RR'

    for descp in compute_morph:
        features_labels_name += '_' + descp

    if reduced_DS:
        features_labels_name += '_reduced'

    if leads_flag[0] == 1:
        features_labels_name += '_MLII'

    if leads_flag[1] == 1:
        features_labels_name += '_V1'

    features_labels_name += '.pkl'

    return features_labels_name


def load_signal(DS, ws, do_preprocess=True):
    """

    :param DS: List[int], record_ids
    :param ws: Tuple[int], (winL, winR), indicates the size of the window centred at R-peak at left and right side
    :param do_preprocess: Bool, indicates if preprocesing of remove baseline on signal is performed
    :return: mitdb object.
    """
    size_RR_max = 20

    Original_R_poses = [np.array([]) for _ in range(len(DS))]
    R_poses = [np.array([]) for _ in range(len(DS))]
    valid_R = [np.array([]) for _ in range(len(DS))]
    # List[int], 1 stands for valid, 0 stands for not valid. valid R will goes into beat and has class_ID
    # a R peak is not valid when its window out of the boundary or the class not belongs to MITBIH

    beat = [[] for _ in range(len(DS))]  # dim: record, beat, lead
    class_ID = [[] for _ in range(len(DS))]

    fRecords, fAnnotations = parse_data_dir(DATA_DIR, DS)

    RAW_signals = []

    # for r, a in zip(fRecords, fAnnotations):
    for r in tqdm(range(0, len(fRecords))):

        # print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

        # 1. Read signalR_poses
        filename = os.path.join(DATA_DIR, fRecords[r])
        MLII, V1 = load_ecg_from_csv(filename)

        RAW_signals.append((MLII, V1))  # NOTE a copy must be created in order to preserve the original signal
        # display_signal(MLII)

        # 2. Preprocessing signal! (very time consuming)
        if do_preprocess:
            MLII = preprocess_sig(MLII)
            V1 = preprocess_sig(V1)

        # 3. Read annotations
        filename = os.path.join(DATA_DIR, fAnnotations[r])
        annotations = load_ann_from_txt(filename)

        # Extract the R-peaks from annotations
        beat_indices, labels, r_peaks, r_peaks_original, is_r_valid = parse_annotations(
            annotations,
            MLII,
            ws,
            size_RR_max)

        beat[r] = [(MLII[beat_start: beat_end], V1[beat_start: beat_end]) for beat_start, _, beat_end in beat_indices]
        class_ID[r] = labels
        valid_R[r] = np.array(is_r_valid)
        R_poses[r] = np.array(r_peaks)
        Original_R_poses[r] = np.array(r_peaks_original)

        # R_poses[r] = R_poses[r][(valid_R[r] == 1)]
        # Original_R_poses[r] = Original_R_poses[r][(valid_R[r] == 1)]

    # Set the data into a bigger struct that keep all the records!
    my_db = mit_db()
    my_db.filename = fRecords

    my_db.raw_signal = RAW_signals
    my_db.beat = beat  # record, beat, lead
    my_db.class_ID = class_ID
    my_db.valid_R = valid_R
    my_db.R_pos = R_poses
    my_db.orig_R_pos = Original_R_poses

    return my_db


def parse_data_dir(data_dir, record_ids):
    """
    return one is a list of ".csv" file, one is a list of ".txt" file
    under the data_dir, whose name has ids in ds

    :param data_dir:
    :param record_ids: List[int]
    :return: Tuple(List[str]), two list,
       one is a list of ".csv" file, one is a list of ".txt" file
    """
    # Read files: signal (.csv )  annotations (.txt)
    fRecords = list()
    fAnnotations = list()

    lst = os.listdir(data_dir)
    lst.sort()
    for filename in lst:
        if filename.endswith(".csv"):
            if int(filename[0:3]) in record_ids:
                fRecords.append(filename)
        elif filename.endswith(".txt"):
            if int(filename[0:3]) in record_ids:
                fAnnotations.append(filename)

    return fRecords, fAnnotations


def load_ecg_from_csv(filename):
    """

    :param filename: str
    :return:
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        next(reader)  # skip first line!
        MLII_index = 1
        V1_index = 2

        ptid = os.path.split(filename)[-1][0:3]
        if int(ptid) == 114:
            MLII_index = 2
            V1_index = 1

        MLII = []
        V1 = []
        for row in reader:
            MLII.append((int(row[MLII_index])))
            V1.append((int(row[V1_index])))

    return MLII, V1


def load_ann_from_txt(filename):
    """

    :param filename: str
    :return:
    """

    with open(filename, "r") as f:
        next(f)  # skip first line!
        annotations = []
        for line in f:
            annotations.append(line)
    return annotations


def parse_annotations(annotations, ref_sig, ws=(90, 90), size_rr_max=20):
    """

    :param annotations:
    :param ref_sig: reference_signal
    :param ws:
    :param size_rr_max:
    :return:
        beat_indices, List[Tuple], Tuple is (beat_start_index, r_peak_index, beat_end_index).
        labels, List[int],
    """

    beat_indices = []
    labels = []
    r_peaks = []
    r_peaks_original = []
    is_r_valid = []

    for a in annotations:
        _, r_pos, beat_label = a.split()[:3]

        r_pos = int(r_pos)
        r_peaks_original.append(r_pos)

        beat_index, label = parse_beats(r_pos, beat_label, ref_sig, ws, rr_max=size_rr_max)

        if label:
            labels.append(AAMI_CLASSES.index(label))
            beat_indices.append(beat_index)
            is_r_valid.append(1)
        else:
            is_r_valid.append(0)

        r_peaks.append(beat_index[1])  # r_pos might be changed after parse_beats

    assert len(r_peaks) == len(r_peaks_original) == len(is_r_valid)

    return beat_indices, labels, r_peaks, r_peaks_original, is_r_valid


def parse_beats(r_pos, beat_type, ref_sig, ref_ws, is_relocate=True, rr_max=20):
    """
    relocate r_peak, get beat index and AAMI class

    :param r_pos: int, r-peak index
    :param beat_type: str, mitbih label.
    :param ref_sig: List[float], reference_signal
    :param ref_ws: Tuple[int], (winL, winR), reference window size
    :param is_relocate: Bool
    :param rr_max: int, related to sample rate
    :return: Tuple, str, str is AAMI label.
    """

    beatL = None
    beatR = None
    class_AAMI = None

    winL, winR = ref_ws

    if is_relocate:
        r_pos = relocate_r_peak(r_pos, ref_sig, rr_max)

    if winL < r_pos < (len(ref_sig) - winR) and beat_type in MITBIH_CLASSES:
        beatL = r_pos - winL
        beatR = r_pos + winR
        class_AAMI = mitbih2aami(beat_type)

    return (beatL, r_pos, beatR), class_AAMI


def relocate_r_peak(r_pos, ref_signal, rr_max):
    """

    :param r_pos: int, r_peak index
    :param ref_signal: List[float], reference signal
    :param rr_max: int,
    :return:
    """

    # relocate r_peak by searching maximum in ref_signal
    # r_pos between [size_RR_max, len(MLII) - size_RR_max]
    if rr_max < r_pos < len(ref_signal) - rr_max:
        index, value = max(enumerate(ref_signal[r_pos - rr_max: r_pos + rr_max]), key=operator.itemgetter(1))
        return (r_pos - rr_max) + index
    return r_pos


def preprocess_sig(signal):
    """

    :param signal: List[float]
    :return: List[float]
    """

    baseline = medfilt(signal, 71)
    baseline = medfilt(baseline, 215)

    # Remove Baseline
    for i in range(0, len(signal)):
        signal[i] = signal[i] - baseline[i]

    # TODO Remove High Freqs
    return signal


def mitbih2aami(label):
    """
    
    :param label: str 
    :return: str 
    """

    for aami_label, mitbih_label in AAMI.items():
        if label in mitbih_label:
            return aami_label
    return None


if __name__ == "__main__":
    # test level 0
    load_mit_db(
        DS="DS1",
        winL=90,
        winR=90,
        do_preprocess=False,
        maxRR=True,
        use_RR=False,
        norm_RR=False,
        compute_morph=[],
        db_path="/home/congyu/dataset/ECG/cache/",
        reduced_DS=True,
        leads_flag=[1, 0],
        is_save=False)

    # test level 1
    load_mit_db(
        DS="DS1",
        winL=90,
        winR=90,
        do_preprocess=False,
        maxRR=True,
        use_RR=True,
        norm_RR=True,
        compute_morph=[],
        db_path="/home/congyu/dataset/ECG/cache/",
        reduced_DS=True,
        leads_flag=[1, 0],
        is_save=False)

    # test level 2
    features, labels, _ = load_mit_db(
        DS="DS1",
        winL=90,
        winR=90,
        do_preprocess=False,
        maxRR=False,
        use_RR=False,
        norm_RR=False,
        compute_morph=['resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'],
        db_path="/home/congyu/dataset/ECG/cache/",
        reduced_DS=True,
        leads_flag=[1, 0],
        is_save=False)

    print(features.shape)
    print(labels[0])
