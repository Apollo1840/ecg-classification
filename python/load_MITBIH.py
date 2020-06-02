#!/usr/bin/env python

"""
Based on

VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import os
import csv
import time
from tqdm import tqdm
import pickle
import gc
import numpy as np
from scipy.signal import medfilt
import operator

from mit_db import mit_db
from constant import *
from path_manager import name_ml_data, name_my_db


def load_mit_db(
        DS,
        ws,
        do_preprocess,
        is_reduce,
        maxRR,
        use_RR,
        norm_RR,
        compute_morph,
        is_save=True,
):
    """
    Load the data with the configuration and features selected

    :param DS: Str, "DS1" or "DS2"
    :param ws: Tuple[int], (winL, winR), window size
    :param do_preprocess: Bool
    :param is_reduce: Bool, use reduced DS or not
    :param maxRR: Bool
    :param use_RR: Bool
    :param norm_RR: Bool
    :param compute_morph: List[str] or Set[str],
        can be ['resample_10', 'raw', 'u-lbp', 'lbp', 'hbf5', 'wvlt', 'wvlt+pca', 'HOS', 'myMorph']
    :param is_save: save loaded as pickle or not
    :return: features, labels, patient_num_beats
    """

    params_for_naming = locals()

    # load directly
    features_labels_name = name_ml_data(**params_for_naming)

    if os.path.isfile(features_labels_name):
        print("Loading pickle: " + features_labels_name + "...")
        with open(features_labels_name, 'rb') as f:
            # disable garbage collector, his improve the required loading time!
            gc.disable()
            features, labels, patient_num_beats = pickle.load(f)
            gc.enable()
        return features, labels, patient_num_beats

    # make one
    # if both lead is usable, then the DS is reduced
    my_db = load_mitbih_db(DS, is_reduce, ws=ws, do_preprocess=do_preprocess)

    # DS for wvlt+pca
    # (winL, winR) for mymorph
    leads_flag = [1, int(is_reduce)] # [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    features = my_db.get_features(leads_flag, maxRR, use_RR, norm_RR, compute_morph, DS, ws)
    labels = my_db.get_labels()
    patient_num_beats = my_db.get_n_beats_per_record()

    # Set labels array!
    if is_save:
        print('writing pickle: ' + features_labels_name + '...')
        with open(features_labels_name, 'wb') as f:
            pickle.dump([features, labels, patient_num_beats], f, 2)

    return features, labels, patient_num_beats


def load_mitbih_db(DS, is_reduce, ws, do_preprocess=False, is_save=True):
    """

    load mitbih db as my_db

    :param DS: Str, "DS1" or "DS2"
    :param ws: Tuple[int], (winL, winR), window size
    :param do_preprocess: Bool
    :param is_reduce: Bool
    :param is_save: Bool
    :return: my_db object
    """

    print("Loading MIT BIH arr (" + DS + ") ...")

    winL, winR = ws
    mit_pickle_name = name_my_db(is_reduce, do_preprocess, winL, winR, DS)

    # If the data with that configuration has been already computed Load pickle
    if os.path.isfile(mit_pickle_name):
        with open(mit_pickle_name, 'rb') as f:
            # disable garbage collector
            gc.disable()  # this improve the required loading time!
            my_db = pickle.load(f)
            gc.enable()
        return my_db
    else:
        return make_mitbih_db(DS, mit_pickle_name, ws, is_reduce, do_preprocess, is_save)


def make_mitbih_db(DS, db_path, ws, is_reduce, do_preprocess=False, is_save=True):
    """

    make my_db based on mitbih

    :param DS: Str, "DS1" or "DS2"
    :param db_path: str
    :param ws: Tuple[int], (winL, winR), window size
    :param do_preprocess: Bool
    :param is_reduce: Bool
    :param is_save: Bool
    :return: my_db object
    """

    bank_key = "reduced" if is_reduce else "normal"
    my_db = load_signals(DS_bank[bank_key][DS], ws, do_preprocess)

    if is_save:
        print("Saving signal processed data ...")
        with open(db_path, 'wb') as f:
            pickle.dump(my_db, f, 2)
            # Protocol version 0 itr_features_balanceds the original ASCII protocol and is backwards compatible with earlier versions of Python.
            # Protocol version 1 is the old binary format which is also compatible with earlier versions of Python.
            # Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes.

    return my_db


def load_signals(record_ids, ws, do_preprocess=True, verbose=False):
    """

    :param record_ids: List[int], record_ids
    :param ws: Tuple[int], (winL, winR), indicates the size of the window centred at R-peak at left and right side
    :param do_preprocess: Bool, indicates if preprocesing of remove baseline on signal is performed
    :param verbose: Bool
    :return: mitdb object.
    """

    list_r_peaks_original = [np.array([]) for _ in range(len(record_ids))]
    list_r_peaks = [np.array([]) for _ in range(len(record_ids))]
    list_is_r_valid = [np.array([]) for _ in range(len(record_ids))]
    # List[int], 1 stands for valid, 0 stands for not valid. valid R will goes into beat and has class_ID
    # a R peak is not valid when its window out of the boundary or the class not belongs to MITBIH

    list_beats = [[] for _ in range(len(record_ids))]  # dim: record, beat, lead, signal
    list_class_id = [[] for _ in range(len(record_ids))]
    list_raw_signal = []

    fRecords, fAnnotations = parse_data_dir(DATA_DIR, record_ids)

    # for r, a in zip(fRecords, fAnnotations):
    # r stands for record_id
    for r in tqdm(range(0, len(fRecords))):
        raw_signal, beats, labels, is_r_valid, r_peaks, r_peaks_original = load_signal_single(fRecords[r],
                                                                                              fAnnotations[r],
                                                                                              ws,
                                                                                              do_preprocess,
                                                                                              verbose)
        list_beats[r] = beats
        list_class_id[r] = labels
        list_is_r_valid[r] = is_r_valid
        list_r_peaks[r] = r_peaks
        list_r_peaks_original[r] = r_peaks_original

        # list_r_peaks[r] = list_r_peaks[r][(list_is_r_valid[r] == 1)]
        # list_r_peaks_original[r] = list_r_peaks_original[r][(list_is_r_valid[r] == 1)]

    # Set the data into a bigger struct that keep all the records!
    my_db = mit_db()
    my_db.filename = fRecords

    my_db.raw_signal = list_raw_signal
    my_db.beat = list_beats  # record, beat, lead, signal
    my_db.class_ID = list_class_id
    my_db.valid_R = list_is_r_valid
    my_db.R_pos = list_r_peaks
    my_db.orig_R_pos = list_r_peaks_original

    return my_db


def load_signal_single(f_record, f_annotation, ws, do_preprocess, verbose=False):
    """

    :param f_record: str
    :param f_annotation: str
    :param ws: Tuple[int], (winL, winR)
    :param do_preprocess: Bool
    :param verbose: Bool
    :return:
    """

    # print("Processing signal " + str(r) + " / " + str(len(fRecords)) + "...")

    filename = os.path.join(DATA_DIR, f_record)
    MLII, V1 = load_ecg_from_csv(filename)

    raw_signal = (MLII, V1)

    # 2. Preprocessing signal! (very time consuming)
    if do_preprocess:
        MLII = preprocess_sig(MLII, verbose)
        V1 = preprocess_sig(V1, verbose)

    # 3. Read annotations
    filename = os.path.join(DATA_DIR, f_annotation)
    annotations = load_ann_from_txt(filename)

    # Extract the R-peaks from annotations
    beat_indices, labels, r_peaks, r_peaks_original, is_r_valid = parse_annotations(
        annotations,
        MLII,
        ws,
        size_rr_max=20)

    beats = [(MLII[beat_start: beat_end], V1[beat_start: beat_end]) for beat_start, _, beat_end in beat_indices]

    return raw_signal, beats, labels, np.array(is_r_valid), np.array(r_peaks), np.array(r_peaks_original)


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
    lst.sort()  # very important to align the records
    for filename in lst:
        if filename.endswith(".csv"):
            if int(filename[0:3]) in record_ids:
                fRecords.append(filename)
        elif filename.endswith(".txt"):
            if int(filename[0:3]) in record_ids:
                fAnnotations.append(filename)

    assert len(fRecords) == len(fAnnotations)
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

    :param annotations: List[str], the str.split()[1] is r_pos, str.split()[2] is beat_label in mitbih format
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

        # todo: varify r_pos and beat_label

        beat_index, label = parse_beats(int(r_pos), beat_label, ref_sig, ws, rr_max=size_rr_max)

        if label:
            labels.append(AAMI_CLASSES.index(label))
            beat_indices.append(beat_index)

            is_r_valid.append(1)
        else:
            is_r_valid.append(0)

        r_peaks_original.append(int(r_pos))
        r_peaks.append(beat_index[1])  # r_pos might be changed after parse_beats

    # definitly:
    # assert len(r_peaks) == len(r_peaks_original) == len(is_r_valid)

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


def preprocess_sig(signal, verbose=False):
    """

    :param signal: List[float]
    :param verbose: Bool
    :return: List[float]
    """
    if verbose:
        print("filtering the signal ...")

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
