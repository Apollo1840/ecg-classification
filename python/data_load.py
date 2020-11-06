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
import json
from collections import Counter

import operator
import matplotlib.pyplot as plt

from data_base import mit_db
from config import *
from utils import load_pkl_from_storage
from utils_ecg import *
from util_path_manager import path_to_ml_data, path_to_my_db


@load_pkl_from_storage(path_func=path_to_ml_data)
def load_features_from_mitdb(
        DS: str,
        ws,
        do_preprocess: bool,
        is_reduce: bool,
        maxRR: bool,
        use_RR: bool,
        norm_RR: bool,
        compute_morph,
        verbose=False,
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
    :return: features, labels, patient_num_beats
    """

    if verbose:
        print("loading my_db from mitbih for {} ...".format(DS))

    my_db = mitbih_db(DS, is_reduce, ws=ws, do_preprocess=do_preprocess, verbose=verbose)

    if verbose:
        introduce_mtibih_db(my_db)
    	print("amplitude range of first beat: {} to {}".format(np.min(my_db.beat[0][0]), np.max(my_db.beat[0][0])))
    	plt.plot(my_db.beat[0][0][0])
    	plt.show()

    leads_flag = [1, int(is_reduce)]  # [MLII, V1] set the value to 0 or 1 to reference if that lead is used

    if verbose:
        print("creating features of {} with leads_flag as {}, including: \n{}".format(DS, leads_flag, compute_morph))

    features = my_db.get_features(leads_flag, maxRR, use_RR, norm_RR, compute_morph, DS, ws)
    labels = my_db.get_labels()
    patient_num_beats = my_db.get_n_beats_per_record()

    return features, labels, patient_num_beats


@load_pkl_from_storage(path_func=path_to_my_db)
def mitbih_db(DS, is_reduce, ws, do_preprocess=False, verbose=False):
    """

    load mitbih db as my_db

    :param DS: Str, "DS1" or "DS2"
    :param ws: Tuple[int], (winL, winR), indicates the size of the window centred at R-peak at left and right side
    :param do_preprocess: Bool, indicates if preprocesing of remove baseline on signal is performed
    :param is_reduce: Bool
    :return: my_db object
    """

    bank_key = "reduced" if is_reduce else "normal"
    record_ids = DS_bank[bank_key][DS]
    fRecords, fAnnotations = parse_data_dir(DATA_DIR, record_ids)

    list_r_peaks_original = [np.array([]) for _ in range(len(record_ids))]
    list_r_peaks = [np.array([]) for _ in range(len(record_ids))]
    list_is_r_valid = [np.array([]) for _ in range(len(record_ids))]
    # List[int], 1 stands for valid, 0 stands for not valid. valid R will goes into beat and has class_ID
    # a R peak is not valid when its window out of the boundary or the class not belongs to MITBIH

    list_beats = [[] for _ in range(len(record_ids))]  # dim: record, beat, lead, signal
    list_class_id = [[] for _ in range(len(record_ids))]
    list_raw_signal = []

    # for r, a in zip(fRecords, fAnnotations):
    # r stands for record_id
    for r in tqdm(range(0, len(fRecords))):
        raw_signal, beats, labels, is_r_valid, r_peaks, r_peaks_original = load_signal(fRecords[r],
                                                                                       fAnnotations[r],
                                                                                       ws,
                                                                                       do_preprocess,
                                                                                       verbose)
        list_raw_signal[r] = raw_signal
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

def introduce_mtibih_db(my_db):
    for i, fRecord in enumerate(my_db.filename):
        print("record {} get {} beats, lost {} beats".format(fRecord,
                                                             len(my_db.beat[i]),
                                                             len(my_db.R_pos[i]) - len(my_db.beat[i])))

    n_beats = sum([len(beats) for beats in my_db.beat])
    n_r_peaks = sum([len(r_peaks) for r_peaks in my_db.R_pos])
    print("== get {} beats, lost {} beats ==".format(n_beats, n_r_peaks - n_beats))


def load_signal(f_record, f_annotation, ws, do_preprocess, verbose=False):
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
    beat_indices, labels, r_peaks_original, r_peaks, is_r_valid, invalid_reasons = parse_annotations(
        annotations,
        MLII,
        ws,
        size_rr_max=20,
        verbose=verbose)

    if verbose:
        print("in record {}, lost {} beats".format(f_record, len(r_peaks) - len(beat_indices)))
        print(Counter(invalid_reasons))

    beats = [(MLII[beat_start: beat_end], V1[beat_start: beat_end]) for beat_start, _, beat_end in beat_indices]
    labels = [AAMI_CLASSES.index(label) for label in labels]

    if verbose:
        print("number of annotations: {}".format(len(r_peaks)))
        print("number of beats: {}".format(len(beats)))

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
    todo: load ecg from record id.

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
            _, r_pos, beat_label = line.split()[:3]

            if len(beat_label) == 1:
                annotations.append((int(r_pos), beat_label))

    return annotations


if __name__ == "__main__":
    # test level 0
    load_features_from_mitdb(
        DS="DS1",
        ws=[90, 90],
        do_preprocess=False,
        maxRR=True,
        use_RR=False,
        norm_RR=False,
        compute_morph=[],
        is_reduce=True,
        is_save=False)

    # test level 1
    load_features_from_mitdb(
        DS="DS1",
        ws=[90, 90],
        do_preprocess=False,
        maxRR=True,
        use_RR=True,
        norm_RR=True,
        compute_morph=[],
        is_reduce=True,
        is_save=False)

    # test level 2
    features, labels, _ = load_features_from_mitdb(
        DS="DS1",
        ws=[90, 90],
        do_preprocess=False,
        maxRR=False,
        use_RR=False,
        norm_RR=False,
        compute_morph=['resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'],
        is_reduce=True,
        is_save=False)

    print(features.shape)
    print(labels[0])
