#!/usr/bin/env python

"""
mit_db.py

Description:
Contains the classes for store the MITBIH database and some utils

VARPA, University of Coruna
Mondejar Guerra, Victor M.
24 Oct 2017
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from utils import PrintTime

from feature_extraction import (calc_RR_intervals,
                                get_features_resample, get_features_rr, get_featurs_hos, get_features_lbp,
                                get_features_raw, get_features_wvlt, get_features_hbf5, get_features_rr_norm,
                                get_features_u_lbp, get_features_mymorph, get_features_wvlt_pca)


class mit_db:
    def __init__(self):
        # Instance atributes
        self.filename = []
        self.raw_signal = []
        self.beat = np.empty([])  # dim: record, beat, lead, signal; num_record, num_beats, num_lead, amplitude
        self.class_ID = []
        self.valid_R = []
        self.R_pos = []
        self.orig_R_pos = []

        self.n_record = len(self.class_ID)

    @property
    def n_beats_per_record(self):
        return [len(beats) for beats in self.beat]

    @property
    def n_beats(self):
        return sum(self.n_beats_per_record)

    def get_features(self, leads_flag, maxRR, use_RR, norm_RR, compute_morph, DS, ws):
        """

        :param leads_flag: [MLII, V1] set the value to 0 or 1 to reference if that lead is used
        :param maxRR: Bool, relocate r-peak or not
        :param use_RR: Bool
        :param norm_RR: Bool
        :param compute_morph: List[str] or Set[str],
                can be ['resample_10', 'raw', 'u-lbp', 'lbp', 'hbf5', 'wvlt', 'wvlt+pca', 'HOS', 'myMorph']
        :param DS: Str, "DS1" or "DS2"
        :param ws: Tuple[int], window size: [winL, winR]
        :return:
        """

        features = np.array([], dtype=float)

        # before actual get the features, there are something to prepare:
        # prepare for use_RR and norm_RR, if it is needed
        RR = []
        if use_RR or norm_RR:
            print("getting rr features ...")
            if maxRR:
                r_poses = self.R_pos
            else:
                r_poses = self.orig_R_pos
            RR = calc_RR_intervals(r_poses, self.valid_R)
            # RR is a list of RR object
            # each RR object represent RR features of one record. it has 4 lists as attributes.

        #########################################################################
        # Compute RR intervals features

        if use_RR:
            with PrintTime("rr"):
                features_rr = get_features_rr(RR)
                features = np.column_stack((features, features_rr)) if features.size else features_rr

        if norm_RR:
            with PrintTime("norm_rr"):
                features_rr_norm = get_features_rr_norm(RR)
                features = np.column_stack((features, features_rr_norm)) if features.size else features_rr_norm

        print(features.shape)
        ##########################################################################
        # Compute morphological features

        if 'raw' in compute_morph:
            with PrintTime("raw"):
                features_raw = get_features_raw(self.beat, leads_flag)
                features = np.column_stack((features, features_raw)) if features.size else features_raw
                print(features.shape)

        if 'resample_10' in compute_morph:
            with PrintTime("resample_10"):
                features_resample = get_features_resample(self.beat, leads_flag)
                features = np.column_stack((features, features_raw)) if features.size else features_resample
                print(features.shape)

        if 'u-lbp' in compute_morph:
            with PrintTime("u-lbp"):
                features_u_lbp = get_features_u_lbp(self.beat, leads_flag)
                features = np.column_stack((features, features_u_lbp)) if features.size else features_u_lbp
                print(features.shape)

        if 'lbp' in compute_morph:
            with PrintTime("lbp"):
                features_lbp = get_features_lbp(self.beat, leads_flag)
                features = np.column_stack((features, features_lbp)) if features.size else features_lbp
                print(features.shape)

        if 'hbf5' in compute_morph:
            with PrintTime("hbf5"):
                features_temp = get_features_hbf5(self.beat, leads_flag)
                features = np.column_stack((features, features_temp)) if features.size else features_temp
                print(features.shape)

        # Wavelets
        if 'wvlt' in compute_morph:
            with PrintTime("wvlt"):
                features_temp = get_features_wvlt(self.beat, leads_flag)
                features = np.column_stack((features, features_temp)) if features.size else features_temp
                print(features.shape)

        # Wavelets
        if 'wvlt+pca' in compute_morph:
            with PrintTime("wvlt+pca"):
                features_temp = get_features_wvlt_pca(self.beat, leads_flag, DS, family="db1", level=3, pca_k=7)
                features = np.column_stack((features, features_temp)) if features.size else features_temp
                print(features.shape)

        # HOS
        if 'HOS' in compute_morph:
            with PrintTime("HOS"):
                features_temp = get_featurs_hos(self.beat, leads_flag)
                features = np.column_stack((features, features_temp)) if features.size else features_temp
                print(features.shape)

        # My morphological descriptor
        if 'MyMorph' in compute_morph:
            with PrintTime("MyMorph"):
                features_temp = get_features_mymorph(self.beat, leads_flag, ws[0], ws[1])
                features = np.column_stack((features, features_temp)) if features.size else features_temp
                print(features.shape)

        return features

    def get_labels(self):
        return np.array(sum(self.class_ID, [])).flatten()

    def get_n_beats_per_record(self):
        patient_num_beats = np.array([], dtype=np.int32)
        for p in range(len(self.beat)):
            patient_num_beats = np.append(patient_num_beats, len(self.beat[p]))
        return patient_num_beats
