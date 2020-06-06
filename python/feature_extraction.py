#!/usr/bin/env python

"""
features_ECG.py
    
VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import numpy as np
from numpy.polynomial.hermite import hermfit
from scipy.signal import medfilt
import scipy.stats
import pywt
import operator
from itertools import chain
import gc
import pickle
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA, IncrementalPCA


# Class for RR intervals features
class RR_intervals:
    def __init__(self):
        # Instance atributes

        # list of pre_RR length
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])


uniform_pattern_list = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,
     128,
     129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
     251, 252, 253, 254, 255])


def compute_RR_intervals(R_poses):
    """
    Input: the R-peaks from a signal
    Return: the features RR intervals
        (pre_RR, post_RR, local_RR, global_RR)
    for each beat

    Pre_RR: int, distance between actual and previous R peak
    post_RR: int, distance between actual and next R peak
    Local_RR: AVG from last 10 pre_R values
    global_RR: AVG from past 5 minutes = 108000 samples
    :param R_poses:
    :return: RR_intervals object
    """

    features_RR = RR_intervals()

    pre_R = np.array([], dtype=int)
    post_R = np.array([], dtype=int)
    local_R = np.array([], dtype=int)
    global_R = np.array([], dtype=int)

    # Pre_R and Post_R
    pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, R_poses[1] - R_poses[0])

    for i in range(1, len(R_poses) - 1):
        pre_R = np.append(pre_R, R_poses[i] - R_poses[i - 1])
        post_R = np.append(post_R, R_poses[i + 1] - R_poses[i])

    pre_R[0] = pre_R[1]
    pre_R = np.append(pre_R, R_poses[-1] - R_poses[-2])

    post_R = np.append(post_R, post_R[-1])

    # Local_RR: AVG from last 10 pre_R values
    for i in range(0, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j + i >= 0:
                avg_val = avg_val + pre_R[i + j]
                num = num + 1
        local_R = np.append(local_R, avg_val / float(num))

    # Global R : AVG from past 5 minutes = 108000 samples
    global_R = np.append(global_R, pre_R[0])
    for i in range(1, len(R_poses)):
        num = 0
        avg_val = 0

        for j in range(0, i):
            if (R_poses[i] - R_poses[j]) < 108000:
                avg_val = avg_val + pre_R[j]
                num = num + 1
        # num = i
        global_R = np.append(global_R, avg_val / float(num))

    # why not directly assign features_RR.pre_R = pre_R
    for i in range(0, len(R_poses)):
        features_RR.pre_R = np.append(features_RR.pre_R, pre_R[i])
        features_RR.post_R = np.append(features_RR.post_R, post_R[i])
        features_RR.local_R = np.append(features_RR.local_R, local_R[i])
        features_RR.global_R = np.append(features_RR.global_R, global_R[i])

        # features_RR.append([pre_R[i], post_R[i], local_R[i], global_R[i]])

    return features_RR


def save_wvlt_PCA(PCA, pca_k, family, level):
    f = open('Wvlt_' + family + '_' + str(level) + '_PCA_' + str(pca_k) + '.p', 'wb')
    pickle.dump(PCA, f, 2)
    f.close()


def load_wvlt_PCA(pca_k, family, level):
    f = open('Wvlt_' + family + '_' + str(level) + '_PCA_' + str(pca_k) + '.p', 'rb')
    # disable garbage collector
    gc.disable()  # this improve the required loading time!
    PCA = pickle.load(f)
    gc.enable()
    f.close()

    return PCA


# Compute the wavelet descriptor for a beat
def compute_wavelet_descriptor(beat, family, level):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]


# Compute my descriptor based on amplitudes of several intervals
def compute_my_own_descriptor(beat, winL, winR):
    R_pos = int((winL + winR) / 2)

    R_value = beat[R_pos]
    my_morph = np.zeros((4))
    y_values = np.zeros(4)
    x_values = np.zeros(4)
    # Obtain (max/min) values and index from the intervals
    [x_values[0], y_values[0]] = max(enumerate(beat[0:40]), key=operator.itemgetter(1))
    [x_values[1], y_values[1]] = min(enumerate(beat[75:85]), key=operator.itemgetter(1))
    [x_values[2], y_values[2]] = min(enumerate(beat[95:105]), key=operator.itemgetter(1))
    [x_values[3], y_values[3]] = max(enumerate(beat[150:180]), key=operator.itemgetter(1))

    x_values[1] = x_values[1] + 75
    x_values[2] = x_values[2] + 95
    x_values[3] = x_values[3] + 150

    # Norm data before compute distance
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))

    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)

    for n in range(0, 4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n])
        y_diff = R_value - y_values[n]
        my_morph[n] = np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))

    if np.isnan(my_morph[n]):
        my_morph[n] = 0.0

    return my_morph


# Compute the HOS descriptor for a beat
# Skewness (3 cumulant) and kurtosis (4 cumulant)
def compute_hos_descriptor(beat, n_intervals, lag):
    hos_b = np.zeros(((n_intervals - 1) * 2))
    for i in range(0, n_intervals - 1):
        pose = (lag * (i + 1))
        interval = beat[(pose - (lag // 2)):(pose + (lag // 2))]

        # Skewness  
        hos_b[i] = scipy.stats.skew(interval, 0, True)

        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0

        # Kurtosis
        hos_b[(n_intervals - 1) + i] = scipy.stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos_b[(n_intervals - 1) + i]):
            hos_b[(n_intervals - 1) + i] = 0.0
    return hos_b


# Compute the uniform LBP 1D from signal with neigh equal to number of neighbours
# and return the 59 histogram:
# 0-57: uniform patterns
# 58: the non uniform pattern
# NOTE: this method only works with neigh = 8
def compute_Uniform_LBP(signal, neigh=8):
    hist_u_lbp = np.zeros(59, dtype=float)

    avg_win_size = 2
    # NOTE: Reduce sampling by half
    # signal_avg = scipy.signal.resample(signal, len(signal) / avg_win_size)

    for i in range(neigh // 2, len(signal) - neigh // 2):
        pattern = np.zeros(neigh)
        ind = 0
        for n in chain(range(-neigh // 2, 0), range(1, neigh // 2 + 1)):
            if signal[i] > signal[i + n]:
                pattern[ind] = 1
            ind += 1
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
        if pattern_id in uniform_pattern_list:
            pattern_uniform_id = int(np.argwhere(uniform_pattern_list == pattern_id))
        else:
            pattern_uniform_id = 58  # Non uniforms patternsuse

        hist_u_lbp[pattern_uniform_id] += 1.0

    return hist_u_lbp


def compute_LBP(signal, neigh=4):
    hist_u_lbp = np.zeros(np.power(2, neigh), dtype=float)

    avg_win_size = 2
    # TODO: use some type of average of the data instead the full signal...
    # Average window-5 of the signal?
    # signal_avg = average_signal(signal, avg_win_size)
    signal_avg = scipy.signal.resample(signal, len(signal) // avg_win_size)

    for i in range(neigh // 2, len(signal) - neigh // 2):
        pattern = np.zeros(neigh)
        ind = 0
        for n in chain(range(-neigh // 2, 0), range(1, neigh // 2 + 1)):
            if signal[i] > signal[i + n]:
                pattern[ind] = 1
            ind += 1
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        hist_u_lbp[pattern_id] += 1.0

    return hist_u_lbp


# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.hermite.html
# Support Vector Machine-Based Expert System for Reliable Heartbeat Recognition
# 15 hermite coefficients!
def compute_HBF(beat):
    coeffs_hbf = np.zeros(15, dtype=float)
    coeffs_HBF_3 = hermfit(range(0, len(beat)), beat, 3)  # 3, 4, 5, 6?
    coeffs_HBF_4 = hermfit(range(0, len(beat)), beat, 4)
    coeffs_HBF_5 = hermfit(range(0, len(beat)), beat, 5)
    # coeffs_HBF_6 = hermfit(range(0,len(beat)), beat, 6)

    coeffs_hbf = np.concatenate((coeffs_HBF_3, coeffs_HBF_4, coeffs_HBF_5))

    return coeffs_hbf


def calc_RR_intervals(r_poses, valid_r):
    """

    :param r_poses: List[List[int]], dim: record, r_peaks
    :param valid_r: List[List[int]], dim: record, valid_r
    :return: List[RR_interval]
    """

    RR = [RR_intervals() for _ in range(len(r_poses))]

    for id_record in range(len(r_poses)):
        RR[id_record] = compute_RR_intervals(r_poses[id_record])

        assert len(RR[id_record].pre_R) == len(valid_r[id_record])

        # only consider valid r_peak
        RR[id_record].pre_R = RR[id_record].pre_R[(valid_r[id_record] == 1)]
        RR[id_record].post_R = RR[id_record].post_R[(valid_r[id_record] == 1)]
        RR[id_record].local_R = RR[id_record].local_R[(valid_r[id_record] == 1)]
        RR[id_record].global_R = RR[id_record].global_R[(valid_r[id_record] == 1)]

    return RR


def get_features_rr(RR):
    """

    :param RR: List[RR_intervals()]
    :return:
    """

    f_RR = np.empty((0, 4))
    for id_record in range(len(RR)):
        row = np.column_stack((RR[id_record].pre_R,
                               RR[id_record].post_R,
                               RR[id_record].local_R,
                               RR[id_record].global_R))
        f_RR = np.vstack((f_RR, row))
    return f_RR


def get_features_rr_norm(RR):
    """

    :param RR: List[RR_intervals()]
    :return:
    """

    f_RR_norm = np.empty((0, 4))
    for id_record in range(len(RR)):
        # Compute avg values!
        avg_pre_R = np.average(RR[id_record].pre_R)
        avg_post_R = np.average(RR[id_record].post_R)
        avg_local_R = np.average(RR[id_record].local_R)
        avg_global_R = np.average(RR[id_record].global_R)

        row = np.column_stack(
            (RR[id_record].pre_R / avg_pre_R, RR[id_record].post_R / avg_post_R, RR[id_record].local_R / avg_local_R,
             RR[id_record].global_R / avg_global_R))
        f_RR_norm = np.vstack((f_RR_norm, row))

    return f_RR_norm


def get_features_raw(beats, leads_flag):
    # beats, dim: record, id_beat, id_lead, signal
    num_leads = sum(leads_flag)

    beat_len = len(beats[0][0][0])

    f_raw = np.empty((0, beat_len * num_leads))

    for p in range(len(beats)):
        for beat in beats[p]:
            f_raw_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_raw_lead.size == 1:
                        f_raw_lead = beat[s]
                    else:
                        f_raw_lead = np.hstack((f_raw_lead, beat[s]))
            f_raw = np.vstack((f_raw, f_raw_lead))

    return f_raw


def get_features_resample(beats, leads_flag):
    num_leads = sum(leads_flag)

    f_raw = np.empty((0, 10 * num_leads))

    for p in range(len(beats)):
        for beat in beats[p]:
            f_raw_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    resamp_beat = scipy.signal.resample(beat[s], 10)
                    if f_raw_lead.size == 1:
                        f_raw_lead = resamp_beat
                    else:
                        f_raw_lead = np.hstack((f_raw_lead, resamp_beat))
            f_raw = np.vstack((f_raw, f_raw_lead))
    return f_raw


# LBP 1D
# 1D-local binary pattern based feature extraction for classification of epileptic EEG signals: 2014, unas 55 citas, Q2-Q1 Matematicas
# https://ac.els-cdn.com/S0096300314008285/1-s2.0-S0096300314008285-main.pdf?_tid=8a8433a6-e57f-11e7-98ec-00000aab0f6c&acdnat=1513772341_eb5d4d26addb6c0b71ded4fd6cc23ed5

# 1D-LBP method, which derived from implementation steps of 2D-LBP, was firstly proposed by Chatlani et al. for detection of speech signals that is non-stationary in nature [23]

# From Raw signal

# TODO: Some kind of preprocesing or clean high frequency noise?

# Compute 2 Histograms: LBP or Uniform LBP
# LBP 8 = 0-255
# U-LBP 8 = 0-58
# Uniform LBP are only those pattern wich only presents 2 (or less) transitions from 0-1 or 1-0
# All the non-uniform patterns are asigned to the same value in the histogram
def get_features_lbp(beats, leads_flag):
    num_leads = sum(leads_flag)

    f_lbp = np.empty((0, 16 * num_leads))

    for p in range(len(beats)):
        for beat in beats[p]:
            f_lbp_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_lbp_lead.size == 1:

                        f_lbp_lead = compute_LBP(beat[s], 4)
                    else:
                        f_lbp_lead = np.hstack((f_lbp_lead, compute_LBP(beat[s], 4)))
            f_lbp = np.vstack((f_lbp, f_lbp_lead))

    return f_lbp


def get_features_u_lbp(beats, leads_flag):
    num_leads = sum(leads_flag)

    f_lbp = np.empty((0, 59 * num_leads))

    for p in range(len(beats)):
        for beat in beats[p]:
            f_lbp_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_lbp_lead.size == 1:

                        f_lbp_lead = compute_Uniform_LBP(beat[s], 8)
                    else:
                        f_lbp_lead = np.hstack((f_lbp_lead, compute_Uniform_LBP(beat[s], 8)))
            f_lbp = np.vstack((f_lbp, f_lbp_lead))

    return f_lbp


def get_features_hbf5(beats, leads_flag):
    num_leads = sum(leads_flag)

    f_hbf = np.empty((0, 15 * num_leads))
    for p in range(len(beats)):
        for beat in beats[p]:
            f_hbf_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_hbf_lead.size == 1:

                        f_hbf_lead = compute_HBF(beat[s])
                    else:
                        f_hbf_lead = np.hstack((f_hbf_lead, compute_HBF(beat[s])))
            f_hbf = np.vstack((f_hbf, f_hbf_lead))
    return f_hbf


def get_features_wvlt(beats, leads_flag):
    num_leads = sum(leads_flag)

    f_wav = np.empty((0, 23 * num_leads))

    for p in range(len(beats)):
        for b in beats[p]:
            f_wav_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_wav_lead.size == 1:
                        f_wav_lead = compute_wavelet_descriptor(b[s], 'db1', 3)
                    else:
                        f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], 'db1', 3)))
            f_wav = np.vstack((f_wav, f_wav_lead))
            # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))
    return f_wav


def get_features_wvlt_pca(beats, leads_flag, DS, family, level, pca_k):
    num_leads = sum(leads_flag)

    f_wav = np.empty((0, 23 * num_leads))

    for p in range(len(beats)):
        for b in beats[p]:
            f_wav_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_wav_lead.size == 1:
                        f_wav_lead = compute_wavelet_descriptor(b[s], family, level)
                    else:
                        f_wav_lead = np.hstack((f_wav_lead, compute_wavelet_descriptor(b[s], family, level)))
            f_wav = np.vstack((f_wav, f_wav_lead))
            # f_wav = np.vstack((f_wav, compute_wavelet_descriptor(b,  'db1', 3)))

    if DS == 'DS1':
        # Compute PCA
        # PCA = sklearn.decomposition.KernelPCA(pca_k) # gamma_pca
        IPCA = IncrementalPCA(n_components=pca_k,
                              batch_size=10)  # NOTE: due to memory errors, we employ IncrementalPCA
        IPCA.fit(f_wav)

        # Save PCA
        save_wvlt_PCA(IPCA, pca_k, family, level)
    else:
        # Load PCAfrom sklearn.decomposition import PCA, IncrementalPCA
        IPCA = load_wvlt_PCA(pca_k, family, level)

    # Extract the PCA
    # f_wav_PCA = np.empty((0, pca_k * num_leads))
    f_wav_PCA = IPCA.transform(f_wav)

    return f_wav


def get_featurs_hos(beats, leads_flag):
    beat_len = len(beats[0][0][0])
    num_leads = sum(leads_flag)

    n_intervals = 6
    lag = int(round((beat_len) / n_intervals))

    f_HOS = np.empty((0, (n_intervals - 1) * 2 * num_leads))
    for p in range(len(beats)):
        for b in beats[p]:
            f_HOS_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_HOS_lead.size == 1:
                        f_HOS_lead = compute_hos_descriptor(b[s], n_intervals, lag)
                    else:
                        f_HOS_lead = np.hstack((f_HOS_lead, compute_hos_descriptor(b[s], n_intervals, lag)))
            f_HOS = np.vstack((f_HOS, f_HOS_lead))
            # f_HOS = np.vstack((f_HOS, compute_hos_descriptor(b, n_intervals, lag)))
    return f_HOS


def get_features_mymorph(beats, leads_flag, winL, winR):
    num_leads = sum(leads_flag)

    f_myMorhp = np.empty((0, 4 * num_leads))
    for p in range(len(beats)):
        for b in beats[p]:
            f_myMorhp_lead = np.empty([])
            for s in range(2):
                if leads_flag[s] == 1:
                    if f_myMorhp_lead.size == 1:
                        f_myMorhp_lead = compute_my_own_descriptor(b[s], winL, winR)
                    else:
                        f_myMorhp_lead = np.hstack(
                            (f_myMorhp_lead, compute_my_own_descriptor(b[s], winL, winR)))
            f_myMorhp = np.vstack((f_myMorhp, f_myMorhp_lead))
            # f_myMorhp = np.vstack((f_myMorhp, compute_my_own_descriptor(b, winL, winR)))
    return f_myMorhp
