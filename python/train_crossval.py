#!/usr/bin/env python

import numpy as np
import random
from sklearn import svm

from evaluation_AAMI import *
from aggregation_voting_strategies import ovo_voting_handler
from utils import *


def run_cross_val(features, labels, patient_num_beats, division_mode, k):
    """

    :param features:
    :param labels:
    :param patient_num_beats:
    :param division_mode: str.  'pat_cv' or 'beat_cv'
    :param k: int.
    :return:
    """

    print("Runing Cross validation...")

    # C_values
    # gamma_values
    C_values = [0.1]
    # C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 1000]

    # ijk_scores = np.zeros(len(C_values))
    cv_scores = []
    for c_svm in C_values:
        cv_scores.append(run_cross_val_single(features, labels, patient_num_beats, division_mode, c_svm, k))

    return np.array(cv_scores), C_values


def run_cross_val_single(features, labels, patient_num_beats, division_mode, c_value, k):
    """

    :param features:
    :param labels:
    :param patient_num_beats:
    :param division_mode: str.  'pat_cv' or 'beat_cv'
    :param k: int.
    :return:
    """

    ################
    # PREPARE DATA
    ################
    features = np.array(features)
    labels = np.array(labels)

    k_folds_indices = []
    if division_mode == 'pat_cv':
        k_folds_indices = cross_val_index_by_patient(patient_num_beats)

    elif division_mode == 'beat_cv':
        k_folds_indices = cross_val_index_by_beat(labels, k)

    else:
        print("You must specify which kind of crossval type you use in do_cross_val, "
              "eg do_cross_val='pat_cv' or 'beat_cv'.")
        return None

    ################
    # RUN CROSS VAL
    ################
    cv_scores = []
    for kk in range(k):

        # 1) prepare data
        print("preparing data ...")
        # print(k_folds_indices)

        indices_val = np.array(k_folds_indices[kk])
        indices_trn = np.array(flatten_list([k_folds_indices[i] for i in range(k) if i != kk]))
        # not k_folds_indices[kk]

        # no overlap between train and test
        assert not any(np.isin(indices_val, indices_trn))

        tr_features = features[indices_trn]
        tr_labels = labels[indices_trn]

        val_features = features[indices_val]
        val_labels = labels[indices_val]

        ####################################################
        # 2) Train
        print("training ...")
        C_value = c_value
        multi_mode = 'ovo'

        # get class_weights based on tr_labels
        class_weights = calc_class_weights(tr_labels)

        # class_weight='balanced',
        svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma='auto',
                            coef0=0.0, shrinking=True, probability=False, tol=0.001,
                            cache_size=200, class_weight=class_weights, verbose=False,
                            max_iter=-1, decision_function_shape=multi_mode, random_state=None)

        # Let's Train!
        # print(tr_features.shape)
        # print(tr_labels.shape)
        svm_model.fit(tr_features, tr_labels)

        #########################################################################
        # 3) Test SVM model
        print("scoring ... ")

        # ovo_voting:
        # Simply add 1 to the win class
        perf_measures = eval_crossval_fold(svm_model,
                                           val_features,
                                           val_labels,
                                           multi_mode,
                                           'ovo_voting_exp')

        # ijk_scores[index_cv] += perf_measures.Ijk

        cv_score = np.average(perf_measures.F_measure)
        print(cv_score)

        cv_scores.append(cv_score)
        print("c value({}) cross val k {}/{} AVG(F-measure) = {}".format(C_value, kk, k, sum(cv_scores)/(kk+1)))

        # TODO g-mean?
        # Zhang et al computes the g-mean.
        # But they computed the g-mean value for each SVM model of the 1 vs 1. NvsS, NvsV, ..., SvsV....

    # beat division

    cv_score = sum(cv_scores)/float(k)  # Average this result with the rest of the k-folds
    # NOTE: what measure maximize in the cross val????

    # c_values
    print(cv_score)

    return cv_score


# Eval the SVM model and export the results
def eval_crossval_fold(svm_model, features, labels, multi_mode, voting_strategy):
    if multi_mode == 'ovo':
        decision_ovo = svm_model.decision_function(features)
        predict_ovo, _ = ovo_voting_handler(decision_ovo, 4, voting_strategy)

        # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
        perf_measures = compute_AAMI_performance_measures(predict_ovo, labels)
    else:
        return None

    return perf_measures


# todo: test this func
def cross_val_index_by_patient(patient_num_beats):
    # NOTE: division by patient and oversampling couldnt used at the same time!!!!

    k = len(patient_num_beats)

    base = 0
    indices = []
    for kk in range(k):
        indices.append(range(base, base + patient_num_beats[kk]))
        base = base + patient_num_beats[kk]
    return indices


# todo: test the func
def cross_val_index_by_beat(labels, k, shuffle=True):
    # NOTE: class sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=False, random_state=None)[source]
    # Stratified K-Folds cross-validator
    # Provides train/test indices to split data in train/test sets.
    # Thirun_cross_vals cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class!!

    indices = [[] for _ in range(k)]
    n_classes = max(labels) + 1
    for c in range(n_classes):
        indices_in_class = [i for i, label in enumerate(labels) if label == c]
        if shuffle:
            random.shuffle(indices_in_class)
        increment = max(len(indices_in_class) // k, 1)
        base = 0
        for kk in range(k):
            indices[kk].extend(indices_in_class[base:base + increment])
            base = base + increment
    # indices = [np.array(i) for i in indices]
    return indices



