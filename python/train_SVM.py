#!/usr/bin/env python

"""
train_SVM.py
    
VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from pprint import pprint

from path_manager import path_to_model, path_to_measure
from load_MITBIH import load_mit_db
from aggregation_voting_strategies import *
from utils import PrintTime, calc_class_weights


def train_and_evaluation(
        multi_mode='ovo',
        winL=90,
        winR=90,
        do_preprocess=True,
        use_weight_class=False,
        maxRR=True,
        use_RR=True,
        norm_RR=True,
        compute_morph={''},
        oversamp_method='',
        pca_k=0,
        feature_selection=False,
        do_cross_val=False,
        c_value=0.001,
        gamma_value=0.0,
        reduced_DS=False,
        leads_flag=[1, 0],
        verbose=True,
):
    """
    train the model on training records.
    test the model on testing records.


    :param multi_mode:
    :param winL:
    :param winR:
    :param do_preprocess:
    :param use_weight_class:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param oversamp_method:
    :param pca_k: int, if it is larger than 0, PCA will be activated
    :param feature_selection: Bool,
    :param do_cross_val: Str, 'pat_cv' or 'beat_cv'
    :param c_value:
    :param gamma_value:
    :param reduced_DS: Bool,
    :param leads_flag: List[int], [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :return:
    """

    if verbose:
        print("Runing trival_main !")

    params_for_naming = locals()

    # 1. Load data
    if verbose:
        print("loading the data ...")

    tr_features, tr_labels, eval_features, eval_labels = load_ml_data((winL, winR),
                                                                      do_preprocess,
                                                                      reduced_DS,
                                                                      maxRR,
                                                                      use_RR,
                                                                      norm_RR,
                                                                      compute_morph,
                                                                      verbose=verbose)

    # 2. train the model
    if verbose:
        print("Ready to train the model on MIT-BIH DS1: ...")

    model_kwargs = {
        "c_value": c_value,
        "gamma_value": gamma_value,
        "multi_mode": multi_mode,
    }
    model_svm_path = path_to_model(**params_for_naming)
    svm_model = train_model_module(model_svm_path, tr_features, tr_labels, **model_kwargs, verbose=verbose)

    # 3. evaluate the model
    if verbose:
        print("Testing model on MIT-BIH DS2: " + model_svm_path + "...")

    perf_measures_path = path_to_measure(**params_for_naming)
    eval_model_module(perf_measures_path,
                      svm_model,
                      tr_features,
                      tr_labels,
                      eval_features,
                      eval_labels,
                      **model_kwargs,
                      verbose=verbose)

    print("congrats! evaluation complete! ")


def load_ml_data(ws,
                 data_do_preprocess,
                 data_is_reduced,
                 ml_is_maxRR,
                 ml_use_RR,
                 ml_norm_RR,
                 ml_compute_morph,
                 is_normalize=True,
                 cross_patient=True,
                 verbose=False):
    if cross_patient:
        # Load train data
        # tr_ means train_
        tr_features, tr_labels, _ = load_mit_db('DS1',
                                                ws,
                                                data_do_preprocess,
                                                data_is_reduced,
                                                ml_is_maxRR,
                                                ml_use_RR,
                                                ml_norm_RR,
                                                ml_compute_morph)

        # Load Test data
        # eval_ means evaluation
        eval_features, eval_labels, _ = load_mit_db('DS2',
                                                    ws,
                                                    data_do_preprocess,
                                                    data_is_reduced,
                                                    ml_is_maxRR,
                                                    ml_use_RR,
                                                    ml_norm_RR,
                                                    ml_compute_morph)
        if is_normalize and verbose:
            print("normalizing the data ... ")

            scaler = StandardScaler()
            scaler.fit(tr_features)
            tr_features_scaled = scaler.transform(tr_features)
            eval_features_scaled = scaler.transform(eval_features)
        else:
            tr_features_scaled = tr_features
            eval_features_scaled = eval_features

    else:
        features, labels, _ = load_mit_db('DS12',
                                          ws,
                                          data_do_preprocess,
                                          data_is_reduced,
                                          ml_is_maxRR,
                                          ml_use_RR,
                                          ml_norm_RR,
                                          ml_compute_morph)

        tr_features, eval_features, tr_labels, eval_labels = train_test_split(features, labels,
                                                                              test_size=0.2,
                                                                              random_state=2020)

        if is_normalize and verbose:
            print("normalizing the data ... ")

            scaler = StandardScaler()
            scaler.fit(tr_features)
            tr_features_scaled = scaler.transform(tr_features)
            eval_features_scaled = scaler.transform(eval_features)
        else:
            tr_features_scaled = tr_features
            eval_features_scaled = eval_features

    # todo: store the scaler
    # input shape is the same
    assert tr_features_scaled.shape[1:] == eval_features_scaled.shape[1:]

    return tr_features_scaled, tr_labels, eval_features_scaled, eval_labels


def train_model_module(model_svm_path, tr_features, tr_labels, verbose=False, **model_kwargs):
    """
    train the model if it is needed.

    :param model_svm_path:
    :param tr_features:
    :param tr_labels:
    :param verbose:
    :param model_kwargs:
    :return:
    """
    if os.path.isfile(model_svm_path):
        # Load the trained model!
        svm_model = joblib.load(model_svm_path)

    else:
        svm_model = train_model(model_svm_path, tr_features, tr_labels, **model_kwargs, verbose=verbose)
    return svm_model


def train_model(model_svm_path, tr_features, tr_labels, verbose=False, **model_kwargs):
    C_value = model_kwargs.get("c_value", 1)
    gamma_value = model_kwargs.get("gamma_value", 0)
    multi_mode = model_kwargs.get("multi_model", "ovo")
    use_probability = False
    class_weights = calc_class_weights(tr_labels)
    # class_weight='balanced',
    gamma_value = gamma_value if gamma_value != 0.0 else "auto"

    # TODO load best params from cross validation!

    # NOTE 0.0 means 1/n_features default value
    svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma=gamma_value,
                        coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                        cache_size=200, class_weight=class_weights, verbose=verbose,
                        max_iter=-1, decision_function_shape=multi_mode, random_state=None)

    with PrintTime("train the model", verbose=verbose):
        svm_model.fit(tr_features, tr_labels)

    if model_svm_path:
        # Export model: save/write trained SVM model
        if not os.path.exists(os.path.dirname(model_svm_path)):
            os.makedirs(os.path.dirname(model_svm_path))
        joblib.dump(svm_model, model_svm_path)

    # TODO Export StandardScaler()

    return svm_model


def eval_model_module(perf_measures_path,
                      svm_model,
                      tr_features,
                      tr_labels,
                      eval_features,
                      eval_labels,
                      multi_mode,
                      c_value,
                      gamma_value,
                      is_include_train=False,
                      is_include_ovo_voting_exp=False,
                      **kwargs
                      ):
    # ovo_voting:
    # Let's test new data!
    print("Evaluation on DS2 ...")
    eval_model(svm_model,
               eval_features,
               eval_labels,
               multi_mode,
               'ovo_voting',
               perf_measures_path,
               c_value,
               gamma_value,
               DS='')

    # Simply add 1 to the win class
    if is_include_train:
        print("Evaluation on DS1 ...")
        eval_model(svm_model,
                   tr_features,
                   tr_labels,
                   multi_mode,
                   'ovo_voting',
                   perf_measures_path,
                   c_value,
                   gamma_value,
                   DS='Train_')

    # ovo_voting_exp:
    # Consider the post prob adding to both classes
    if is_include_ovo_voting_exp:
        # Let's test new data!
        print("Evaluation on DS2 ...")
        eval_model(svm_model,
                   eval_features,
                   eval_labels,
                   multi_mode,
                   'ovo_voting_exp',
                   perf_measures_path,
                   c_value,
                   gamma_value,
                   DS='')

        if is_include_train:
            print("Evaluation on DS1 ...")
            eval_model(svm_model,
                       tr_features,
                       tr_labels,
                       multi_mode,
                       'ovo_voting_exp',
                       perf_measures_path,
                       c_value,
                       gamma_value,
                       DS='Train_')


# Eval the SVM model and export the results
def eval_model(svm_model, features, labels, multi_mode, voting_strategy,
               output_path, C_value, gamma_value, DS, verbose=False):
    """

    :param svm_model:
    :param features:
    :param labels:
    :param multi_mode: str, 'ovo' or 'ovr'
    :param voting_strategy: str, has effect when multi_model is ovo
    :param output_path:
    :param C_value:
    :param gamma_value:
    :param DS: Str, "" or "Train_", basically used as a marker in path name.
    :return:
    """

    decision = svm_model.decision_function(features)

    if verbose:
        print("make evaluation predict...")

    if multi_mode == 'ovr':
        predict = svm_model.predict(features)

    else:  # elif multi_mode == 'ovo':
        predict, _ = ovo_voting_handler(decision, 4, voting_strategy)

    # record the performance
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
    perf_measures = compute_AAMI_performance_measures(predict, labels)
    pprint(perf_measures.__dict__, indent=2)

    # save results
    gamma_str = 'g_' + str(gamma_value) if gamma_value != 0.0 else ""

    # 1) save AAMI results
    path_to_AAMI_results = os.path.join(output_path,
                                        "{}C_{}{}_score_Ijk_{}_{}.txt".format(DS,
                                                                              C_value,
                                                                              gamma_str,
                                                                              format(perf_measures.Ijk, '.2f'),
                                                                              voting_strategy))
    write_AAMI_results(perf_measures, path_to_AAMI_results)

    # 2) save decsion
    path_to_decision = os.path.join(output_path, "{}C_{}{}_decision_ovo.csv".format(DS, C_value, gamma_str))
    np.savetxt(path_to_decision, decision)

    # 3) save predict
    path_to_predict = os.path.join(output_path, "{}C_{}_predict_{}.csv").format(DS, C_value, gamma_value,
                                                                                voting_strategy)
    np.savetxt(path_to_predict, predict.astype(int), '%.0f')

    print("Results writed at " + output_path + '/' + DS + 'C_' + str(C_value))


if __name__ == "__main__":
    train_and_evaluation(
        multi_mode="ovo",
        winL=90,
        winR=90,
        do_preprocess=True,
        maxRR=True,
        use_RR=False,
        norm_RR=True,
        compute_morph={'resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'},
        reduced_DS=True,
        leads_flag=[1, 0],
    )
