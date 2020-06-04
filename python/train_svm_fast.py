#!/usr/bin/env python

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from pprint import pprint

from util_path_manager import path_to_model, path_to_measure
from data_load import load_mit_db
from model_aggregation import *
from utils import PrintTime, calc_class_weights


def model_search_unit():
    fixed_parameters = {
        "multi_mode": "ovo",
        "winL": 90,
        "winR": 90,
        "do_preprocess": True,
        "use_weight_class": True,
        "maxRR": True,
        "use_RR": False,
        "norm_RR": True,
        "oversamp_method": "",
        "cross_patient": False,
        "reduced_DS": False,
        "verbose": True,
    }

    searchable_params = {
        "c_value": 0.001,
        "gamma_value": 0.0,
        "compute_morph": ['resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'],
    }

    train_and_evaluation(**fixed_parameters, **searchable_params)


def train_and_evaluation(
        multi_mode='ovo',
        winL=90,
        winR=90,
        do_preprocess=True,
        use_weight_class=True,
        maxRR=True,
        use_RR=False,
        norm_RR=True,
        compute_morph={''},
        oversamp_method='',
        cross_patient=False,
        c_value=0.001,
        gamma_value=0.0,
        reduced_DS=False,
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
    :param cross_patient: Bool
    :param c_value:
    :param gamma_value:
    :param reduced_DS: Bool,
    :param leads_flag: List[int], [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :return:
    """

    if verbose:
        print("Running train and evaluation !")

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
                                                                      cross_patient=cross_patient,
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

    pred_labels = svm_model.predict(eval_features)
    print("marco f1 score: ", f1_score(eval_labels, pred_labels, average='macro'))

    print("congrats! evaluation complete! ")

    return svm_model, pred_labels, eval_labels


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

        tr_features, eval_features, tr_labels, eval_labels = train_test_split(features,
                                                                              labels,
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


def train_model_module(model_svm_path,
                       tr_features,
                       tr_labels,
                       verbose=False,
                       **model_kwargs):
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


def train_model(model_svm_path,
                tr_features,
                tr_labels,
                verbose=False,
                **model_kwargs):

    C_value = model_kwargs.get("c_value", 1)
    gamma_value = model_kwargs.get("gamma_value", 0)
    multi_mode = model_kwargs.get("multi_model", "ovo")

    use_probability = False
    class_weights = calc_class_weights(tr_labels)
    # class_weight='balanced',
    gamma_value = gamma_value if gamma_value != 0.0 else "auto"

    # TODO load best params from cross validation!

    # NOTE 0.0 means 1/n_features default value
    svm_model = svm.SVC(C=C_value,
                        kernel='rbf',
                        degree=3,
                        gamma=gamma_value,
                        coef0=0.0,
                        shrinking=True,
                        probability=use_probability,
                        tol=0.001,
                        cache_size=200,
                        class_weight=class_weights,
                        verbose=verbose,
                        max_iter=-1,
                        decision_function_shape=multi_mode,
                        random_state=None)

    with PrintTime("train the model", verbose=verbose):
        svm_model.fit(tr_features, tr_labels)

    if model_svm_path:
        # Export model: save/write trained SVM model
        if not os.path.exists(os.path.dirname(model_svm_path)):
            os.makedirs(os.path.dirname(model_svm_path))
        joblib.dump(svm_model, model_svm_path)

    # TODO Export StandardScaler()

    return svm_model


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
