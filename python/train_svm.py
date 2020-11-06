#!/usr/bin/env python

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from pprint import pprint
from tqdm import tqdm

# internal depencency:
from data_load import load_features_from_mitdb
from model_aggregation import *

from utils import PrintTime, calc_class_weights
from util_path_manager import path_to_model, path_to_measure


def train_and_evaluation(

        # date_parameters
        winL=90,
        winR=90,
        do_preprocess=False,
        reduced_DS=False,  # include lead 2 or not.
        maxRR=False,
        use_RR=False,
        norm_RR=False,
        compute_morph={''},
        is_normalize=False,
        cross_patient=False,

        # model_parameters
        multi_mode='ovr',
        c_value=0.001,
        gamma_value=0.0,  # 0.0 == "auto", in train_model() defined

        # training_parameters
        oversamp_method='',
        pca_k=0,
        feature_selection=False,

        verbose=False,
):
    """
    train the model on training records.
    test the model on testing records.


    :param multi_mode:
    :param winL:
    :param winR:
    :param do_preprocess:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param oversamp_method:
    :param pca_k: int, if it is larger than 0, PCA will be activated
    :param feature_selection: Bool,
    :param cross_patient: Bool
    :param c_value:
    :param gamma_value:
    :param reduced_DS: Bool,
    :return:
    """

    if verbose:
        print("Running train and evaluation !")

    params_for_naming = locals()
    path_to_svm_model = path_to_model(leads_flag=(1, int(reduced_DS)),
                                      use_weight_class=True,
                                      **params_for_naming)
    path_to_perf_measures = path_to_measure(leads_flag=(1, int(reduced_DS)),
                                            use_weight_class=True,
                                            **params_for_naming)

    ###
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
                                                                      is_normalize=is_normalize,
                                                                      cross_patient=cross_patient,
                                                                      verbose=verbose)

    check_data_shape(tr_features, tr_labels, eval_features, eval_labels)

    ###
    # 2. train the model
    if verbose and cross_patient:
        print("Ready to train the model on MIT-BIH DS1: ...")

    model_kwargs = {
        "multi_mode": multi_mode,
        "c_value": c_value,
        "gamma_value": gamma_value,
    }
    svm_model = train_model_module(path_to_svm_model, tr_features, tr_labels, **model_kwargs, verbose=verbose)

    ###
    # 3. evaluate the model
    if verbose:
        print("Testing model on MIT-BIH DS2: " + path_to_svm_model + "...")

    eval_model_module(path_to_perf_measures,
                      svm_model,
                      tr_features,
                      tr_labels,
                      eval_features,
                      eval_labels,
                      **model_kwargs,
                      is_include_train=False,
                      is_include_ovo_voting_exp=False,
                      is_save_prediction=False,
                      verbose=verbose)

    print("congrats! evaluation complete! ")

    return svm_model, eval_features, eval_labels


def load_ml_data(ws,
                 data_do_preprocess,
                 data_is_reduced,
                 ml_is_max_rr,
                 ml_use_rr,
                 ml_norm_rr,
                 ml_compute_morph,
                 is_normalize=True,
                 cross_patient=True,
                 verbose=False):
    if cross_patient:  # inter-patient
        # Load train data
        # tr_ means train_
        tr_features, tr_labels, _ = load_features_from_mitdb('DS1',
                                                             ws,
                                                             data_do_preprocess,
                                                             data_is_reduced,
                                                             ml_is_max_rr,
                                                             ml_use_rr,
                                                             ml_norm_rr,
                                                             ml_compute_morph,
                                                             verbose=verbose)

        # Load Test data
        # eval_ means evaluation
        eval_features, eval_labels, _ = load_features_from_mitdb('DS2',
                                                                 ws,
                                                                 data_do_preprocess,
                                                                 data_is_reduced,
                                                                 ml_is_max_rr,
                                                                 ml_use_rr,
                                                                 ml_norm_rr,
                                                                 ml_compute_morph,
                                                                 verbose=verbose)
    else:  # intra-patient
        features, labels, _ = load_features_from_mitdb('DS12',
                                                       ws,
                                                       data_do_preprocess,
                                                       data_is_reduced,
                                                       ml_is_max_rr,
                                                       ml_use_rr,
                                                       ml_norm_rr,
                                                       ml_compute_morph,
                                                       verbose=verbose)

        tr_features, eval_features, tr_labels, eval_labels = train_test_split(features, labels,
                                                                              test_size=0.2,
                                                                              random_state=2020)

    if is_normalize:

        if verbose:
            print("preprocess: \t normalizing the data ... ")

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
    c_value = model_kwargs.get("c_value", 1)
    gamma_value = model_kwargs.get("gamma_value", 0)
    multi_mode = model_kwargs.get("multi_model", "ovo")
    use_probability = False
    class_weights = calc_class_weights(tr_labels)
    # class_weights = "balanced",
    gamma_value = gamma_value if gamma_value != 0.0 else "auto"

    # TODO load best params from cross validation!

    # NOTE 0.0 means 1/n_features default value
    svm_model = svm.SVC(C=c_value, kernel='rbf', degree=3, gamma=gamma_value,
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
                      is_save_prediction=False,
                      verbose=False,
                      ):
    # ovo_voting:
    # Let's test new data!
    print("Evaluation on name_ds2 ...")
    eval_model(svm_model,
               eval_features,
               eval_labels,
               multi_mode,
               'ovo_voting',
               perf_measures_path,
               c_value,
               gamma_value,
               is_save_prediction=is_save_prediction,
               name_ds='',
               verbose=verbose,
               )

    # Simply add 1 to the win class
    if is_include_train:
        print("Evaluation on name_ds1 ...")
        eval_model(svm_model,
                   tr_features,
                   tr_labels,
                   multi_mode,
                   'ovo_voting',
                   perf_measures_path,
                   c_value,
                   gamma_value,
                   verbose=verbose,
                   name_ds='Train_')

    # ovo_voting_exp:
    # Consider the post prob adding to both classes
    if is_include_ovo_voting_exp:
        # Let's test new data!
        print("Evaluation on name_ds2 ...")
        eval_model(svm_model,
                   eval_features,
                   eval_labels,
                   multi_mode,
                   'ovo_voting_exp',
                   perf_measures_path,
                   c_value,
                   gamma_value,
                   verbose=verbose,
                   name_ds='')

        if is_include_train:
            print("Evaluation on name_ds1 ...")
            eval_model(svm_model,
                       tr_features,
                       tr_labels,
                       multi_mode,
                       'ovo_voting_exp',
                       perf_measures_path,
                       c_value,
                       gamma_value,
                       verbose=verbose,
                       name_ds='Train_')


# Eval the SVM model and export the results
def eval_model(svm_model,
               features,
               labels,
               multi_mode,
               voting_strategy,
               output_path,
               c_value,
               gamma_value,
               name_ds,
               is_save_prediction=False,
               verbose=False):
    """

    :param svm_model:
    :param features:
    :param labels:
    :param multi_mode: str, 'ovo' or 'ovr'
    :param voting_strategy: str, has effect when multi_model is ovo
    :param output_path:
    :param c_value:
    :param gamma_value:
    :param name_ds: Str, "" or "Train_", basically used as a marker in path name.
    :param is_save_prediction:
    :param verbose
    :return:
    """

    if verbose:
        print("make evaluation predict...")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if verbose:
        print("model predicting with mode: {} ...".format(multi_mode))

    # 1) get prediction based on multi_mode
    if multi_mode == 'ovr':
        predict = svm_model.predict(features)

    else:  # elif multi_mode == 'ovo':
        decision = svm_model.decision_function(features)
        # decision is the probablities predictions of each model pair,
        # decision dim: id_item, prob_positive_of_model_pair_i
        # you could use ovo_class_combinations() to find out pos_label and neg_label of each pair
        predict, _ = ovo_voting_handler(decision, 4, voting_strategy)

        is_save_decision = is_save_prediction
        if is_save_decision:
            path_to_decision = os.path.join(output_path, name_decision(name_ds, c_value, gamma_value))
            np.savetxt(path_to_decision, decision)

    if is_save_prediction:
        path_to_predict = os.path.join(output_path, name_prediction(name_ds, c_value, gamma_value, voting_strategy))
        np.savetxt(path_to_predict, predict.astype(int), '%.0f')

    if verbose:
        print("calculating the performance ... ")

    # 2) calculate and record the performance
    # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
    # predict, labels : List[int]
    perf_measures = compute_AAMI_performance_measures(predict, labels, verbose=verbose)
    pprint(perf_measures.__dict__, indent=2)

    path_to_perf_results = os.path.join(output_path, name_perf_results_file(name_ds, c_value, gamma_value,
                                                                            perf_measures.Ijk, voting_strategy))
    write_AAMI_results(perf_measures, path_to_perf_results)

    if verbose:
        print("Results writed at " + output_path + '/' + name_ds + 'C_' + str(c_value))


def name_gamma_str(gamma_value):
    if isinstance(gamma_value, float):
        gamma_str = 'g_' + str(gamma_value) if gamma_value != 0.0 else ""
    else:
        gamma_str = str(gamma_value)
    return gamma_str


def name_perf_results_file(name_ds, c_value, gamma_value, ijk, voting_strategy):
    return "{}C_{}{}_score_Ijk_{}_{}.txt".format(name_ds,
                                                 c_value,
                                                 name_gamma_str(gamma_value),
                                                 format(ijk, '.2f'),
                                                 voting_strategy)


def name_decision(name_ds, c_value, gamma_value):
    return "{}C_{}{}_decision_ovo.csv".format(name_ds, c_value,
                                              name_gamma_str(gamma_value))


def name_prediction(name_ds, c_value, gamma_value, voting_strategy):
    return "{}C_{}_predict_{}.csv".format(name_ds, c_value, name_gamma_str(gamma_value), voting_strategy)


def check_data_shape(x_train, y_train, x_test, y_test):
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    assert x_train.shape[1] == x_test.shape[1]

    print("y_train length: ", len(y_train))
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)

    print("y_train labels:", np.unique(y_train))
    print("y_train[:10]: ", y_train[:10])

    print("y_test length: ", len(y_test))
    print("y_test[:10]: ", y_test[:10])


if __name__ == "__main__":
    for c_value in tqdm([0.1, 1, 5, 10, 20, 50]):
        train_and_evaluation(

            # data_parameters
            winL=90,
            winR=90,
            do_preprocess=True,
            reduced_DS=False,
            maxRR=True,
            use_RR=True,
            norm_RR=True,
            compute_morph={'u-lbp', 'wvlt', 'HOS', 'MyMorph'},
            is_normalize=True,
            cross_patient=True,

            # model_parameters
            multi_mode="ovo",
            c_value=c_value,

            verbose=True
        )
