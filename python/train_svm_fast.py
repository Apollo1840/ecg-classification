#!/usr/bin/env python

import json
from sklearn.metrics import f1_score, classification_report
from pprint import pprint

from util_path_manager import path_to_model, path_to_measure
from data_load import load_mit_db
from model_aggregation import *
from utils import PrintTime, calc_class_weights
from config import *
from train_svm import load_ml_data, train_model_module


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
        pca_k=0,
        leads_flag=[1, 0],
        feature_selection=""
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

    f1score_marco = f1_score(eval_labels, pred_labels, average='macro')
    print("marco f1 score: ", f1score_marco)

    cls_report = classification_report(eval_labels, pred_labels, target_names=AAMI_CLASSES)
    print(cls_report)

    report_dict = {
        "config": params_for_naming,
        "cls_report": cls_report,
        "marco_f1": f1score_marco
    }

    cross_val_type = "pat_cv" if cross_patient else "beat_cv"
    with open("hypersearch/{}_f1_{:.4}.json".format(cross_val_type, f1score_marco), "w") as f:
        json.dump(report_dict, f)

    print("congrats! evaluation complete! ")

    return svm_model, pred_labels, eval_labels

