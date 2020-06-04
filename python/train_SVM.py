#!/usr/bin/env python

"""
train_SVM.py
    
VARPA, University of Coruna
Mondejar Guerra, Victor M.
23 Oct 2017
"""

import os
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import json
from pprint import pprint
from sklearn import decomposition

from load_MITBIH import load_mit_db
from evaluation_AAMI import *
from aggregation_voting_strategies import *
from oversampling import *
from cross_validation import *
from feature_selection import *
from path_manager import create_svm_model_name, path_to_model, create_oversamp_name, path_to_measure
from constant import *


def trival_main(
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
    :param C_value:
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
                 verbose=False):
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
    if verbose:
        print("normalizing the data ... ")

    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)
    eval_features_scaled = scaler.transform(eval_features)

    # todo: store the scaler

    # input shape is the same
    assert tr_features_scaled.shape[1:] == eval_features_scaled.shape[1:]

    return tr_features_scaled, tr_labels, eval_features_scaled, eval_labels


def train_model_module(model_svm_path, tr_features, tr_labels, verbose=False, **model_kwargs):
    if os.path.isfile(model_svm_path):
        # Load the trained model!
        svm_model = joblib.load(model_svm_path)

    else:
        svm_model = train_model(model_svm_path, tr_features, tr_labels, **model_kwargs, verbose=verbose)
    return svm_model


def train_model(model_svm_path, tr_features, tr_labels, verbose=False, **model_kwargs):
    C_value = model_kwargs.get("c_value", 1)
    gamma_value = model_kwargs.get("gamma_value", 0)
    multi_mode = model_kwargs.get("multi_model", "ovr")

    # TODO load best params from cross validation!
    use_probability = False

    class_weights = {}
    for c in range(4):
        class_weights.update({c: len(tr_labels) / float(np.count_nonzero(tr_labels == c))})

    # class_weight='balanced',
    if gamma_value != 0.0:  # NOTE 0.0 means 1/n_features default value
        svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma=gamma_value,
                            coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                            cache_size=200, class_weight=class_weights, verbose=verbose,
                            max_iter=-1, decision_function_shape=multi_mode, random_state=None)
    else:
        svm_model = svm.SVC(C=C_value, kernel='rbf', degree=3, gamma='auto',
                            coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                            cache_size=200, class_weight=class_weights, verbose=verbose,
                            max_iter=-1, decision_function_shape=multi_mode, random_state=None)

    # Let's Train!
    if verbose:
        print("Training the model with training data shape as {}! ...".format(tr_features.shape))

    start = time.time()
    svm_model.fit(tr_features, tr_labels)
    end = time.time()

    # TODO assert that the class_ID appears with the desired order,
    # with the goal of ovo make the combinations properly
    if verbose:
        print("Trained completed!\n\t" + model_svm_path + "\n \
            \tTime required: " + str(format(end - start, '.2f')) + " sec")

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
def eval_model(svm_model, features, labels, multi_mode, voting_strategy, output_path, C_value, gamma_value, DS):
    """

    :param svm_model:
    :param features:
    :param labels:
    :param multi_mode:
    :param voting_strategy:
    :param output_path:
    :param C_value:
    :param gamma_value:
    :param DS: Str, "" or "Train_", basically used as a marker in path name.
    :return:
    """

    if multi_mode == 'ovo':
        eval_model_ovo(svm_model, features, labels, voting_strategy, output_path, C_value, gamma_value, DS)

    else:  # elif multi_mode == 'ovr':
        eval_model_ovr(svm_model, features, labels, voting_strategy, output_path, C_value, gamma_value, DS)

    print("Results writed at " + output_path + '/' + DS + 'C_' + str(C_value))


def eval_model_ovo(svm_model, features, labels, voting_strategy, output_path, C_value, gamma_value, DS):
    decision_ovo = svm_model.decision_function(features)

    if voting_strategy == 'ovo_voting':
        predict_ovo, counter = ovo_voting(decision_ovo, 4)

    elif voting_strategy == 'ovo_voting_both':
        predict_ovo, counter = ovo_voting_both(decision_ovo, 4)

    else:  # elif voting_strategy == 'ovo_voting_exp':
        predict_ovo, counter = ovo_voting_exp(decision_ovo, 4)

    # svm_model.predict_log_proba  svm_model.predict_proba   svm_model.predict ...
    perf_measures = compute_AAMI_performance_measures(predict_ovo, labels)
    write_AAAMI_results_gamma(output_path, gamma_value, perf_measures, C_value, voting_strategy, DS)

    # save decision and predict
    if gamma_value != 0.0:
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) + '_decision_ovo.csv',
                   decision_ovo)
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) + '_predict_' +
                   voting_strategy + '.csv',
                   predict_ovo.astype(int), '%.0f')
    else:
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_decision_ovo.csv',
                   decision_ovo)
        np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_predict_' + voting_strategy + '.csv',
                   predict_ovo.astype(int), '%.0f')


def eval_model_ovr(svm_model, features, labels, voting_strategy, output_path, C_value, gamma_value, DS, verbose=True):
    if verbose:
        print("make evaluation decision...")

    decision_ovr = svm_model.decision_function(features)

    if verbose:
        print("make evaluation predict...")

    predict_ovr = svm_model.predict(features)

    perf_measures = compute_AAMI_performance_measures(predict_ovr, labels)
    write_AAAMI_results_gamma(output_path, gamma_value, perf_measures, C_value, voting_strategy, DS)

    # save decision and predict
    np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_decision_ovr.csv', decision_ovr)
    np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_predict_' + voting_strategy + '.csv',
               predict_ovr.astype(int), '%.0f')


def write_AAAMI_results_gamma(output_path, gamma_value, perf_measures, C_value, voting_strategy, DS):
    pprint(perf_measures.__dict__, indent=2)

    # Write results and also predictions on DS2
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if gamma_value != 0.0:
        write_AAMI_results(perf_measures,
                           output_path + '/' + DS + 'C_' + str(C_value) + 'g_' + str(gamma_value) +
                           '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + voting_strategy + '.txt')
    else:
        write_AAMI_results(perf_measures,
                           output_path + '/' + DS + 'C_' + str(C_value) +
                           '_score_Ijk_' + str(format(perf_measures.Ijk, '.2f')) + '_' + voting_strategy + '.txt')


def main(
        multi_mode='ovo',
        winL=90,
        winR=90,
        do_preprocess=True,
        use_weight_class=True,
        maxRR=True,
        use_RR=True,
        norm_RR=True,
        compute_morph={''},
        oversamp_method='',
        pca_k=0,
        feature_selection=False,
        do_cross_val=False,
        C_value=0.001,
        gamma_value=0.0,
        reduced_DS=False,
        leads_flag=[1, 0]):
    """

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
    :param feature_selection: Str,
    :param do_cross_val: Str, 'pat_cv' or 'beat_cv'
    :param C_value:
    :param gamma_value:
    :param reduced_DS: Bool, = True if leads_flag == [1, 1] else False
    :param leads_flag: List[int], [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :return:
    """

    params_for_naming = locals()

    print("Runing train_SVM.py!")

    # 1. load data

    # Load train data
    # tr_ means train_
    tr_features, tr_labels, tr_patient_num_beats = load_mit_db('DS1',
                                                               (winL, winR),
                                                               do_preprocess,
                                                               maxRR,
                                                               use_RR,
                                                               norm_RR,
                                                               compute_morph,
                                                               leads_flag)

    # Load Test data
    # eval_ means evaluation
    eval_features, eval_labels, eval_patient_num_beats = load_mit_db('DS2',
                                                                     (winL, winR),
                                                                     do_preprocess,
                                                                     maxRR,
                                                                     use_RR,
                                                                     norm_RR,
                                                                     compute_morph,
                                                                     leads_flag)

    if reduced_DS:
        np.savetxt('mit_db/' + 'exp_2_' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')
    else:
        np.savetxt('mit_db/' + 'DS2_labels.csv', eval_labels.astype(int), '%.0f')

        # if reduced_DS == True:
    #    np.savetxt('mit_db/' + 'exp_2_' + 'DS1_labels.csv', tr_labels.astype(int), '%.0f')
    # else:
    # np.savetxt('mit_db/' + 'DS1_labels.csv', tr_labels.astype(int), '%.0f')

    ##############################################################
    # 2. preprocess
    # 0) TODO if feature_Selection:
    # before oversamp!!?????

    # TODO perform normalization before the oversampling?
    if oversamp_method:
        # Filename
        oversamp_features_pickle_name = create_oversamp_name(reduced_DS, do_preprocess, compute_morph, winL, winR,
                                                             maxRR, use_RR, norm_RR, pca_k)

        # Do oversampling
        tr_features, tr_labels = perform_oversampling(oversamp_method, DB_PATH + 'oversamp/python_mit',
                                                      oversamp_features_pickle_name, tr_features, tr_labels)

    # Normalization of the input data
    # scaled: zero mean unit variance ( z-score )
    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)

    # scaled: zero mean unit variance ( z-score )
    eval_features_scaled = scaler.transform(eval_features)
    ##############################################################

    # 0) ????????????? feature_Selection: also after Oversampling???
    if feature_selection:
        print("Runing feature selection")
        best_features = 7
        tr_features_scaled, features_index_sorted = run_feature_selection(tr_features_scaled, tr_labels,
                                                                          feature_selection, best_features)
        eval_features_scaled = eval_features_scaled[:, features_index_sorted[0:best_features]]

    # 1)
    if pca_k > 0:
        # Load if exists??
        # NOTE PCA do memory error!
        print("Runing IPCA " + str(pca_k) + "...")

        # NOTE 11 Enero: TEST WITH IPCA!!!!!!
        start = time.time()

        # Run PCA
        IPCA = sklearn.decomposition.IncrementalPCA(pca_k, batch_size=pca_k)  # gamma_pca

        # tr_features_scaled = KPCA.fit_transform(tr_features_scaled)
        IPCA.fit(tr_features_scaled)

        # Apply PCA on test data!
        tr_features_scaled = IPCA.transform(tr_features_scaled)
        eval_features_scaled = IPCA.transform(eval_features_scaled)

        """
        print("Runing TruncatedSVD (singular value decomposition (SVD)!!!) (alternative to PCA) " + str(pca_k) + "...")

        svd = decomposition.TruncatedSVD(n_components=pca_k, algorithm='arpack')
        svd.fit(tr_features_scaled)
        tr_features_scaled = svd.transform(tr_features_scaled)
        eval_features_scaled = svd.transform(eval_features_scaled)

        """
        end = time.time()

        print("Time runing IPCA (rbf): " + str(format(end - start, '.2f')) + " sec")

    ################################################################################################
    # 3) Train SVM model
    if not do_cross_val:
        train_and_eval_svm_model(tr_features_scaled,
                                 tr_labels,
                                 eval_features_scaled,
                                 eval_labels,
                                 multi_mode,
                                 winL,
                                 winR,
                                 do_preprocess,
                                 maxRR,
                                 use_RR,
                                 norm_RR,
                                 compute_morph,
                                 use_weight_class,
                                 feature_selection,
                                 oversamp_method,
                                 leads_flag,
                                 reduced_DS,
                                 pca_k,
                                 gamma_value,
                                 C_value,
                                 )

    # 3. run cross validation
    # 2) Cross-validation:
    else:
        # todo: this cross validation sucks, should not seperate train and eval before cross validation.
        # todo: should not only use training data to do cross validation.

        print("Runing cross val...")
        start = time.time()

        # k: number of folds
        k = len(tr_patient_num_beats) if do_cross_val == "pat_cv" else 5

        cv_scores, c_values = run_cross_val(tr_features_scaled,
                                            tr_labels,
                                            tr_patient_num_beats,
                                            do_cross_val,
                                            k)

        perf_measures_path = create_svm_model_name(
            model_svm_path='/home/congyu/ECG/model_train_log/ecg_classification/results/' + multi_mode,
            delimiter='_', **params_for_naming)

        if do_cross_val == 'pat_cv':  # Cross validation with one fold per patient
            # TODO implement this method! check to avoid NaN scores....
            np.savetxt(perf_measures_path + '/cross_val_k-pat_cv_F_score.csv',
                       (c_values, cv_scores.astype(float)), "%f")

        elif do_cross_val == 'beat_cv':  # cross validation by class id samples
            # TODO Save data over the k-folds and ranked by the best average values in separated files
            np.savetxt(perf_measures_path + '/cross_val_k-' + str(k) + '_Ijk_score.csv',
                       (c_values, cv_scores.astype(float)), "%f")

        end = time.time()
        print("Time runing Cross Validation: " + str(format(end - start, '.2f')) + " sec")


def train_and_eval_svm_model(
        tr_features_scaled,
        tr_labels,
        eval_features_scaled,
        eval_labels,
        multi_mode,
        winL,
        winR,
        do_preprocess,
        maxRR,
        use_RR,
        norm_RR,
        compute_morph,
        use_weight_class,
        feature_selection,
        oversamp_method,
        leads_flag,
        reduced_DS,
        pca_k,
        gamma_value,
        C_value,
):
    # TODO load best params from cross validation!

    # train the model
    model_svm_path = path_to_model(**locals())

    print("Training model on MIT-BIH DS1: " + model_svm_path + "...")
    svm_model = train_model_module(model_svm_path,
                                   tr_features_scaled,
                                   tr_labels,
                                   C_value=C_value,
                                   gamma_value=gamma_value,
                                   multi_mode=multi_mode)

    # evaluate the model
    print("Testing model on MIT-BIH DS2: " + model_svm_path + "...")
    perf_measures_path = path_to_measure(**locals())

    eval_model_module(svm_model,
                      tr_features_scaled, tr_labels,
                      eval_features_scaled, eval_labels,
                      multi_mode,
                      perf_measures_path,
                      C_value,
                      gamma_value)


if __name__ == "__main__":
    trival_main(
        multi_mode="ovo",
        winL=90,
        winR=90,
        do_preprocess=True,
        maxRR=True,
        use_RR=False,
        norm_RR=True,
        compute_morph=['resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'],
        reduced_DS=True,
        leads_flag=[1, 0],
    )

    """
    main(
        multi_mode="ovr",
        winL=90,
        winR=90,
        do_preprocess=False,
        maxRR=False,
        use_RR=False,
        norm_RR=False,
        compute_morph=['resample_10', 'lbp', 'hbf5', 'wvlt', 'HOS'],
        reduced_DS=True,
        leads_flag=[1, 0],
        do_cross_val="pat_cv",
    )
    """
