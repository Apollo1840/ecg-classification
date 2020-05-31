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

from sklearn import decomposition

from load_MITBIH import load_mit_db
from evaluation_AAMI import *
from aggregation_voting_strategies import *
from oversampling import *
from cross_validation import *
from feature_selection import *


def create_svm_model_name(model_svm_path, winL, winR, do_preprocess,
                          maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection,
                          oversamp_method, leads_flag, reduced_DS, pca_k, delimiter="/"):
    """

    :param model_svm_path:
    :param winL:
    :param winR:
    :param do_preprocess:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param use_weight_class:
    :param feature_selection:
    :param oversamp_method:
    :param leads_flag:
    :param reduced_DS:
    :param pca_k:
    :param delimiter:
    :return: Str
    """

    if reduced_DS:
        model_svm_path = model_svm_path + delimiter + 'exp_2'

    if leads_flag[0] == 1:
        model_svm_path = model_svm_path + delimiter + 'MLII'

    if leads_flag[1] == 1:
        model_svm_path = model_svm_path + delimiter + 'V1'

    if oversamp_method:
        model_svm_path = model_svm_path + delimiter + oversamp_method

    if feature_selection:
        model_svm_path = model_svm_path + delimiter + feature_selection

    if do_preprocess:
        model_svm_path = model_svm_path + delimiter + 'rm_bsln'

    if maxRR:
        model_svm_path = model_svm_path + delimiter + 'maxRR'

    if use_RR:
        model_svm_path = model_svm_path + delimiter + 'RR'

    if norm_RR:
        model_svm_path = model_svm_path + delimiter + 'norm_RR'

    for descp in compute_morph:
        model_svm_path = model_svm_path + delimiter + descp

    if use_weight_class:
        model_svm_path = model_svm_path + delimiter + 'weighted'

    if pca_k > 0:
        model_svm_path = model_svm_path + delimiter + 'pca_' + str(pca_k)

    return model_svm_path


def train_model(model_svm_path, tr_features_scaled, tr_labels, verbose=False, **model_kwargs):
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
    start = time.time()
    svm_model.fit(tr_features_scaled, tr_labels)
    end = time.time()

    # TODO assert that the class_ID appears with the desired order,
    # with the goal of ovo make the combinations properly
    print("Trained completed!\n\t" + model_svm_path + "\n \
        \tTime required: " + str(format(end - start, '.2f')) + " sec")

    if model_svm_path:
        # Export model: save/write trained SVM model
        if not os.path.exists(os.path.dirname(model_svm_path)):
            os.makedirs(os.path.dirname(model_svm_path))
        joblib.dump(svm_model, model_svm_path)

    # TODO Export StandardScaler()

    return svm_model


def train_model_if_needed(model_svm_path, tr_features_scaled, tr_labels, C_value, gamma_value, multi_mode):
    if os.path.isfile(model_svm_path):
        # Load the trained model!
        svm_model = joblib.load(model_svm_path)

    else:
        model_kwargs = {
            "c_value": C_value,
            "gamma_value": gamma_value,
            "multi_mode": multi_mode,
        }
        svm_model = train_model(model_svm_path, tr_features_scaled, tr_labels, **model_kwargs)
    return svm_model


def get_svm_model_path(db_path, multi_mode,
                       winL, winR,
                       do_preprocess,
                       maxRR,
                       use_RR, norm_RR,
                       compute_morph,
                       use_weight_class,
                       feature_selection,
                       oversamp_method,
                       leads_flag,
                       reduced_DS,
                       pca_k,
                       gamma_value,
                       C_value
                       ):
    model_svm_path = db_path + 'svm_models/' + multi_mode + '_rbf'

    model_svm_path = create_svm_model_name(model_svm_path,
                                           winL, winR,
                                           do_preprocess,
                                           maxRR, use_RR, norm_RR,
                                           compute_morph,
                                           use_weight_class,
                                           feature_selection,
                                           oversamp_method,
                                           leads_flag,
                                           reduced_DS,
                                           pca_k,
                                           '_')

    if gamma_value != 0.0:
        model_svm_path = model_svm_path + '_C_' + str(C_value) + '_g_' + str(gamma_value) + '.joblib.pkl'
    else:
        model_svm_path = model_svm_path + '_C_' + str(C_value) + '.joblib.pkl'

    return model_svm_path


def eval_model_module(svm_model,
                      tr_features_scaled,
                      tr_labels,
                      eval_features_scaled,
                      eval_labels,
                      multi_mode,
                      perf_measures_path,
                      C_value,
                      gamma_value,
                      is_include_train=True,
                      is_include_ovo_voting_exp=True
                      ):
    # ovo_voting:
    # Simply add 1 to the win class
    if is_include_train:
        print("Evaluation on DS1 ...")
        eval_model(svm_model,
                   tr_features_scaled,
                   tr_labels,
                   multi_mode,
                   'ovo_voting', perf_measures_path, C_value,
                   gamma_value, 'Train_')

    # Let's test new data!
    print("Evaluation on DS2 ...")
    eval_model(svm_model,
               eval_features_scaled,
               eval_labels,
               multi_mode,
               'ovo_voting', perf_measures_path, C_value,
               gamma_value, '')

    # ovo_voting_exp:
    # Consider the post prob adding to both classes
    if is_include_ovo_voting_exp:
        if is_include_train:
            print("Evaluation on DS1 ...")
            eval_model(svm_model,
                       tr_features_scaled,
                       tr_labels,
                       multi_mode,
                       'ovo_voting_exp', perf_measures_path, C_value,
                       gamma_value, 'Train_')

        # Let's test new data!
        print("Evaluation on DS2 ...")
        eval_model(svm_model,
                   eval_features_scaled,
                   eval_labels,
                   multi_mode,
                   'ovo_voting_exp', perf_measures_path,
                   C_value, gamma_value, '')


# Eval the SVM model and export the results
def eval_model(svm_model, features, labels, multi_mode, voting_strategy, output_path, C_value, gamma_value, DS):

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
    np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_decision_ovr.csv',
               decision_ovr)
    np.savetxt(output_path + '/' + DS + 'C_' + str(C_value) + '_predict_' + voting_strategy + '.csv',
               predict_ovr.astype(int), '%.0f')


def write_AAAMI_results_gamma(output_path, gamma_value, perf_measures, C_value, voting_strategy, DS):

    print(perf_measures.__dict__)

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


def create_oversamp_name(reduced_DS, do_preprocess, compute_morph, winL, winR, maxRR, use_RR, norm_RR, pca_k):
    oversamp_features_pickle_name = ''
    if reduced_DS:
        oversamp_features_pickle_name += '_reduced_'

    if do_preprocess:
        oversamp_features_pickle_name += '_rm_bsline'

    if maxRR:
        oversamp_features_pickle_name += '_maxRR'

    if use_RR:
        oversamp_features_pickle_name += '_RR'

    if norm_RR:
        oversamp_features_pickle_name += '_norm_RR'

    for descp in compute_morph:
        oversamp_features_pickle_name += '_' + descp

    if pca_k > 0:
        oversamp_features_pickle_name += '_pca_' + str(pca_k)

    oversamp_features_pickle_name += '_wL_' + str(winL) + '_wR_' + str(winR)

    return oversamp_features_pickle_name


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
    :param feature_selection: Bool,
    :param do_cross_val: Str, 'pat_cv' or 'beat_cv'
    :param C_value:
    :param gamma_value:
    :param reduced_DS: Bool,
    :param leads_flag: List[int], [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :return:
    """

    print("Runing train_SVM.py!")

    # 1. load data

    # store the intermidiary data
    db_path = '/home/congyu/dataset/ECG/mitdb/ml_learning/'

    # Load train data
    # tr_ means train_
    tr_features, tr_labels, tr_patient_num_beats = load_mit_db('DS1',
                                                               winL, winR,
                                                               do_preprocess,
                                                               maxRR, use_RR, norm_RR,
                                                               compute_morph, db_path,
                                                               reduced_DS, leads_flag)

    # Load Test data
    eval_features, eval_labels, eval_patient_num_beats = load_mit_db('DS2',
                                                                     winL, winR,
                                                                     do_preprocess,
                                                                     maxRR, use_RR, norm_RR,
                                                                     compute_morph, db_path,
                                                                     reduced_DS, leads_flag)

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
        tr_features, tr_labels = perform_oversampling(oversamp_method, db_path + 'oversamp/python_mit',
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
        # TODO load best params from cross validation!

        model_svm_path = get_svm_model_path(
            db_path,
            multi_mode,
            winL,
            winR,
            do_preprocess,
            maxRR,
            use_RR, norm_RR,
            compute_morph,
            use_weight_class,
            feature_selection,
            oversamp_method,
            leads_flag,
            reduced_DS,
            pca_k,
            gamma_value,
            C_value
        )

        print("Training model on MIT-BIH DS1: " + model_svm_path + "...")
        svm_model = train_model_if_needed(model_svm_path,
                                          tr_features_scaled,
                                          tr_labels,
                                          C_value,
                                          gamma_value,
                                          multi_mode)

        # 4) Test SVM model
        print("Testing model on MIT-BIH DS2: " + model_svm_path + "...")

        # Evaluate the model on the training data
        perf_measures_path = create_svm_model_name(
            '/home/congyu/ECG/model_train_log/ecg_classification/results/' + multi_mode, winL, winR,
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
            '/')

        eval_model_module(svm_model,
                          tr_features_scaled, tr_labels,
                          eval_features_scaled, eval_labels,
                          multi_mode,
                          perf_measures_path,
                          C_value, gamma_value)

    # 3. run cross validation
    # 2) Cross-validation:
    else:
        print("Runing cross val...")
        start = time.time()

        # TODO Save data over the k-folds and ranked by the best average values in separated files   
        perf_measures_path = create_svm_model_name(
            '/home/congyu/ECG/model_train_log/ecg_classification/results/' + multi_mode,
            winL, winR,
            do_preprocess,
            maxRR, use_RR, norm_RR,
            compute_morph,
            use_weight_class,
            feature_selection,
            oversamp_method,
            leads_flag,
            reduced_DS,
            pca_k,
            '/')

        # TODO implement this method! check to avoid NaN scores....

        if do_cross_val == 'pat_cv':  # Cross validation with one fold per patient
            cv_scores, c_values = run_cross_val(tr_features_scaled,
                                                tr_labels,
                                                tr_patient_num_beats,
                                                do_cross_val,
                                                len(tr_patient_num_beats))

            if not os.path.exists(perf_measures_path):
                os.makedirs(perf_measures_path)

            np.savetxt(perf_measures_path + '/cross_val_k-pat_cv_F_score.csv',
                       (c_values, cv_scores.astype(float)), "%f")

        elif do_cross_val == 'beat_cv':  # cross validation by class id samples
            k_folds = {5}
            for k in k_folds:
                ijk_scores, c_values = run_cross_val(tr_features_scaled,
                                                     tr_labels,
                                                     tr_patient_num_beats,
                                                     do_cross_val,
                                                     k)

                # TODO Save data over the k-folds and ranked by the best average values in separated files
                perf_measures_path = create_svm_model_name(
                    '/home/congyu/ECG/model_train_log/ecg_classification/results/' + multi_mode,
                    winL, winR,
                    do_preprocess,
                    maxRR, use_RR, norm_RR,
                    compute_morph,
                    use_weight_class,
                    feature_selection,
                    oversamp_method,
                    leads_flag,
                    reduced_DS,
                    pca_k,
                    '/')

                if not os.path.exists(perf_measures_path):
                    os.makedirs(perf_measures_path)
                np.savetxt(perf_measures_path + '/cross_val_k-' + str(k) + '_Ijk_score.csv',
                           (c_values, ijk_scores.astype(float)), "%f")

            end = time.time()
            print("Time runing Cross Validation: " + str(format(end - start, '.2f')) + " sec")
        else:
            print("You must specify which kind of crossval type you use in do_cross_val, "
                  "eg do_cross_val='pat_cv' or 'beat_cv'.")


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
        C_value=0.001,
        gamma_value=0.0,
        reduced_DS=False,
        leads_flag=[1, 0]
):
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
    :param feature_selection: Bool,
    :param do_cross_val: Str, 'pat_cv' or 'beat_cv'
    :param C_value:
    :param gamma_value:
    :param reduced_DS: Bool,
    :param leads_flag: List[int], [MLII, V1] set the value to 0 or 1 to reference if that lead is used
    :return:
    """

    print("Runing train_SVM.py!")

    # 1. load data

    # store the intermidiary data
    db_path = '/home/congyu/dataset/ECG/mitdb/ml_learning/'

    # Load train data
    # tr_ means train_
    tr_features, tr_labels, tr_patient_num_beats = load_mit_db(
        'DS1',
        winL,
        winR,
        do_preprocess,
        maxRR,
        use_RR,
        norm_RR,
        compute_morph,
        db_path,
        reduced_DS,
        leads_flag
    )

    # Load Test data
    eval_features, eval_labels, eval_patient_num_beats = load_mit_db(
        'DS2',
        winL,
        winR,
        do_preprocess,
        maxRR,
        use_RR,
        norm_RR,
        compute_morph,
        db_path,
        reduced_DS,
        leads_flag
    )

    scaler = StandardScaler()
    scaler.fit(tr_features)
    tr_features_scaled = scaler.transform(tr_features)

    # scaled: zero mean unit variance ( z-score )
    eval_features_scaled = scaler.transform(eval_features)

    model_svm_path = get_svm_model_path(
        db_path,
        multi_mode,
        winL,
        winR,
        do_preprocess,
        maxRR,
        use_RR, norm_RR,
        compute_morph,
        use_weight_class,
        feature_selection,
        oversamp_method,
        leads_flag,
        reduced_DS,
        pca_k,
        gamma_value,
        C_value
    )

    print("Training model on MIT-BIH DS1: " + model_svm_path + "...")
    svm_model = train_model_if_needed(model_svm_path,
                                      tr_features_scaled,
                                      tr_labels,
                                      C_value,
                                      gamma_value,
                                      multi_mode
                                      )

    print("Testing model on MIT-BIH DS2: " + model_svm_path + "...")
    perf_measures_path = create_svm_model_name(
        '/home/congyu/ECG/model_train_log/ecg_classification/results/' + multi_mode, winL, winR,
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
        '/')

    eval_model_module(svm_model,
                      tr_features_scaled, tr_labels,
                      eval_features_scaled, eval_labels,
                      multi_mode,
                      perf_measures_path,
                      C_value,
                      gamma_value)

    print("congrats! evaluation complete! ")


if __name__ == "__main__":
    """
        trival_main(
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
        do_cross_val="beat_cv",
    )
