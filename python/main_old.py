#!/usr/bin/env python

import numpy as np
import time
import sklearn
from sklearn.preprocessing import StandardScaler

from constant import *
from path_manager import path_to_measure, path_to_model
from train_SVM import (load_mit_db, eval_model_module, create_oversamp_name, run_cross_val, create_svm_model_name,
                       perform_oversampling, run_feature_selection, train_model_module)


def main(
        multi_mode='ovo',
        winL=90,
        winR=90,
        do_preprocess=True,
        use_weight_class=True,
        maxRR=True,
        use_RR=True,
        norm_RR=True,
        compute_morph=[""],
        oversamp_method='',
        pca_k=0,
        feature_selection="",
        do_cross_val="",
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
    :param feature_selection: Str, 'select_K_Best' or 'LassoCV' or 'slct_percentile'
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
                                                               reduced_DS,
                                                               maxRR,
                                                               use_RR,
                                                               norm_RR,
                                                               compute_morph)

    # Load Test data
    # eval_ means evaluation
    eval_features, eval_labels, eval_patient_num_beats = load_mit_db('DS2',
                                                                     (winL, winR),
                                                                     do_preprocess,
                                                                     reduced_DS,
                                                                     maxRR,
                                                                     use_RR,
                                                                     norm_RR,
                                                                     compute_morph)

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

    # Call different configurations for train_SVM.py

    ####################################################################################
    winL = 90
    winR = 90
    do_preprocess = True
    use_weight_class = True
    maxRR = True
    compute_morph = {''}  # 'wvlt', 'HOS', 'myMorph'

    multi_mode = 'ovo'
    voting_strategy = 'ovo_voting'  # 'ovo_voting_exp', 'ovo_voting_both'

    use_RR = False
    norm_RR = False

    oversamp_method = ''
    feature_selection = ''
    do_cross_val = ''
    C_value = 0.001
    reduced_DS = False  # To select only patients in common with MLII and V1
    leads_flag = [1, 0]  # MLII, V1

    pca_k = 0

    ################

    # With feature selection
    ov_methods = {''}  # , 'SMOTE_regular'}

    C_values = {0.001, 0.01, 0.1, 1, 10, 100}
    gamma_values = {0.0}
    gamma_value = 0.0
    for C_value in C_values:
        pca_k = 0

        # Single
        use_RR = False
        norm_RR = False
        compute_morph = {'u-lbp'}
        main(multi_mode,
             winL,
             winR,
             do_preprocess,
             use_weight_class,
             maxRR, use_RR, norm_RR,
             compute_morph,
             oversamp_method,
             pca_k,
             feature_selection,
             do_cross_val,
             C_value,
             gamma_value,
             reduced_DS,
             leads_flag)

        """
        # Two
        use_RR = True
        norm_RR = True
        compute_morph = {'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
        use_RR = False
        norm_RR = False
    
        compute_morph = {'wvlt', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
                
        compute_morph = {'HOS', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
    
        compute_morph = {'myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
                 
    
    
        # Three
        use_RR = True
        norm_RR = True
        compute_morph = {'wvlt', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
                
        compute_morph = {'HOS', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
    
        compute_morph = {'myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
        use_RR = False
        norm_RR = False
        compute_morph = {'wvlt','HOS', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
                
        compute_morph = {'wvlt','myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
    
        compute_morph = {'HOS','myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
    
    
        # four
        use_RR = True
        norm_RR = True
        compute_morph = {'wvlt', 'HOS', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
    
        compute_morph = {'wvlt', 'myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
    
        compute_morph = {'HOS','myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
        use_RR = False
        norm_RR = False
        compute_morph = {'wvlt', 'HOS','myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
         
    
    
        # five
        use_RR = True
        norm_RR = True
        compute_morph = {'wvlt', 'HOS','myMorph', 'u-lbp'} 
        main(multi_mode, 90, 90, do_preprocess, use_weight_class, maxRR, use_RR, norm_RR, compute_morph, oversamp_method, pca_k, feature_selection, do_cross_val, C_value, gamma_value, reduced_DS, leads_flag)
                 
        """
