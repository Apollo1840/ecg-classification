import os
from constant import DB_PATH


def create_svm_model_name(model_svm_path, winL, winR, do_preprocess,
                          maxRR, use_RR, norm_RR, compute_morph, use_weight_class, feature_selection,
                          oversamp_method, leads_flag, reduced_DS, pca_k, delimiter="/", prepare=True, **kwargs):
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

    if prepare:
        if not os.path.exists(model_svm_path):
            os.makedirs(model_svm_path)

    return model_svm_path


def get_svm_model_path(multi_mode,
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
                       C_value,
                       **kwargs):
    model_svm_path = DB_PATH + 'svm_models/' + multi_mode + '_rbf'

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


def name_my_db(reduced_DS, do_preprocess, winL, winR, DS):
    """

    :param reduced_DS:
    :param do_preprocess:
    :param winL:
    :param winR:
    :param DS:
    :return: str
    """

    mit_pickle_name = DB_PATH + 'python_mit'
    if reduced_DS:
        mit_pickle_name = mit_pickle_name + '_reduced_'

    if do_preprocess:
        mit_pickle_name = mit_pickle_name + '_rm_bsline'

    mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR)
    mit_pickle_name = mit_pickle_name + '_' + DS + '.pkl'

    return mit_pickle_name


def name_ml_data(DS, ws, do_preprocess, maxRR, use_RR, norm_RR,
                 compute_morph, leads_flag, prepare=True, **kwargs):
    """


    :param DS:
    :param do_preprocess:
    :param maxRR:
    :param use_RR:
    :param norm_RR:
    :param compute_morph:
    :param leads_flag:
    :return: Str, eg. "..._v1.p"
    """

    winL, winR = ws
    reduced_DS = True if leads_flag == [1, 1] else False
    features_labels_name = DB_PATH + 'features/' + 'w_' + str(winL) + '_' + str(winR) + '_' + DS

    if do_preprocess:
        features_labels_name += '_rm_bsline'

    if maxRR:
        features_labels_name += '_maxRR'

    if use_RR:
        features_labels_name += '_RR'

    if norm_RR:
        features_labels_name += '_norm_RR'

    for descp in compute_morph:
        features_labels_name += '_' + descp

    if reduced_DS:
        features_labels_name += '_reduced'

    if leads_flag[0] == 1:
        features_labels_name += '_MLII'

    if leads_flag[1] == 1:
        features_labels_name += '_V1'

    features_labels_name += '.pkl'

    if prepare:
        if not os.path.exists(os.path.dirname(features_labels_name)):
            os.makedirs(os.path.dirname(features_labels_name))

    return features_labels_name
