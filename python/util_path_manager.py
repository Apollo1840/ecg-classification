import os
from config import DB_PATH, MEASURE_PATH


def create_svm_model_name(somename,
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
                          delimiter="/",
                          **kwargs):
    """
    mark the name with all the hyperparameters, joined by delimiter.


    :param somename:
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
        somename = somename + delimiter + 'exp_2'

    if leads_flag[0] == 1:
        somename = somename + delimiter + 'MLII'

    if leads_flag[1] == 1:
        somename = somename + delimiter + 'V1'

    if oversamp_method:
        somename = somename + delimiter + oversamp_method

    if feature_selection:
        somename = somename + delimiter + feature_selection

    if do_preprocess:
        somename = somename + delimiter + 'rm_bsln'

    if maxRR:
        somename = somename + delimiter + 'maxRR'

    if use_RR:
        somename = somename + delimiter + 'RR'

    if norm_RR:
        somename = somename + delimiter + 'norm_RR'

    for descp in compute_morph:
        somename = somename + delimiter + descp

    if use_weight_class:
        somename = somename + delimiter + 'weighted'

    if pca_k > 0:
        somename = somename + delimiter + 'pca_' + str(pca_k)

    return somename


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


def path_to_model(multi_mode,
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
                  c_value,
                  delimiter="_",
                  cross_patient=False,
                  **kwargs):

    dirpath = os.path.join(DB_PATH, "svm_models")
    model_name = multi_mode + '_rbf'
    model_name = create_svm_model_name(somename=model_name, **locals())

    model_postfix = '_C_' + str(c_value)
    if gamma_value:
        model_postfix += '_g_' + str(gamma_value)
    if cross_patient:
        model_postfix += '_crossp'

    model_full_name = model_name + model_postfix + '.joblib.pkl'

    path = os.path.join(dirpath, model_full_name)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    return path


def path_to_measure(multi_mode,
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
                    delimiter="_",
                    **kwargs):

    perf_measures_name = create_svm_model_name(somename=multi_mode, **locals())
    path = os.path.join(MEASURE_PATH, perf_measures_name)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    return path


def path_to_my_db(DS, is_reduce, ws, do_preprocess, **kwargs):
    """

    :param is_reduce:
    :param do_preprocess:
    :param ws: Tuple[int], (winL, winR)
    :param DS:
    :return: str
    """

    winL, winR = ws

    mit_pickle_name = DB_PATH + 'python_mit'
    if is_reduce:
        mit_pickle_name = mit_pickle_name + '_reduced_'

    if do_preprocess:
        mit_pickle_name = mit_pickle_name + '_rm_bsline'

    mit_pickle_name = mit_pickle_name + '_wL_' + str(winL) + '_wR_' + str(winR)
    mit_pickle_name = mit_pickle_name + '_' + DS + '.pkl'

    return mit_pickle_name


def path_to_ml_data(DS,
                    ws,
                    do_preprocess,
                    is_reduce,
                    maxRR,
                    use_RR,
                    norm_RR,
                    compute_morph,
                    prepare=True,
                    **kwargs):
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

    if is_reduce:
        features_labels_name += '_reduced'

    if True:
        features_labels_name += '_MLII'

    if is_reduce:
        features_labels_name += '_V1'

    features_labels_name += '.pkl'

    if prepare:
        if not os.path.exists(os.path.dirname(features_labels_name)):
            os.makedirs(os.path.dirname(features_labels_name))

    return features_labels_name
