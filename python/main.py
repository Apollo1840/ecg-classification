from tqdm import tqdm
from train_svm_fast import train_and_evaluation


def model_search_unit():
    fixed_parameters = {
        "multi_mode": "ovo",
        "winL": 90,
        "winR": 90,
        "do_preprocess": True,
        "use_weight_class": True,
        "maxRR": True,
        "use_RR": True,
        "norm_RR": True,
        "oversamp_method": "",
        "cross_patient": True,
        "reduced_DS": False,
        "verbose": True,
    }

    searchable_params = {
        "c_value": 0.1,
        "gamma_value": 0.0,
        "compute_morph": ['wvlt', 'HOS', 'u-lbp', 'OurMorph'],
    }

    train_and_evaluation(**fixed_parameters, **searchable_params)


def hypersearch():
    fixed_parameters = {
        "multi_mode": "ovo",
        "winL": 90,
        "winR": 90,
        "do_preprocess": True,
        "use_weight_class": True,
        "maxRR": True,
        "use_RR": True,
        "norm_RR": True,
        "oversamp_method": "",
        "cross_patient": True,
        "reduced_DS": False,
        "verbose": True,
    }

    searchable_params = {
        "c_value": 0.1,
        "gamma_value": 0.0,
        "compute_morph": ['wvlt', 'HOS', 'u-lbp', 'OurMorph'],
    }

    # c_values = [10, 50, 80, 100, 120, 150]
    # c_values = [0.1, 1, 5, 10, 50, 100, 500, 1000]
    c_values = [1, 10, 100]
    cross_patient = [True, False]

    for c_value in tqdm(c_values):
        for cross in cross_patient:
            searchable_params["c_value"] = c_value
            fixed_parameters["cross_patient"] = cross
            train_and_evaluation(**fixed_parameters, **searchable_params)


if __name__ == "__main__":
    hypersearch()
