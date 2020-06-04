
DATA_DIR = '/home/congyu/dataset/ECG/mitbih/csv/'
DB_PATH = "/home/congyu/dataset/ECG/mitdb/ml_learning/"
MEASURE_PATH = "/home/congyu/ECG/model_train_log/ecg_classification/results/"

MITBIH_CLASSES = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']  # , 'P', '/', 'f', 'u']
AAMI = {
    "N": ['N', 'L', 'R'],
    "SVEB": ['A', 'a', 'J', 'S', 'e', 'j'],
    "VEB": ['V', 'E'],
    "F": ['F'],
    # "Q": ['P', '/', 'f', 'u'],
}

AAMI_CLASSES = sorted(AAMI.keys())
# AAMI_CLASSES = ['F', 'N', 'SVEB', 'VEB']

DS_bank = {

    # ML-II
    "normal": {
        "DS1": [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                223, 230],
        "DS2": [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                233, 234]
    },

    # ML-II + V1
    "reduced": {
        "DS1": [101, 106, 108, 109, 112, 115, 118, 119, 201, 203,
                205, 207, 208, 209, 215, 220, 223, 230],
        "DS2": [105, 111, 113, 121, 200, 202, 210, 212, 213, 214,
                219, 221, 222, 228, 231, 232, 233, 234]
    }
}

for bank_key in DS_bank:
    DS_bank[bank_key]["DS12"] = DS_bank[bank_key]["DS1"] + DS_bank[bank_key]["DS2"]

