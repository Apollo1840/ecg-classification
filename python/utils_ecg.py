from config import *
import operator
from scipy.signal import medfilt
import matplotlib.pyplot as plt


# Show a 2D plot with the data in beat
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()


def display_beat(beat_index, ref_sig, r_pos_original=False):
    beat_start, r_pos, beat_end = beat_index
    plt.plot(ref_sig[beat_start:beat_end])
    plt.plot(r_pos-beat_start, ref_sig[r_pos], "xr")
    if r_pos_original:
        plt.plot(r_pos_original - beat_start, ref_sig[r_pos_original], "xb")
    plt.ylabel('Signal')
    plt.show()


def parse_annotations(annotations, ref_sig, ws=(90, 90), size_rr_max=20):
    """

    :param annotations: List[Tuple], each tuple is (int, str) stands for r_pos, beat_label
    :param ref_sig: reference_signal
    :param ws: Tuple[int], window size (winL, winR)
    :param size_rr_max: int
    :return:
        beat_indices, List[Tuple], Tuple is (beat_start_index, r_peak_index, beat_end_index).
        labels, List[int],
    """

    beat_indices = []
    labels = []
    r_peaks_original = []
    r_peaks = []
    is_r_valid = []
    invalid_reasons = []

    for r_pos, beat_label in annotations:
        r_peaks_original.append(r_pos)

        beat_index, label, invalid_reason = parse_beat(r_pos, beat_label, ref_sig, ws, rr_max=size_rr_max)
        _, r_pos, _ = beat_index
        r_peaks.append(r_pos)

        if label:
            beat_indices.append(beat_index)
            labels.append(label)

            is_r_valid.append(True)
        else:
            is_r_valid.append(False)
            invalid_reasons.append(invalid_reason)

    # definitly:
    assert len(r_peaks) == len(r_peaks_original) == len(is_r_valid)

    return beat_indices, labels, r_peaks_original, r_peaks, is_r_valid, invalid_reasons


BEAT_LABEL_ERROR = "beat_type not in mitbih_class"
BEAT_CLASS_ERROR = "beat_type not in AAMI"
BEAT_EDGE_ERROR = "beat_boudary out of signal"


def parse_beat(r_pos, beat_type, ref_sig, ref_ws, is_relocate=True, rr_max=20):
    """
    relocate r_peak, get beat index and AAMI class

    :param r_pos: int, r-peak index
    :param beat_type: str, mitbih label.
    :param ref_sig: List[float], reference_signal
    :param ref_ws: Tuple[int], (winL, winR), reference window size
    :param is_relocate: Bool
    :param rr_max: int, related to sample rate
    :return: Tuple, str, str is AAMI label.
    """

    if is_relocate:
        r_pos = relocate_r_peak(r_pos, ref_sig, rr_max)

    winL, winR = ref_ws
    class_AAMI = mitbih2aami(beat_type)

    # condition 1: r_pos is feasible region
    # condition 2: beat_type is accepted
    error_source = None
    if beat_type not in MITBIH_CLASSES:
        error_source = BEAT_LABEL_ERROR
    elif class_AAMI is None:
        error_source = BEAT_CLASS_ERROR
    elif not winL < r_pos < (len(ref_sig) - winR):
        error_source = BEAT_EDGE_ERROR

    if error_source is None:
        beatL = r_pos - winL
        beatR = r_pos + winR
        return (beatL, r_pos, beatR), class_AAMI, error_source
    else:
        return (None, r_pos, None), None, error_source


def relocate_r_peak(r_pos, ref_signal, rr_max=20):
    """

    :param r_pos: int, r_peak index
    :param ref_signal: List[float], reference signal
    :param rr_max: int,
    :return:
    """

    # relocate r_peak by searching maximum in ref_signal
    # r_pos between [size_RR_max, len(MLII) - size_RR_max]
    if rr_max < r_pos < len(ref_signal) - rr_max:
        index, value = max(enumerate(ref_signal[r_pos - rr_max: r_pos + rr_max]), key=operator.itemgetter(1))
        return (r_pos - rr_max) + index
    return r_pos


def preprocess_sig(signal, verbose=False):
    """

    :param signal: List[float]
    :param verbose: Bool
    :return: List[float]
    """
    if verbose:
        print("filtering the signal ...")

    baseline = medfilt(signal, 71)
    baseline = medfilt(baseline, 215)

    # Remove Baseline
    for i in range(0, len(signal)):
        signal[i] = signal[i] - baseline[i]

    # TODO Remove High Freqs
    return signal


def mitbih2aami(label):
    """

    :param label: str
    :return: str
    """

    for aami_label, mitbih_label in AAMI.items():
        if label in mitbih_label:
            return aami_label
    return None
