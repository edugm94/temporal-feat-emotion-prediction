#!/usr/bin/env python#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.04.12
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.04.14
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden

import numpy as np
from statsmodels import robust
from src.d00_utils.featuring_tools import means_absolute_values, prctile
import pyhrv.frequency_domain as fd


def get_features(hr, freq=1):
    """
    :param hr: Vector slice from a fixed window to extract features from
    :param freq: Frequency of the signal
    :return:
    """
    features = []

    # Calculation of time-domain features
    # Feat 1: mean
    hr_mean = np.mean(hr)
    features.extend([hr_mean])

    # Feat 2: Std
    hr_std = np.std(hr, ddof=1)
    features.extend([hr_std])

    # Feat 3: Max
    hr_max = np.max(hr)
    features.extend([hr_max])

    # Feat 4: Min
    hr_min = np.min(hr)
    features.extend([hr_min])

    # Feat 5: P90
    hr_p90 = prctile(hr, 90)
    features.extend([hr_p90])

    # Feat 6: Variance
    hr_var = np.var(hr, ddof=1)
    features.extend([hr_var])

    # Feat 7: MAD
    hr_mad = np.mean(np.abs(hr-np.mean(hr)))
    features.extend([hr_mad])

    # Feat 8: Norm
    hr_norm = np.linalg.norm(hr)
    features.extend([hr_norm])

    # Feat 9 - 12: MAVFD, MAVFDN, MAVFSD, MAVFSDN
    hr_dif1_m, hr_dif2_m, hr_dif1_norm_m, hr_dif2_norm_m = means_absolute_values(hr)
    features.extend([hr_dif1_m, hr_dif2_m, hr_dif1_norm_m, hr_dif2_norm_m])

    # Feat 13 & 14: Smooth signal
    hann_ = np.hanning(len(hr))   # Smoothing window
    hr_smooth = np.multiply(hr, hann_)
    hr_smooth_mean = np.mean(hr_smooth)

    hr_1dif = hr[1:] - hr[0:-1]
    hr_1dif_m = np.mean(hr_1dif)
    features.extend([hr_smooth_mean, hr_1dif_m])

    # Calculation of Frequency Domain features
    bands = {
        "ulf": None,
        "vlf": (0, 0.16),
        "lf": (0.16, 0.6),
        "hf": (0.7, 3)
    }
    ibi = 60 / hr   # Conversion from HR to IBI (in seconds). It is needed for the below function
    freq_features = fd.welch_psd(nni=ibi, fbands=bands, mode='dev')

    aVLF = freq_features[0]['fft_abs'][1]
    aLF = freq_features[0]['fft_abs'][2]
    aHLF = freq_features[0]['fft_abs'][3]
    aTotal = freq_features[0]['fft_total']
    features.extend([aVLF, aLF, aHLF, aTotal])

    pVLF = freq_features[0]['fft_rel'][1]
    pLF = freq_features[0]['fft_rel'][2]
    pHLF = freq_features[0]['fft_rel'][3]
    features.extend([pVLF, pLF, pHLF])

    nLF = freq_features[0]['fft_norm'][0]
    hLF = freq_features[0]['fft_norm'][1]
    features.extend([nLF, hLF])

    LFHF = freq_features[0]['fft_ratio']
    features.extend([LFHF])

    peakVLF = freq_features[0]['fft_peak'][1]
    peakLF = freq_features[0]['fft_peak'][2]
    peakHF = freq_features[0]['fft_peak'][3]
    features.extend([peakVLF, peakLF, peakHF])

    return features
