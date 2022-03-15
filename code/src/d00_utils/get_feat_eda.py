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
from src.d00_utils.featuring_tools import means_absolute_values, arc_length, integral, norm_average_power
from src.d00_utils.featuring_tools import get_freq_features, bandpower, norm_root_mean_square
from scipy.stats import skew, kurtosis, moment


def get_features(eda, scl, scr, freq=4):

    features = []
    # Feat 1: Mean
    scr_mean = np.mean(scr)
    scl_mean = np.mean(scl)
    eda_mean = np.mean(eda)
    features.extend([scr_mean, scl_mean, eda_mean])

    # Feat 2: Standard Deviation
    scr_std = np.std(scr, ddof=1)
    scl_std = np.std(scl, ddof=1)
    eda_std = np.std(eda, ddof=1)
    features.extend([scr_std, scl_std, eda_std])

    # Feat 3: Maximum
    scr_max = np.max(scr)
    scl_max = np.max(scl)
    eda_max = np.max(eda)
    features.extend([scr_max, scl_max, eda_max])

    # Feat 4: Minimum
    scr_min = np.min(scr)
    scl_min = np.min(scl)
    eda_min = np.min(eda)
    features.extend([scr_min, scl_min, eda_min])

    # Feat 5: Dynamic range
    scr_range = scr_max - scr_min
    scl_range = scl_max - scl_min
    eda_range = eda_max - eda_min
    features.extend([scr_range, scl_range, eda_range])

    # Feat 6: Fist Derivative Mean
    scr_1dif = np.diff(scr)
    scl_1dif = np.diff(scl)
    eda_1dif = np.diff(eda)

    scr_1dif_mean = np.mean(scr_1dif)
    scl_1dif_mean = np.mean(scl_1dif)
    eda_1dif_mean = np.mean(eda_1dif)
    features.extend([scr_1dif_mean, scl_1dif_mean, eda_1dif_mean])

    # Feat 7: Fist Derivative Standart Deviation
    scr_1dif_std = np.std(scr_1dif, ddof=1)
    scl_1dif_std = np.std(scl_1dif, ddof=1)
    eda_1dif_std = np.std(eda_1dif, ddof=1)
    features.extend([scr_1dif_std, scl_1dif_std, eda_1dif_std])

    # Feat 8: Second Derivative Mean
    scr_2dif = np.diff(scr_1dif)
    scl_2dif = np.diff(scl_1dif)
    eda_2dif = np.diff(eda_1dif)

    scr_2dif_mean = np.mean(scr_2dif)
    scl_2dif_mean = np.mean(scl_2dif)
    eda_2dif_mean = np.mean(eda_2dif)
    features.extend([scr_2dif_mean, scl_2dif_mean, eda_2dif_mean])

    # Feat 9: Second Derivative Standard Deviation
    scr_2dif_std = np.std(scr_2dif, ddof=1)
    scl_2dif_std = np.std(scl_2dif, ddof=1)
    eda_2dif_std = np.std(eda_2dif, ddof=1)
    features.extend([scr_2dif_std, scl_2dif_std, eda_2dif_std])

    # Feat 10: MAVFD, MAVFDN, MAVFSD, MAVFSDN
    scr_MAVFD, scr_MAVFSD, scr_MAVFDN, scr_MAVFSDN = means_absolute_values(scr)
    scl_MAVFD, scl_MAVFSD, scl_MAVFDN, scl_MAVFSDN = means_absolute_values(scl)
    eda_MAVFD, eda_MAVFSD, eda_MAVFDN, eda_MAVFSDN = means_absolute_values(eda)
    features.extend([scr_MAVFD, scr_MAVFSD, scr_MAVFDN, scr_MAVFSDN])
    features.extend([scl_MAVFD, scl_MAVFSD, scl_MAVFDN, scl_MAVFSDN])
    features.extend([eda_MAVFD, eda_MAVFSD, eda_MAVFDN, eda_MAVFSDN])

    # Feat 11: Fist differences of the Smooth EDA signal (SMFD)
    hann_ = np.hanning(len(eda))
    eda_smooth = np.multiply(eda, hann_)
    eda_smooth_1dif = eda_smooth[1:] - eda_smooth[0:-1]
    smfd_ = np.mean(eda_smooth_1dif)
    features.extend([smfd_])

    # Morphological Features
    # Feat 12: Arc Length
    scr_arc = arc_length(scr)
    scl_arc = arc_length(scl)
    eda_arc = arc_length(eda)
    features.extend([scr_arc, scl_arc, eda_arc])

    # Feature 13: Integral
    scr_integral = integral(scr)
    scl_integral = integral(scl)
    eda_integral = integral(eda)
    features.extend([scr_integral, scl_integral, eda_integral])

    # Feature 14: Normalized Average Power
    scr_nap = norm_average_power(scr)
    scl_nap = norm_average_power(scl)
    eda_nap = norm_average_power(eda)
    features.extend([scr_nap, scl_nap, eda_nap])

    # Feature 15: Normalized Root mean Square
    scr_nrms = norm_root_mean_square(scr_nap)
    scl_nrms = norm_root_mean_square(scl_nap)
    eda_nrms = norm_root_mean_square(eda_nap)
    features.extend([scr_nrms, scl_nrms, eda_nrms])

    # Feature 16: Area-perimeter Ratio (apr)
    scr_apr = scr_integral / scr_arc
    scl_apr = scl_integral / scl_arc
    eda_apr = eda_integral / eda_arc
    features.extend([scr_apr, scl_apr, eda_apr])

    # Feature 17: Energy-perimeter Ratio (epr)
    scr_epr = scr_nrms / scr_arc
    scl_epr = scl_nrms / scl_arc
    eda_epr = eda_nrms / eda_arc
    features.extend([scr_epr, scl_epr, eda_epr])

    # Feature 18: Skewness
    scr_skew = skew(scr)
    scl_skew = skew(scl)
    eda_skew = skew(eda)
    features.extend([scr_skew, scl_skew, eda_skew])

    # Feature 19: Kurtosis 
    scr_kurtosis = kurtosis(scr, fisher=False)
    scl_kurtosis = kurtosis(scl, fisher=False)
    eda_kurtosis = kurtosis(eda, fisher=False)
    features.extend([scr_kurtosis, scl_kurtosis, eda_kurtosis])

    # Feature 20: Central moment
    degree = 5
    scr_moment = moment(scr, degree)
    scl_moment = moment(scl, degree)
    eda_moment = moment(eda, degree)
    features.extend([scr_moment, scl_moment, eda_moment])

    # Frequancy-domain features
    scr_f_mean, scr_f_std, scr_f_p25, scr_f_p50, scr_f_p75 = get_freq_features(scr, freq)
    scl_f_mean, scl_f_std, scl_f_p25, scl_f_p50, scl_f_p75 = get_freq_features(scl, freq)
    eda_f_mean, eda_f_std, eda_f_p25, eda_f_p50, eda_f_p75 = get_freq_features(eda, freq)
    features.extend([scr_f_mean, scr_f_std, scr_f_p25, scr_f_p50, scr_f_p75])
    features.extend([scl_f_mean, scl_f_std, scl_f_p25, scl_f_p50, scl_f_p75])
    features.extend([eda_f_mean, eda_f_std, eda_f_p25, eda_f_p50, eda_f_p75])

    # Bandpower features
    scr_band_1 = bandpower(scr, freq, 0.1, 0.2)
    scr_band_2 = bandpower(scr, freq, 0.2, 0.3)
    scr_band_3 = bandpower(scr, freq, 0.3, 0.4)
    features.extend([scr_band_1, scr_band_2, scr_band_3])

    scl_band_1 = bandpower(scl, freq, 0.1, 0.2)
    scl_band_2 = bandpower(scl, freq, 0.2, 0.3)
    scl_band_3 = bandpower(scl, freq, 0.3, 0.4)
    features.extend([scl_band_1, scl_band_2, scl_band_3])

    eda_band_1 = bandpower(eda, freq, 0.1, 0.2)
    eda_band_2 = bandpower(eda, freq, 0.2, 0.3)
    eda_band_3 = bandpower(eda, freq, 0.3, 0.4)
    features.extend([eda_band_1, eda_band_2, eda_band_3])

    #TODO: Comprobar que las funciones basicas actuan como en Matlab!!
    return features
