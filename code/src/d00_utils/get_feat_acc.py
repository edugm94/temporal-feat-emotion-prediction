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
#   Date: 2021.05.05
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden

import numpy as np
from statsmodels import robust
from src.d00_utils.featuring_tools import get_freq_features, means_absolute_values, prctile


def get_features(x, y, z, n, freq=32):
    """
    This function calculates time-domain and frequancy-domain features for the ACC signal
    :return: time and frequency features for tht window slice
    """
    features = []

    # Calculaction of time-domain features
    # Feat 1: Maximum
    x_max = np.max(x)
    y_max = np.max(y)
    z_max = np.max(z)
    n_max = np.max(n)
    features.extend([x_max, y_max, z_max, n_max])

    # Feat 2: Percentile 90
    x_p90 = prctile(x, 90)
    y_p90 = prctile(y, 90)
    z_p90 = prctile(z, 90)
    n_p90 = prctile(n, 90)
    features.extend([x_p90, y_p90, z_p90, n_p90])

    # Feat 3: Variance
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    z_var = np.var(z, ddof=1)
    n_var = np.var(n, ddof=1)
    features.extend([x_var, y_var, z_var, n_var])

    # Feat 4: MAD
    x_mad = np.mean(np.abs(x-np.mean(x)))
    y_mad = np.mean(np.abs(y-np.mean(y)))
    z_mad = np.mean(np.abs(z-np.mean(z)))
    n_mad = np.mean(np.abs(n-np.mean(n)))
    features.extend([x_mad, y_mad, z_mad, n_mad])

    # Feat 5: Norm
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    z_norm = np.linalg.norm(z)
    n_norm = np.linalg.norm(n)
    features.extend([x_norm, y_norm, z_norm, n_norm])

    # Feat 6: Amplitude
    x_amp = x_max - np.mean(x)
    y_amp = y_max - np.mean(y)
    z_amp = z_max - np.mean(z)
    n_amp = n_max - np.mean(n)
    features.extend([x_amp, y_amp, z_amp, n_amp])

    # Feat 7: Minimum
    x_min = np.min(x)
    y_min = np.min(y)
    z_min = np.min(z)
    n_min = np.min(n)
    features.extend([x_min, y_min, z_min, n_min])

    # Feat 8: Standard Deviation
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    z_std = np.std(z, ddof=1)
    n_std = np.std(n, ddof=1)
    features.extend([x_std, y_std, z_std, n_std])

    # Feat 9: RMS
    x_rms = np.sqrt(np.mean(x**2))
    y_rms = np.sqrt(np.mean(y**2))
    z_rms = np.sqrt(np.mean(z**2))
    n_rms = np.sqrt(np.mean(n**2))
    features.extend([x_rms, y_rms, z_rms, n_rms])

    x_dif1_m, x_dif2_m, x_dif1_norm_m, x_dif2_norm_m = means_absolute_values(x)
    y_dif1_m, y_dif2_m, y_dif1_norm_m, y_dif2_norm_m = means_absolute_values(y)
    z_dif1_m, z_dif2_m, z_dif1_norm_m, z_dif2_norm_m = means_absolute_values(z)
    n_dif1_m, n_dif2_m, n_dif1_norm_m, n_dif2_norm_m = means_absolute_values(n)

    # Feat 10: MAVFD
    features.extend([x_dif1_m, y_dif1_m, z_dif1_m, n_dif1_m])

    # Feat 11: MAVFDN
    features.extend([x_dif2_m, y_dif2_m, z_dif2_m, n_dif2_m])

    # Feat 12: MAVSD
    features.extend([x_dif1_norm_m, y_dif1_norm_m, z_dif1_norm_m, n_dif1_norm_m])

    # Feat 13: MAVSDN
    features.extend([x_dif2_norm_m, y_dif2_norm_m, z_dif2_norm_m, n_dif2_norm_m])

    # Calculaction of frequency-domain features
    x_f_mean, x_f_std, x_f_p25, x_f_p50, x_f_p75 = get_freq_features(x, freq)
    y_f_mean, y_f_std, y_f_p25, y_f_p50, y_f_p75 = get_freq_features(y, freq)
    z_f_mean, z_f_std, z_f_p25, z_f_p50, z_f_p75 = get_freq_features(z, freq)
    n_f_mean, n_f_std, n_f_p25, n_f_p50, n_f_p75 = get_freq_features(n, freq)

    # Feat 14: Mean Periodogram
    features.extend([x_f_mean, y_f_mean, z_f_mean, n_f_mean])

    # Feat 15: Std Periodogram
    features.extend([x_f_std, y_f_std, z_f_std, n_f_std])

    # Feat 16: P25 Periodogram
    features.extend([x_f_p25, y_f_p25, z_f_p25, n_f_p25])

    # Feat 17: P50 Periodogram
    features.extend([x_f_p50, y_f_p50, z_f_p50, n_f_p50])

    # Feat 18 : P75 Periodogram
    features.extend([x_f_p75, y_f_p75, z_f_p75, n_f_p75])

    return features
