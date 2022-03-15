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
from src.d00_utils.featuring_tools import linear_regression, means_absolute_values, get_freq_features


def get_features(t, freq=4, threshold=2):
    """
    :param t: Window slice vector to obtain the features
    :param freq:
    :param threshold:
    :return:
    """
    features = []
    diff = np.diff(t)
    aux = diff > threshold
    if aux.any() == True:  # Check if there are two consecutive samples with a difference of "threshold"
        n_feat = 13
        null_block = list(np.zeros(n_feat))
        return null_block
    else:

        # Calculation of time-domain features
        # Feat 1: Mean
        t_mean = np.mean(t)
        features.extend([t_mean])

        # Feat 2:
        t_std = np.std(t, ddof=1)
        features.extend([t_std])

        # Feat 3 & 4: Intercept & Slope fitted Line
        t_intercept, t_slope = linear_regression(t)
        features.extend([t_intercept[0], t_slope[0][0]])

        # Feat 5 - 8: MAVFD, MAVFDN, MAVFSD, MAVFSDN
        t_dif1_m, t_dif2_m, t_dif1_norm_m, t_dif2_norm_m = means_absolute_values(t)
        features.extend([t_dif1_m, t_dif2_m, t_dif1_norm_m, t_dif2_norm_m])

        # Calculation of frequency-domain features

        # Feat 9 - 13: mean, std, p25, p50, p75 of the periodogram
        t_f_mean, t_f_std, t_f_p25, t_f_50, t_f_p75 = get_freq_features(t, freq)
        features.extend([t_f_mean, t_f_std, t_f_p25, t_f_50, t_f_p75])

        return features
