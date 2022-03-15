# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.23
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
import numpy as np
from scipy.signal import butter, lfilter


def compute_norm(dataframe):
    x = dataframe['x'].to_numpy()
    y = dataframe['y'].to_numpy()
    z = dataframe['z'].to_numpy()

    # Reshape them to concatenate lately
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    z = z.reshape(z.shape[0], -1)

    aux = np.concatenate((x, y, z), axis=1)
    return np.linalg.norm(aux, axis=1)


def filter_acc_signal(data):
    def butter_bandpass(lowcut=0.2, highcut=10, order=3, fs=32):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass()
    return lfilter(b, a, data, axis=0)


def filter_eda_signal(data):
    def butter_lowpass(lowcut=1.5, order=3, fs=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='low')
        return b, a
    b, a = butter_lowpass()
    return lfilter(b, a, data, axis=0)