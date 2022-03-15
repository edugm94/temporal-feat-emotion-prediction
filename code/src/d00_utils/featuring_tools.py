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
#   Date: 2021.04.15
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.mlab import psd


def meanabs(x):
    m = np.mean(np.abs(x))
    n = np.count_nonzero(np.isnan(x))

    return m, n


def means_absolute_values(x):
    mean_ = np.mean(x)
    std_ = np.std(x, ddof=1)
    x_norm = (x - mean_) / std_ if std_ != 0 else np.zeros(len(x))
    #x_norm = (x - mean_) / std_
    #x_norm = np.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)

    x_dif1 = x[1:] - x[0:-1]
    x_dif2 = x[2:] - x[0:-2]
    x_dif1_norm = x_norm[1:] - x_norm[0:-1]
    x_dif2_norm = x_norm[2:] - x_norm[0:-2]

    [x_dif1_m, x_dif1_n] = meanabs(x_dif1)
    [x_dif2_m, x_dif2_n] = meanabs(x_dif2)
    [x_dif1_norm_m, x_dif1_norm_n] = meanabs(x_dif1_norm)
    [x_dif2_norm_m, x_dif2_norm_n] = meanabs(x_dif2_norm)

    return x_dif1_m, x_dif2_m, x_dif1_norm_m, x_dif2_norm_m


def quantile(x, q):
    n = len(x)
    y = np.sort(x)
    return np.interp(q, np.linspace(1 / (2 * n), (2 * n - 1) / (2 * n), n), y)


def prctile(x, p):
    return quantile(x, np.array(p) / 100)


def get_periodogram(x, f):
    t = np.arange(0, 1-(1/f)+0.001, 1/f)
    N = x.shape[-1]
    xdft = np.fft.fft(x)
    xdft = xdft[0: int(N/2+1)]
    psdx = (1/(f*N)) * (np.abs(xdft))**2
    psdx[1:-1] = 2*psdx[1:-1]
    freq = np.arange(0, f/2+0.001, f/x.shape[-1])

    return psdx, freq


def get_freq_features(x, f):
    np.seterr(divide='ignore')  # Para ignorar los warning
    #np.seterr(divide='warn')   # Para activar los warning
    pgram, freq = get_periodogram(x, f)
    pgram_log = 10*np.log10(pgram)
    pgram_ = np.nan_to_num(pgram_log, nan=0, posinf=0, neginf=0)

    mean_ = np.mean(pgram_)
    std_ = np.std(pgram_, ddof=1)
    p25 = prctile(pgram_, 25)
    p50 = prctile(pgram_, 50)
    p75 = prctile(pgram_, 75)

    return mean_, std_, p25, p50, p75


def linear_regression(y):
    m = len(y)
    ts = np.linspace(0, m-1, m).reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression().fit(ts, y)

    return model.intercept_, model.coef_


def arc_length(x):
    return np.sum(np.sqrt(1 + (x[1:] - x[0:-1])**2))


def integral(x):
    return np.sum(np.abs(x))


def norm_average_power(x):
    return np.sum(x**2)/len(x)


def norm_root_mean_square(x_norm_avg_power):
    return np.sqrt(x_norm_avg_power)


def bandpower(x, Fs, lF, uF):
    ps, freqs = psd(x=x, Fs=Fs, window=np.hamming(len(x)), NFFT=len(x))

    W_dif = np.diff(freqs)
    W_dif = np.append(W_dif, 0)

    ind_min = np.where(freqs <= lF)[0][-1]
    ind_max = np.where(freqs >= uF)[0][0] + 1

    pwr = np.dot(W_dif[ind_min:ind_max], ps[ind_min:ind_max])
    return pwr


