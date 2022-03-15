# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.12.14
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from csaps import csaps

def clean_dataset(labels, discard=0.1):

    # Obtain an accounting of hbiw many vectors there is for each emotion
    unique, counts = np.unique(labels, return_counts=True)
    counting = dict(zip(unique, counts))

    # Get the total amount of vectors and the threshold to filter dictionary
    tot = sum(counting.values())
    threshold = tot * discard

    # Get a dictionary with the emotions that should be cleaned from the initial variables
    # It is kept a dictionary to check the lenght of the cleaned values at the end
    emo_del_dict = dict(filter(lambda elem: elem[1] < threshold, counting.items()))
    # Array that contains the value of the emotions to be cleaned in the "labels" variable
    emo_del_arr = np.array(list(emo_del_dict.keys()))

    # Array containing the index that should be deleted from "data" and "label"
    indx_del_arr = np.where(labels == emo_del_arr)[0]
    assert indx_del_arr.shape[0] == sum(
        emo_del_dict.values()), "The amount of vectors to delete does not match! Check it."

    return indx_del_arr


def filter_nrows(feature_, lab_):
    # 1 Step: Find min number of rows within signals
    min_nrow = np.inf
    for arr in feature_:
        nrow_ = arr.shape[0]
        min_nrow = nrow_ if nrow_ < min_nrow else min_nrow
    # 2 Step: Modify each array in the list in case
    feat_signals_filter = []
    lab_signals_filter = []
    for arr, lab in zip(feature_, lab_):
        arr_ = arr[0:min_nrow, :]
        lab_ = lab[0:min_nrow,]
        feat_signals_filter.append(arr_)
        lab_signals_filter.append(lab_)
    return feat_signals_filter, lab_signals_filter[0]


#def extract_ts_features(df, emotion, weda, patient, day, signal):
def extract_ts_features(df, signal):
    #DATASET = "/home/eduardo/phd/projects/opt-physio-feat-extractor/2-emotion-to-vector/out/filter/"
    #DATASET = "/home/eduardo/phd/projects/physio-feat-extractor/physio-feat-extractor/2-emotion-to-vector/out/filter/"
    signal2Freq = {
        "HR": 1,
        "ACC": 32,
        "EDA": 4,
        "TEMP": 4
    }
    FREQ = signal2Freq[signal]
    Ts = 1 / FREQ
    WINDOW = 60  # sliding window size
    OVERLAP = 0.1

    #datapath = DATASET + "{}/{}/{}/{}/{}.csv".format(emotion, weda, patient, day, signal)
    #df = pd.read_csv(datapath, sep='\t')
    df = df.reset_index()

    if signal == 'EDA':
        eda = df['eda'].to_numpy()
        x = np.arange(0, len(eda), 1)
        scl = csaps(x, eda, x, smooth=0.5)
        scr = np.real_if_close(ifft(fft(eda) / fft(scl)))
        df['scr'] = scr
        df['scl'] = scl


    if df.shape[0] == 1:
        # You should create an Dataframe with -1 to indicate that there is no data available
        print("Empty DataFrame. Exiting program...")
        return -1, -1

    init_id = df['index'].iloc[0]
    end_id = df['index'].iloc[-1]

    # Boundaries to control cases where EMA is rigth at beginning and end of Dataframe
    init_bound_ind = int(init_id + WINDOW * FREQ / 2)
    end_bound_ind = int(end_id - WINDOW * FREQ / 2)

    # Get id that correspond to a label; Filtering: checking boundaries
    idx = df.index[(df['label'] != -1) & (init_bound_ind < df['index']) & (df['index'] < end_bound_ind)]
    idxs = np.asarray(idx)

    idxs_aux = idxs[0:len(idxs) - 1]
    init_ema_aux = (idxs[1:len(idxs)] - idxs_aux).reshape(-1, 1)
    init_ema_id = np.where(np.any(init_ema_aux > 1, axis=1))[0] + 1
    init_ema_indices = idxs[init_ema_id].tolist()

    counter = 1

    w_central = idxs[0]  # It is chosen first element: filtered list
    id_ = 0
    offset = (1 - OVERLAP) * WINDOW * FREQ

    ts_df = pd.DataFrame(columns=['id', 'time', 'kind', 'value'])
    ts_df_label = pd.DataFrame(columns=['id', 'label'])


    while id_ <= len(idxs):
        w_left = w_central - (WINDOW * FREQ / 2 - 1)
        w_right = w_central + (WINDOW * FREQ / 2 - 1)

        df_window = df.loc[df['index'].between(w_left, w_right)]
        label_ = int(df['label'][df['index'].iloc[w_central]])

        if signal == 'ACC':
            x_ = df_window['x'].to_numpy()
            y_ = df_window['y'].to_numpy()
            z_ = df_window['z'].to_numpy()
            n_ = df_window['n'].to_numpy()
            ts_ = np.arange(len(x_))

            data_x = {'id': counter, 'time': ts_, 'kind': 'acc_x', 'value': x_}
            data_y = {'id': counter, 'time': ts_, 'kind': 'acc_y', 'value': y_}
            data_z = {'id': counter, 'time': ts_, 'kind': 'acc_z', 'value': z_}
            data_n = {'id': counter, 'time': ts_, 'kind': 'acc_n', 'value': n_}

            aux_x = pd.DataFrame(data=data_x)
            aux_y = pd.DataFrame(data=data_y)
            aux_z = pd.DataFrame(data=data_z)
            aux_n = pd.DataFrame(data=data_n)
            aux_ = pd.concat([aux_x, aux_y, aux_z, aux_n], axis=0)

        elif signal == "EDA":
            eda_ = df_window[signal.lower()].to_numpy()
            ts_ = np.arange(len(eda_))
            data_eda = {'id': counter, 'time': ts_, 'kind': "eda", 'value': eda_}
            aux_eda = pd.DataFrame(data=data_eda)

            scl_ = df_window['scl'].to_numpy()
            ts_ = np.arange(len(scl_))
            data_scl = {'id': counter, 'time': ts_, 'kind': "scl", 'value': scl_}
            aux_scl = pd.DataFrame(data=data_scl)

            scr_ = df_window['scr'].to_numpy()
            ts_ = np.arange(len(scr_))
            data_scr = {'id': counter, 'time': ts_, 'kind': "scr", 'value': scr_}
            aux_scr = pd.DataFrame(data=data_scr)

            aux_ = pd.concat([aux_eda, aux_scl, aux_scr], axis=0)

        elif signal == "TEMP":
            data_ = df_window[signal.lower()].to_numpy()
            ts_ = np.arange(len(data_))
            data = {'id': counter, 'time': ts_, 'kind': "temp", 'value': data_}
            aux_ = pd.DataFrame(data=data)
        else:
            data_ = df_window[signal.lower()].to_numpy()
            ts_ = np.arange(len(data_))
            data = {'id': counter, 'time': ts_, 'kind': "hr", 'value': data_}
            aux_ = pd.DataFrame(data=data)

        data_label = {'id': [counter], "label": [label_]}
        aux_label = pd.DataFrame(data=data_label)
        ts_df = pd.concat([ts_df, aux_], axis=0)
        ts_df_label = pd.concat([ts_df_label, aux_label], axis=0)

        # Logic code to move sliding window
        w_central_aux = int(w_central + offset)
        if w_central_aux not in idxs:
            if not init_ema_indices:  # if this list is empty it means that you arrived to the end
                break
            # Skip to next starting EMA indicated by init_ema_indices
            w_central = init_ema_indices.pop(0)
            # modify variable id_
            id_ = np.where(idxs == w_central)[0][0]
        else:
            # id_ = indice que ocupa w_central_aux en la lista idxs
            w_central = w_central_aux
            id_ = np.where(idxs == w_central_aux)[0][0]

        counter += 1

    return ts_df, ts_df_label