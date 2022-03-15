# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.23
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
import pandas as pd
from scipy import interpolate
from scipy.fft import fft, ifft
from csaps import csaps
import numpy as np
import json
import os
from src.d00_utils.get_feat_hr import get_features as get_features_hr
from src.d00_utils.get_feat_temp import get_features as get_features_temp
from src.d00_utils.get_feat_eda import get_features as get_features_eda
from src.d00_utils.get_feat_acc import get_features as get_features_acc


class PhysioFeatExtractor:
    """
    This class extracts physiological features for each of day specified in the constructor
    """
    def __init__(self, participant, n_days, weda, w_slide):
        self.participant = participant
        self.days = int(n_days)
        self.weda = weda
        self.w_slide = w_slide
        self.signals = ['acc', 'eda', 'temp', 'hr']
        self.freqs = [32, 4, 4, 1]
        self.overlap = 0.1
        self.acc = None
        self.eda = None
        self.temp = None
        self.hr = None
        self.acc_feat = None
        self.eda_feat = None
        self.temp_feat = None
        self.hr_feat = None

    def run(self):  # This method will iterate though all day set by attribute "days"
        for day in range(1, self.days+1):
            output_path = "data/03_features/physio-feat/{}/{}/{}/".format(str(self.weda), self.participant, str(day))
            self.load_data(day=day)
            cols = self.acc.columns
            if "label_m" not in cols:
                feat_lab_dict = {"features": -1, "labels_m": -1, "labels_h": -1, "labels_a": -1}
            else:
                self.extract_features()
                feat_lab_dict = self.concat_features()

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_filename = output_path + "features.json"
            with open(output_filename, 'w') as json_f:
                json.dump(feat_lab_dict, json_f)

    def load_data(self, day):
        df_path = "data/02_intermediate/{}/{}/{}/".format(self.weda, self.participant, day)
        acc_df_path = df_path + "ACC.csv"
        eda_df_path = df_path + "EDA.csv"
        temp_df_path = df_path + "TEMP.csv"
        hr_df_path = df_path + "HR.csv"

        self.acc = pd.read_csv(acc_df_path, sep='\t')
        self.eda = pd.read_csv(eda_df_path, sep='\t')
        self.temp = pd.read_csv(temp_df_path, sep='\t')
        self.hr = pd.read_csv(hr_df_path, sep='\t')

    def extract_features(self):
        def sliding_window_feat_extraction(df, freq, signal):
            init_id = df['index'].iloc[0]
            end_id = df['index'].iloc[-1]

            # Boundaries to control cases where EMA is rigth at beginning and end of Dataframe
            init_bound_ind = int(init_id + self.w_slide * freq / 2)
            end_bound_ind = int(end_id - self.w_slide * freq / 2)
            init_bound = df['ts'][df['index'] == init_bound_ind].to_numpy()[0]
            end_bound = df['ts'][df['index'] == end_bound_ind].to_numpy()[0]

            # Get id that correspond to a label; Filtering: checking boundaries
            idx = df.index[(df['label_m'] != -1) & (init_bound_ind < df['index']) & (df['index'] < end_bound_ind)]
            idxs = np.asarray(idx)

            # Get indices where each EMA starts
            idxs_aux = idxs[0:len(idxs) - 1]
            init_ema_aux = (idxs[1:len(idxs)] - idxs_aux).reshape(-1, 1)
            init_ema_id = np.where(np.any(init_ema_aux > 1, axis=1))[0] + 1
            init_ema_indices = idxs[init_ema_id].tolist()

            counter = 1
            features = []
            labels_m = []
            labels_h = []
            labels_a = []
            w_central = idxs[0]  # It is chosen first element: filtered list
            id_ = 0
            offset = (1 - self.overlap) * self.w_slide * freq
            while id_ <= len(idxs):
                w_left = w_central - (self.w_slide * freq / 2 - 1)
                w_right = w_central + (self.w_slide * freq / 2 - 1)

                df_window = df.loc[df['index'].between(w_left, w_right)]

                if signal == 'acc':
                    x_ = df_window['x'].to_numpy()
                    y_ = df_window['y'].to_numpy()
                    z_ = df_window['z'].to_numpy()
                    n_ = df_window['n'].to_numpy()
                    feat_ = get_features_acc(x_, y_, z_, n_)
                elif signal == 'eda':
                    eda, scl, scr = df_window['eda'].to_numpy(), df_window['scl'].to_numpy(), \
                                    df_window['scr'].to_numpy()
                    feat_ = get_features_eda(eda, scl, scr)
                elif signal == 'temp':
                    temp_ = df_window['temp'].to_numpy()
                    feat_ = get_features_temp(temp_)
                elif signal == 'hr':
                    hr_ = df_window['hr'].to_numpy()
                    feat_ = get_features_hr(hr_)

                label_m = int(df['label_m'][df['index'].iloc[w_central]])
                label_h = int(df['label_h'][df['index'].iloc[w_central]])
                label_a = int(df['label_a'][df['index'].iloc[w_central]])

                feat_ = np.asarray(feat_)
                features.append(feat_)
                labels_m.append(label_m)
                labels_h.append(label_h)
                labels_a.append(label_a)

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

            features_ = np.vstack(features)
            features_ = np.nan_to_num(features_, nan=0, posinf=0, neginf=0)
            labels_m_ = np.vstack(labels_m)
            labels_h_ = np.vstack(labels_h)
            labels_a_ = np.vstack(labels_a)
            feat_label_dict = {"features": features_.tolist(), "labels_m": labels_m_.tolist(),
                               "labels_h": labels_h_.tolist(), "labels_a": labels_a_.tolist()}

            return feat_label_dict

        for signal, freq in zip(self.signals, self.freqs):
            df_ = getattr(self, signal)
            df_ = df_.reset_index()
            columns = df_.columns
            if 'label_m' not in columns:
                continue
            else:
                # Prepara EDA Dataframe before extractint features: Calculate SCR and SCL
                if signal == 'eda':
                    eda = df_['eda'].to_numpy()
                    x = np.arange(0, len(eda), 1)
                    scl = csaps(x, eda, x, smooth=0.5)
                    scr = np.real_if_close(ifft(fft(eda) / fft(scl)))
                    df_['scr'] = scr
                    df_['scl'] = scl

                feat_dict_ = sliding_window_feat_extraction(df_, freq, signal)
                if signal == 'acc':
                    self.acc_feat = feat_dict_
                elif signal == 'eda':
                    self.eda_feat = feat_dict_
                elif signal == 'temp':
                    self.temp_feat = feat_dict_
                else:
                    self.hr_feat = feat_dict_

    def concat_features(self):
        min_n_row = np.inf
        # Loop to get the number of rows: Sometimes some signals stop sampling earlier, therefore we get less
        # windows/vectors
        for signal in self.signals:
            feat_dict_ = getattr(self, signal + "_feat")
            df_ = feat_dict_['features']
            n_row_ = len(df_)
            min_n_row = n_row_ if n_row_ < min_n_row else min_n_row

        concat_feat = np.empty((min_n_row, 0))  # Array with concatenated features
        lab_m_list = []
        lab_h_list = []
        lab_a_list = []
        for signal in self.signals:
            feat_dict_ = getattr(self, signal + "_feat")
            df_ = np.array(feat_dict_['features'])[0:min_n_row, :]
            lab_m = np.array(feat_dict_['labels_m']).reshape(-1, 1)[0:min_n_row, :]
            lab_h = np.array(feat_dict_['labels_h']).reshape(-1, 1)[0:min_n_row, :]
            lab_a = np.array(feat_dict_['labels_a']).reshape(-1, 1)[0:min_n_row, :]

            concat_feat = np.concatenate([concat_feat, df_], axis=1)
            lab_m_list.append(lab_m)    # We create a list to assess if all lab arrays are same for all signals
            lab_h_list.append(lab_h)
            lab_a_list.append(lab_a)

        assert lab_m_list[0].all() == lab_m_list[1].all() == lab_m_list[2].all() == lab_m_list[3].all(), "Error! " \
                                                                                                 "Something went wrong"
        assert lab_h_list[0].all() == lab_h_list[1].all() == lab_h_list[2].all() == lab_h_list[3].all(), "Error! " \
                                                                                                 "Something went wrong"
        assert lab_a_list[0].all() == lab_a_list[1].all() == lab_a_list[2].all() == lab_a_list[3].all(), "Error! " \
                                                                                                 "Something went wrong"

        concat_lab_m = lab_m_list[0]    # If they are all the same, we can take first array in the list, for example
        concat_lab_h = lab_h_list[0]
        concat_lab_a = lab_a_list[0]

        concat_feat_lab_dict = {"features": concat_feat.tolist(), "labels_m": concat_lab_m.tolist(),
                                "labels_h": concat_lab_h.tolist(), "labels_a":concat_lab_a.tolist()}

        return concat_feat_lab_dict
