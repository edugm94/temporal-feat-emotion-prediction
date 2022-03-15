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
import numpy as np
import os
import json
from tsfresh import extract_relevant_features, extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from scipy.fft import fft, ifft
from csaps import csaps

class TempFeatExtractor:
    def __init__(self, participant, n_days, weda, w_slide):
        self.participant = participant
        self.days = int(n_days)
        self.weda = weda
        self.w_slide = w_slide
        self.signals = ['acc', 'eda', 'temp', 'hr']
        #self.signals = ['acc', 'temp', 'hr']
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

    def run(self):

        for day in range(1, self.days+1):
            output_path = "data/03_features/temporal-feat/{}/{}/{}/".format(str(self.weda), self.participant, str(day))
            #self.load_data(day=10)
            self.load_data(day=day)
            cols = self.acc.columns
            if "label_m" not in cols:
                feat_lab_dict = {"features": -1, "labels_m": -1, "labels_h": -1, "labels_a": -1, "feat_col_names": -1}
            else:
                #self.extract_features()
                #feat_lab_dict = self.concat_features()
                feat_lab_dict = self.extract_feat_stackedDF()

            # TODO: Uncomment after debugging
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
        def sliding_window_feat_extraction(df, signal, freq):
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

            idxs_aux = idxs[0:len(idxs) - 1]
            init_ema_aux = (idxs[1:len(idxs)] - idxs_aux).reshape(-1, 1)
            init_ema_id = np.where(np.any(init_ema_aux > 1, axis=1))[0] + 1
            init_ema_indices = idxs[init_ema_id].tolist()

            counter = 1

            w_central = idxs[0]  # It is chosen first element: filtered list
            id_ = 0
            offset = (1 - self.overlap) * self.w_slide * freq

            # Creation of Dataframes where subsequences and corresponding labels will be stored
            if signal == 'acc':
                ts_df = pd.DataFrame(columns=['id', 'time', 'x', 'y', 'z', 'n'])
            else:
                ts_df = pd.DataFrame(columns=['id', 'time', 'data'])
            ts_df_label_m = pd.DataFrame(columns=['id', 'label'])
            ts_df_label_h = pd.DataFrame(columns=['id', 'label'])
            ts_df_label_a = pd.DataFrame(columns=['id', 'label'])

            while id_ <= len(idxs):
                w_left = w_central - (self.w_slide * freq / 2 - 1)
                w_right = w_central + (self.w_slide * freq / 2 - 1)

                df_window = df.loc[df['index'].between(w_left, w_right)]
                label_m = int(df['label_m'][df['index'].iloc[w_central]])
                label_h = int(df['label_h'][df['index'].iloc[w_central]])
                label_a = int(df['label_a'][df['index'].iloc[w_central]])

                if signal == 'acc':
                    x_ = df_window['x'].to_numpy()
                    y_ = df_window['y'].to_numpy()
                    z_ = df_window['z'].to_numpy()
                    n_ = df_window['n'].to_numpy()
                    ts_ = np.arange(len(x_))
                    data = {'id': counter,
                            'time': ts_,
                            'x': x_,
                            'y': y_,
                            'z': z_,
                            'n': n_}
                else:
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter,
                            'time': ts_,
                            'data': data_}

                # Concatenation of temporal subsequences in final DataFrame
                aux = pd.DataFrame(data=data)
                data_label_m = {'id': [counter],
                              "label": [label_m]}
                data_label_h = {'id': [counter],
                                "label": [label_h]}
                data_label_a = {'id': [counter],
                                "label": [label_a]}

                aux_label_m = pd.DataFrame(data=data_label_m)
                aux_label_h = pd.DataFrame(data=data_label_h)
                aux_label_a = pd.DataFrame(data=data_label_a)
                ts_df = pd.concat([ts_df, aux], axis=0)
                ts_df_label_m = pd.concat([ts_df_label_m, aux_label_m], axis=0)
                ts_df_label_h = pd.concat([ts_df_label_h, aux_label_h], axis=0)
                ts_df_label_a = pd.concat([ts_df_label_a, aux_label_a], axis=0)

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

            # Extract relevant features
            ts_df_label_m = ts_df_label_m.reset_index(drop=True)
            ts_df_label_h = ts_df_label_h.reset_index(drop=True)
            ts_df_label_a = ts_df_label_a.reset_index(drop=True)

            #TODO: Prueba para ver donde casca
            #ts_df = ts_df[ts_df['id'].between(133, 139)]
            #ts_df['data'] = 0
            if signal != 'acc':
                #ts_df['data'][ts_df['data'] < 1e-10] = 0
                ts_df.loc[ts_df.data < 1e-10, "data"] = 0 # Important: This was done in order not to crush extract_features() function
            else:
                pass
                #ts_df['x'][ts_df['x'] < 1e-10] = 0
                #ts_df['y'][ts_df['y'] < 1e-10] = 0
                #ts_df['z'][ts_df['z'] < 1e-10] = 0
                #ts_df['n'][ts_df['n'] < 1e-10] = 0

            features_ = extract_features(ts_df, column_id='id', column_sort='time')
            features_ = features_.to_numpy()
            features_ = np.nan_to_num(features_, nan=0, posinf=0, neginf=0)
            #X_extrt_clean = X_extracted.dropna(how='any', axis=1)
            feat_label_dict = {"features": features_.tolist(),
                               "labels_m": ts_df_label_m['label'].tolist(),
                               "labels_h": ts_df_label_h['label'].tolist(),
                               "labels_a": ts_df_label_a['label'].tolist()}

            return feat_label_dict

        for signal, freq in zip(self.signals, self.freqs):
            df_ = getattr(self, signal)
            df_ = df_.reset_index()
            columns = df_.columns
            if 'label_m' not in columns:
                continue
            else:
                feat_dict_ = sliding_window_feat_extraction(df_, signal, freq)
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
                                "labels_h": concat_lab_h.tolist(), "labels_a": concat_lab_a.tolist()}

        return concat_feat_lab_dict

    def extract_feat_stackedDF(self):   # TODO: If you use this method, you do not need to call concat_features() afterwards
        def get_stacked_df(df, signal, freq):

            if signal == 'eda':
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
            init_bound_ind = int(init_id + self.w_slide * freq / 2)
            end_bound_ind = int(end_id - self.w_slide * freq / 2)
            init_bound = df['ts'][df['index'] == init_bound_ind].to_numpy()[0]
            end_bound = df['ts'][df['index'] == end_bound_ind].to_numpy()[0]

            # Get id that correspond to a label; Filtering: checking boundaries
            idx = df.index[(df['label_m'] != -1) & (init_bound_ind < df['index']) & (df['index'] < end_bound_ind)]
            idxs = np.asarray(idx)

            idxs_aux = idxs[0:len(idxs) - 1]
            init_ema_aux = (idxs[1:len(idxs)] - idxs_aux).reshape(-1, 1)
            init_ema_id = np.where(np.any(init_ema_aux > 1, axis=1))[0] + 1
            init_ema_indices = idxs[init_ema_id].tolist()

            counter = 1

            w_central = idxs[0]  # It is chosen first element: filtered list
            id_ = 0
            offset = (1 - self.overlap) * self.w_slide * freq

            ts_df = pd.DataFrame(columns=['id', 'time', 'kind', 'value'])
            ts_df_lab_mood = pd.DataFrame(columns=['id', 'label']) #TODO: Here you can create 3 dataframes and monitor three labels
            ts_df_lab_happ = pd.DataFrame(columns=['id', 'label'])
            ts_df_lab_acti = pd.DataFrame(columns=['id', 'label'])

            while id_ <= len(idxs):
                w_left = w_central - (self.w_slide * freq / 2 - 1)
                w_right = w_central + (self.w_slide * freq / 2 - 1)

                df_window = df.loc[df['index'].between(w_left, w_right)]
                label_m = int(df['label_m'][df['index'].iloc[w_central]])
                label_h = int(df['label_h'][df['index'].iloc[w_central]])
                label_a = int(df['label_a'][df['index'].iloc[w_central]])

                if signal == 'acc':
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

                elif signal == "eda":
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
                elif signal == "temp":
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter, 'time': ts_, 'kind': "temp", 'value': data_}
                    aux_ = pd.DataFrame(data=data)

                # elif signal == "HR":
                else:
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter, 'time': ts_, 'kind': "hr", 'value': data_}
                    aux_ = pd.DataFrame(data=data)

                data_label_m = {'id': [counter], "label": [label_m]}
                data_label_h = {'id': [counter], "label": [label_h]}
                data_label_a = {'id': [counter], "label": [label_a]}

                aux_label_m = pd.DataFrame(data=data_label_m)
                aux_label_h = pd.DataFrame(data=data_label_h)
                aux_label_a = pd.DataFrame(data=data_label_a)

                ts_df = pd.concat([ts_df, aux_], axis=0)
                ts_df_lab_mood = pd.concat([ts_df_lab_mood, aux_label_m], axis=0)
                ts_df_lab_happ = pd.concat([ts_df_lab_happ, aux_label_h], axis=0)
                ts_df_lab_acti = pd.concat([ts_df_lab_acti, aux_label_a], axis=0)

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

            return ts_df, ts_df_lab_mood, ts_df_lab_happ, ts_df_lab_acti

        df_feat_aux = pd.DataFrame(columns=['id', 'time', 'kind', 'value'])
        for signal, freq in zip(self.signals, self.freqs):
            df_ = getattr(self, signal)
            df_ = df_.reset_index()
            columns = df_.columns
            if 'label_m' not in columns:
                continue
            else:
                df_feat_, df_lab_mood, df_lab_happ, df_lab_acti = get_stacked_df(df_, signal, freq)
                df_feat_aux = pd.concat([df_feat_aux, df_feat_])

        y_mood = pd.Series(data=df_lab_mood['label'].to_numpy(), index=df_lab_mood['id'].to_numpy())
        y_happ = pd.Series(data=df_lab_happ['label'].to_numpy(), index=df_lab_happ['id'].to_numpy())
        y_acti = pd.Series(data=df_lab_acti['label'].to_numpy(), index=df_lab_acti['id'].to_numpy())

        # IMPORTANT CODE LINE: We need below line in case we have residual data, e.g -6.123e-14, that is, zero value
        df_feat_aux.loc[df_feat_aux.value < 10e-5, "value"] = 0
        features_ = extract_features(df_feat_aux, column_id="id", column_sort="time", column_kind="kind",
                                          column_value="value", impute_function=impute)

        feat_col_names_ = features_.columns.to_numpy().tolist()

        #feat_mood = extract_relevant_features(df_feat_aux, y_mood, column_id="id", column_sort="time", column_kind="kind",
        #                                  column_value="value")
        #feat_happ = extract_relevant_features(df_feat_aux, y_happ, column_id="id", column_sort="time", column_kind="kind",
        #                                      column_value="value")
        #feat_acti = extract_relevant_features(df_feat_aux, y_acti, column_id="id", column_sort="time", column_kind="kind",
        #                                      column_value="value")

        feat_lab_dict = {'features': features_.to_numpy().tolist(),
                         'labels_m': y_mood.to_numpy().tolist(),
                         'labels_h': y_happ.to_numpy().tolist(),
                         'labels_a': y_acti.to_numpy().tolist(),
                         'feat_col_names': feat_col_names_
                         }
        return feat_lab_dict
    '''
    def extract_feat_stackedDF_backup(self):   # TODO: If you use this method, you do not need to call concat_features() afterwards
        def get_stacked_df(df, signal, freq):
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

            idxs_aux = idxs[0:len(idxs) - 1]
            init_ema_aux = (idxs[1:len(idxs)] - idxs_aux).reshape(-1, 1)
            init_ema_id = np.where(np.any(init_ema_aux > 1, axis=1))[0] + 1
            init_ema_indices = idxs[init_ema_id].tolist()

            counter = 1

            w_central = idxs[0]  # It is chosen first element: filtered list
            id_ = 0
            offset = (1 - self.overlap) * self.w_slide * freq

            ts_df = pd.DataFrame(columns=['id', 'time', 'kind', 'value'])
            ts_df_lab_mood = pd.DataFrame(columns=['id', 'label']) #TODO: Here you can create 3 dataframes and monitor three labels
            ts_df_lab_happ = pd.DataFrame(columns=['id', 'label'])
            ts_df_lab_acti = pd.DataFrame(columns=['id', 'label'])

            while id_ <= len(idxs):
                w_left = w_central - (self.w_slide * freq / 2 - 1)
                w_right = w_central + (self.w_slide * freq / 2 - 1)

                df_window = df.loc[df['index'].between(w_left, w_right)]
                label_m = int(df['label_m'][df['index'].iloc[w_central]])
                label_h = int(df['label_h'][df['index'].iloc[w_central]])
                label_a = int(df['label_a'][df['index'].iloc[w_central]])

                if signal == 'acc':
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

                elif signal == "eda":
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter, 'time': ts_, 'kind': "eda", 'value': data_}
                    aux_ = pd.DataFrame(data=data)
                elif signal == "temp":
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter, 'time': ts_, 'kind': "temp", 'value': data_}
                    aux_ = pd.DataFrame(data=data)

                # elif signal == "HR":
                else:
                    data_ = df_window[signal.lower()].to_numpy()
                    ts_ = np.arange(len(data_))
                    data = {'id': counter, 'time': ts_, 'kind': "hr", 'value': data_}
                    aux_ = pd.DataFrame(data=data)

                data_label_m = {'id': [counter], "label": [label_m]}
                data_label_h = {'id': [counter], "label": [label_h]}
                data_label_a = {'id': [counter], "label": [label_a]}

                aux_label_m = pd.DataFrame(data=data_label_m)
                aux_label_h = pd.DataFrame(data=data_label_h)
                aux_label_a = pd.DataFrame(data=data_label_a)

                ts_df = pd.concat([ts_df, aux_], axis=0)
                ts_df_lab_mood = pd.concat([ts_df_lab_mood, aux_label_m], axis=0)
                ts_df_lab_happ = pd.concat([ts_df_lab_happ, aux_label_h], axis=0)
                ts_df_lab_acti = pd.concat([ts_df_lab_acti, aux_label_a], axis=0)

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

            return ts_df, ts_df_lab_mood, ts_df_lab_happ, ts_df_lab_acti

        df_feat_aux = pd.DataFrame(columns=['id', 'time', 'kind', 'value'])
        for signal, freq in zip(self.signals, self.freqs):
            df_ = getattr(self, signal)
            df_ = df_.reset_index()
            columns = df_.columns
            if 'label_m' not in columns:
                continue
            else:
                df_feat_, df_lab_mood, df_lab_happ, df_lab_acti = get_stacked_df(df_, signal, freq)
                df_feat_aux = pd.concat([df_feat_aux, df_feat_])

        y_mood = pd.Series(data=df_lab_mood['label'].to_numpy(), index=df_lab_mood['id'].to_numpy())
        y_happ = pd.Series(data=df_lab_happ['label'].to_numpy(), index=df_lab_happ['id'].to_numpy())
        y_acti = pd.Series(data=df_lab_acti['label'].to_numpy(), index=df_lab_acti['id'].to_numpy())
        
        
        feat_mood = extract_relevant_features(df_feat_aux, y_mood, column_id="id", column_sort="time", column_kind="kind",
                                          column_value="value")
        feat_happ = extract_relevant_features(df_feat_aux, y_happ, column_id="id", column_sort="time", column_kind="kind",
                                              column_value="value")
        feat_acti = extract_relevant_features(df_feat_aux, y_acti, column_id="id", column_sort="time", column_kind="kind",
                                              column_value="value")

        feat_lab_dict = {'features_mood': feat_mood.to_numpy().tolist(),
                         'features_happ': feat_happ.to_numpy().tolist(),
                         'features_acti': feat_acti.to_numpy().tolist(),
                         'labels_m': y_mood.to_numpy().tolist(),
                         'labels_h': y_happ.to_numpy().tolist(),
                         'labels_a': y_acti.to_numpy().tolist()}
        return feat_lab_dict
    '''