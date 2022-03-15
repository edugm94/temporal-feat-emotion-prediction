# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.23
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
from src.d00_utils.constants import RAW_DATA_PATH
from src.d00_utils.filtering import compute_norm, filter_acc_signal, filter_eda_signal
import pandas as pd
import os


class SetupDataset:
    def __init__(self, participant, n_days, weda):
        self.participant = participant
        self.days = n_days
        self.weda = weda
        self.signals = ['acc', 'eda', 'temp', 'hr']
        self.acc = None
        self.eda = None
        self.temp = None
        self.hr = None

    def run(self):
        for day in range(1, self.days+1):
            self.load_signals(day=day)
            self.filter_signals()
            self.emotion_to_signal(day=day)

            out_path = "data/02_intermediate/{}/{}/{}/".format(
                str(self.weda), self.participant, str(day))
            out_filename_acc = out_path + "ACC.csv"
            out_filename_eda = out_path + "EDA.csv"
            out_filename_temp = out_path + "TEMP.csv"
            out_filename_hr = out_path + "HR.csv"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.acc.to_csv(out_filename_acc, header=True, index=False, sep='\t', mode='w')
            self.eda.to_csv(out_filename_eda, header=True, index=False, sep='\t', mode='w')
            self.temp.to_csv(out_filename_temp, header=True, index=False, sep='\t', mode='w')
            self.hr.to_csv(out_filename_hr, header=True, index=False, sep='\t', mode='w')

    def load_signals(self, day): # This function loads into an attribute a corresponding signal
        path = RAW_DATA_PATH + "{}/{}_Complete/".format(self.participant, self.participant)
        acc_path = path + "ACC{}.csv".format(day)
        eda_path = path + "EDA{}.csv".format(day)
        temp_path = path + "TEMP{}.csv".format(day)
        hr_path = path + "HR{}.csv".format(day)

        def get_acc_df(acc_path):
            f = 32
            Ts = 1/f
            acc = pd.read_csv(acc_path, names=['x', 'y', 'z'])
            init_timestamp = acc.iloc[0, :][0]
            acc = acc.drop([0, 1]).reset_index(drop=True).reset_index()
            acc['ts'] = init_timestamp + acc['index'] * Ts
            cols = ['ts', 'x', 'y', 'z']
            acc = acc[cols]

            # Synchro with HR signal
            actual_ts = init_timestamp + 10
            acc = acc[acc['ts'] >= actual_ts]

            # Get normalized signal
            acc['n'] = compute_norm(acc)
            return acc

        def get_eda_df(eda_path):
            f = 4
            Ts = 1 / f
            eda = pd.read_csv(eda_path, names=['eda'])
            init_timestamp = eda.iloc[0, :][0]
            eda = eda.drop([0, 1]).reset_index(drop=True).reset_index()
            eda['ts'] = init_timestamp + eda['index'] * Ts
            cols = ['ts', 'eda']
            eda = eda[cols]

            # Synchro with HR signal
            actual_ts = init_timestamp + 10
            eda = eda[eda['ts'] >= actual_ts]
            return eda

        def get_temp_df(temp_path):
            f = 4
            Ts = 1 / f
            temp = pd.read_csv(temp_path, names=['temp'])
            init_timestamp = temp.iloc[0, :][0]
            temp = temp.drop([0, 1]).reset_index(drop=True).reset_index()
            temp['ts'] = init_timestamp + temp['index'] * Ts
            cols = ['ts', 'temp']
            temp = temp[cols]

            # Synchro with HR signal
            actual_ts = init_timestamp + 10
            temp = temp[temp['ts'] >= actual_ts]
            return temp

        def get_hr_df(hr_path):
            f = 1
            Ts = 1 / f
            hr = pd.read_csv(hr_path, names=['hr'])
            init_timestamp = hr.iloc[0, :][0]
            hr = hr.drop([0, 1]).reset_index(drop=True).reset_index()
            hr['ts'] = init_timestamp + hr['index'] * Ts
            cols = ['ts', 'hr']
            hr = hr[cols]
            return hr

        self.acc = get_acc_df(acc_path)
        self.eda = get_eda_df(eda_path)
        self.temp = get_temp_df(temp_path)
        self.hr = get_hr_df(hr_path)

    def filter_signals(self):  # To filter each of the signals
        for signal in self.signals:
            if signal == 'acc':
                self.acc['x'] = filter_acc_signal(self.acc['x'])
                self.acc['y'] = filter_acc_signal(self.acc['y'])
                self.acc['z'] = filter_acc_signal(self.acc['z'])
                self.acc['n'] = filter_acc_signal(self.acc['n'])
            elif signal == 'eda':
                self.eda['eda'] = filter_eda_signal(self.eda['eda'])
            else:
                continue

    def emotion_to_signal(self, day):  # assign to each signal an emotion
        WINDOW = self.weda * 60   # EMA window converted to seconds
        path_EMA = RAW_DATA_PATH + "{}/{}_Complete/EMAs{}.xlsx".format(self.participant, self.participant, day)
        # path to EMA
        # Ask for dimension / or compute all dimensions at the same time
        df_ema = pd.read_excel(path_EMA, engine='openpyxl').iloc[:, 0:8].dropna(how='all')

        init_ts = self.acc['ts'].iloc[0]
        ts_aux = df_ema.iloc[:, 4]
        timestamps = ts_aux + init_ts

        timestamps = timestamps.dropna()
        end_ts = timestamps.iloc[-1]
        timestamps = timestamps[1:-1]  # Delete first and last timestamp (reference)

        # be aware that EMA excel may be empty
        if timestamps.empty:
            pass
        else:
            # Assign -1 to all labels and dataframes
            self.acc['label_m'] = -1
            self.acc['label_h'] = -1
            self.acc['label_a'] = -1

            self.eda['label_m'] = -1
            self.eda['label_h'] = -1
            self.eda['label_a'] = -1

            self.temp['label_m'] = -1
            self.temp['label_h'] = -1
            self.temp['label_a'] = -1

            self.hr['label_m'] = -1
            self.hr['label_h'] = -1
            self.hr['label_a'] = -1

            for idx, ts in enumerate(timestamps):
                # You could try to do all labels at the same time in this script to optimize
                mood = int(df_ema.iloc[idx + 1, 7])
                happ = int(df_ema.iloc[idx + 1, 5])
                act = int(df_ema.iloc[idx + 1, 6])

                # En estos if definir el upper y lower bound para cada uno de los casos.
                if idx == 0:
                    ts_pos = timestamps.iloc[idx + 1]
                    ts_pre = None

                    l_dis = ts - init_ts
                    r_dis = ts_pos - ts
                    l_bound = init_ts if l_dis < WINDOW / 2 else ts - WINDOW / 2
                    r_bound = ts + r_dis / 2 if r_dis < WINDOW else ts + WINDOW / 2

                elif idx == len(timestamps) - 1:
                    ts_pre = timestamps.iloc[idx - 1]
                    ts_pos = None

                    l_dis = ts - ts_pre
                    r_dis = end_ts - ts
                    l_bound = ts - l_dis / 2 if l_dis < WINDOW else ts - WINDOW / 2
                    r_bound = end_ts if r_dis < WINDOW / 2 else ts + WINDOW / 2

                else:
                    ts_pre = timestamps.iloc[idx - 1]
                    ts_pos = timestamps.iloc[idx + 1]

                    l_dis = ts - ts_pre
                    r_dis = ts_pos - ts
                    l_bound = ts - l_dis / 2 if l_dis < WINDOW else ts - WINDOW / 2
                    r_bound = ts + r_dis / 2 if r_dis < WINDOW else ts + WINDOW / 2

                # Add label to each colum within the dataframe
                self.acc.loc[self.acc['ts'].between(l_bound, r_bound), 'label_m'] = mood
                self.acc.loc[self.acc['ts'].between(l_bound, r_bound), 'label_h'] = happ
                self.acc.loc[self.acc['ts'].between(l_bound, r_bound), 'label_a'] = act

                self.eda.loc[self.eda['ts'].between(l_bound, r_bound), 'label_m'] = mood
                self.eda.loc[self.eda['ts'].between(l_bound, r_bound), 'label_h'] = happ
                self.eda.loc[self.eda['ts'].between(l_bound, r_bound), 'label_a'] = act

                self.temp.loc[self.temp['ts'].between(l_bound, r_bound), 'label_m'] = mood
                self.temp.loc[self.temp['ts'].between(l_bound, r_bound), 'label_h'] = happ
                self.temp.loc[self.temp['ts'].between(l_bound, r_bound), 'label_a'] = act

                self.hr.loc[self.hr['ts'].between(l_bound, r_bound), 'label_m'] = mood
                self.hr.loc[self.hr['ts'].between(l_bound, r_bound), 'label_h'] = happ
                self.hr.loc[self.hr['ts'].between(l_bound, r_bound), 'label_a'] = act
