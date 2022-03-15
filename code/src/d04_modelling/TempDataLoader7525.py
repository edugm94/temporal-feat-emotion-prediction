# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2022.02.16
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden

import json
import numpy as np
from src.d00_utils.feat_utils import clean_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tsfresh import select_features
import os


class TempDataLoader7525:
    def __init__(self, participant, n_days, weda, label):
        self.participant = participant
        self.n_days = n_days
        self.weda = weda
        self.temp_path = 'data/03_features/temporal-feat/'
        self.label = label
        self.seeds = [42, 25, 56]
        #self.seeds = [42]
        self.mood_labels = []
        self.n_exp = len(self.seeds)
        self.experiment = '75-25'

    def run(self):
        feat_type = 'stable'
        df_, labels_, col_names = self._stackDataFrame()    # Cleaned feature matrix (based on non-representative moods)
        if feat_type == 'stable':
            self.__get_stable_features(df_, labels_, col_names)

    def __get_stable_features(self, df, labels, col_names):
        n_chunks = 3    # Number of submatrices the 75 of training data will be split to search stable feat
        for id_, seed in enumerate(self.seeds):
            print("Setting up dataset number {}...\n".format(id_ + 1))
            x_train, x_test, y_train, y_test = train_test_split(df.to_numpy(), labels,
                                                                test_size=0.25, random_state=seed)
            df_x_train = pd.DataFrame(data=x_train, columns=col_names)
            df_x_test = pd.DataFrame(data=x_test, columns=col_names)

            # Training data is chunked to search stable features
            # Chunking data consecutively
            #chunks_x = np.array_split(x_train, indices_or_sections=n_chunks, axis=0)
            #chunks_y = np.array_split(y_train, indices_or_sections=n_chunks, axis=0)

            # Chunking data randomly

            #np.random.seed(seed=254)
            n_samples = x_train.shape[0]
            idx_ = np.arange(n_samples)
            np.random.shuffle(idx_)
            x_train_shuffled = x_train[idx_, :]
            y_train_shuffled = y_train[idx_, :]
            
            chunks_x = np.array_split(x_train_shuffled, indices_or_sections=n_chunks, axis=0)
            chunks_y = np.array_split(y_train_shuffled, indices_or_sections=n_chunks, axis=0)


            """
            chunks_x = []
            chunks_y = []
            number_of_rows = int(x_train.shape[0] / 2)
            for chunk_id in range(1, n_chunks + 1):
                random_indices = np.random.choice(x_train.shape[0], size=number_of_rows, replace=False)
                x_chunk_ = x_train[random_indices, :]
                y_chunk_ = y_train[random_indices, :]
                chunks_x.append(x_chunk_)
                chunks_y.append(y_chunk_)
            """
            selected_feat_chunks = []
            for x_chunk, y_chunk in zip(chunks_x, chunks_y):
                x_chunk_df = pd.DataFrame(data=x_chunk, columns=col_names)
                y_chunk_series = pd.Series(data=y_chunk.reshape(-1,), index=np.arange(0, y_chunk.shape[0]))
                x_chunk_selected = select_features(x_chunk_df, y_chunk_series)
                selected_feat = x_chunk_selected.columns.to_numpy().tolist()
                selected_feat_chunks.append(selected_feat)

            stable_feat = list(set.intersection(*map(set, selected_feat_chunks)))
            df_x_train_stable = df_x_train[stable_feat]
            df_x_test_stable = df_x_test[stable_feat]

            x_train_stable = df_x_train_stable.to_numpy()
            x_test_stable = df_x_test_stable.to_numpy()


            dict_ = {
                'x_train': x_train_stable.tolist(),
                'x_test': x_test_stable.tolist(),
                'y_train': y_train.tolist(),
                'y_test': y_test.tolist(),
                'feat_col_names': stable_feat
            }

            outpath_ = 'data/04_models/sets/temp-stable/{}/{}/{}/{}/'.format(self.experiment, self.label, self.weda,
                                                                                 self.participant)
            if not os.path.exists(outpath_):
                os.makedirs(outpath_)

            filename = outpath_ + 'dataset{}.json'.format(str(id_ + 1))
            with open(filename, 'w') as json_f:
                json.dump(dict_, json_f)

    def __get_selected_features(self, df, labels, col_names):
        pass

    def _stackDataFrame(self):
        """Method to stack each day's feature matrix.
        It is needed to organize columns since extract_features() method can change column order."""
        # global df_stacked
        features_ = []
        labels_ = []
        labels_mood = []  # Auxiliary list to clean dataset according to mood states
        template_flag = False
        for day in range(1, self.n_days + 1):
            path_ = self.temp_path + '{}/{}/{}/features.json'.format(self.weda, self.participant, str(day))

            with open(path_, 'r') as json_f:
                dict_ = json.load(json_f)
                feat_ = np.asarray(dict_['features'])
                labels_m = np.asarray(dict_['labels_m']).reshape(-1, 1)
                labels_h = np.asarray(dict_['labels_h']).reshape(-1, 1)
                labels_a = np.asarray(dict_['labels_a']).reshape(-1, 1)
                col_names = dict_['feat_col_names']
                if feat_.shape == ():
                    template_flag = True
                    continue

            if day == 1:
                col_template = col_names  # It is taken first day as feature order template
                df_ = pd.DataFrame(data=feat_, columns=col_names)
                df_stacked = df_
            else:
                if template_flag:
                    col_template = col_names
                    df_ = pd.DataFrame(data=feat_, columns=col_names)
                    df_stacked = df_
                    template_flag = False
                else:
                    df_ = pd.DataFrame(data=feat_, columns=col_names)
                    df_sorted = df_[col_template]
                    df_stacked = pd.concat([df_stacked, df_sorted])

            if self.label == 'mood':
                labels_.append(labels_m)
                labels_mood.append(labels_m)
            elif self.label == 'happiness':
                labels_.append(labels_h)
                labels_mood.append(labels_m)
            else:
                labels_.append(labels_a)
                labels_mood.append(labels_m)

        df_stacked = df_stacked.reset_index(drop=True)
        labels_ = np.vstack(labels_)
        labels_mood = np.vstack(labels_mood)
        # Clean dataset
        ind = clean_dataset(labels_mood)

        labels_clean = np.delete(labels_, ind, axis=0)
        df_stacked_clean = df_stacked.drop(df_stacked.index[ind]).reset_index(drop=True)

        return df_stacked_clean, labels_clean, col_template