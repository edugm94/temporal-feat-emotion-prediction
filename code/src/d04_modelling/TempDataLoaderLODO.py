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


class TempDataLoaderLODO:
    def __init__(self, participant, n_days, weda, label):
        self.participant = participant
        self.n_days = n_days
        self.weda = weda
        self.label = label
        self.temp_path = 'data/03_features/temporal-feat/'
        self.experiment = 'lodo'

    def run(self):
        feat_type = 'stable'
        if feat_type == 'stable':
            self.__get_stable_features()

    def __get_stable_features(self):
        n_chunks = 3    # Number of chunks to trunk training data for searching stable feat
        for test_day in range(1, self.n_days + 1):
            x_train, y_train, x_test, y_test, col_names = self.__get_train_test_sets(test_day)

            if np.asarray(x_test).shape == ():
                dict_ = {
                    'x_train': -1,
                    'x_test': -1,
                    'y_train': -1,
                    'y_test': -1,
                    'feat_col_names': -1
                }

                outpath_ = 'data/04_models/sets/temp-stable/{}/{}/{}/{}/'.format(self.experiment, self.label, self.weda,
                                                                                 self.participant)
                if not os.path.exists(outpath_):
                    os.makedirs(outpath_)

                filename = outpath_ + 'dataset{}.json'.format(str(test_day))
                with open(filename, 'w') as json_f:
                    json.dump(dict_, json_f)
                continue

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            print("Searching stable features for testing day {} from participant "
                  "{}".format(test_day, self.participant))


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
                y_chunk_series = pd.Series(data=y_chunk.reshape(-1, ), index=np.arange(0, y_chunk.shape[0]))
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

            filename = outpath_ + 'dataset{}.json'.format(str(test_day))
            with open(filename, 'w') as json_f:
                json.dump(dict_, json_f)


    def __get_train_test_sets(self, test_day):
        feat_path = self.temp_path + '{}/{}/'.format(self.weda, self.participant)
        training_days = np.arange(1, self.n_days + 1).tolist()
        training_days.remove(test_day)

        test_feat_path = feat_path + str(test_day) + '/features.json'
        with open(test_feat_path, 'r') as f_json:
            test_dict_ = json.load(f_json)

        x_test = np.asarray(test_dict_["features"])
        col_template = test_dict_['feat_col_names']

        if x_test.shape == ():  # There is no testing data available
            return -1, -1, -1, -1, -1

        if self.label == 'mood':
            y_test = np.asarray(test_dict_["labels_m"])
            y_test_mood = np.asarray(test_dict_["labels_m"])
        elif self.label == 'happiness':
            y_test = np.asarray(test_dict_["labels_h"])
            y_test_mood = np.asarray(test_dict_["labels_m"])
        else:
            y_test = np.asarray(test_dict_["labels_a"])
            y_test_mood = np.asarray(test_dict_["labels_m"])

        train_feat_ = []
        train_lab_ = []
        y_train_mood = []
        for training_day in training_days:
            train_feat_path = feat_path + str(training_day) + '/features.json'
            with open(train_feat_path, 'r') as f_json:
                train_dict_ = json.load(f_json)
            x_train_day = np.asarray(train_dict_['features'])
            if x_train_day.shape == ():    # In case training data is empty
                continue
            col_names = train_dict_['feat_col_names']
            df_ = pd.DataFrame(data=x_train_day, columns=col_names)
            df_sorted = df_[col_template]
            x_train_day_sorted = df_sorted.to_numpy()

            if self.label == 'mood':
                y_train_aux = np.asarray(train_dict_["labels_m"])
                y_train_mood_aux = np.asarray(train_dict_["labels_m"])
            elif self.label == 'happiness':
                y_train_aux = np.asarray(train_dict_["labels_h"])
                y_train_mood_aux = np.asarray(train_dict_["labels_m"])
            else:
                y_train_aux = np.asarray(train_dict_["labels_a"])
                y_train_mood_aux = np.asarray(train_dict_["labels_m"])

            if x_train_day_sorted.shape == ():
                continue
            else:
                train_feat_.append(x_train_day_sorted)
                train_lab_.append(y_train_aux)
                y_train_mood.append(y_train_mood_aux)
        x_train = np.vstack(train_feat_)
        y_train = np.hstack(train_lab_)
        y_train_mood = np.hstack(y_train_mood)

        # Cleaning non-representative labels
        discard = 0.1  # Percentage to discard labels
        labels_ = np.concatenate((y_train_mood, y_test_mood), axis=0)
        unique, counts = np.unique(labels_, return_counts=True)
        counting = dict(zip(unique, counts))

        # Get the total amount of vectors and the threshold to filter dictionary
        tot = sum(counting.values())
        threshold = tot * discard

        # Get a dictionary with the emotions that should be cleaned from the initial variables
        # It is kept a dictionary to check the lenght of the cleaned values at the end
        emo_del_dict = dict(filter(lambda elem: elem[1] < threshold, counting.items()))
        # Array that contains the value of the emotions to be cleaned in the "labels" variable
        emo_del_arr = np.array(list(emo_del_dict.keys()))


        id_test = []
        id_train = []
        [id_test.append(np.where(y_test_mood == id)[0]) for id in list(emo_del_arr)]
        id_test = np.empty([]) if id_test == [] else np.hstack(id_test)
        [id_train.append(np.where(y_train_mood == id)[0]) for id in list(emo_del_arr)]
        id_train = np.empty([]) if id_train == [] else np.hstack(id_train)

        if id_train.size != 0:
            x_train_clean = np.delete(x_train, id_train, axis=0)
            y_train_clean = np.delete(y_train, id_train, axis=0)
        else:
            x_train_clean = x_train
            y_train_clean = y_train

        if id_test.size != 0:
            x_test_clean = np.delete(x_test, id_test, axis=0)
            y_test_clean = np.delete(y_test, id_test, axis=0)
        else:
            x_test_clean = x_test
            y_test_clean = y_test

        return x_train_clean, y_train_clean, x_test_clean, y_test_clean, col_template
