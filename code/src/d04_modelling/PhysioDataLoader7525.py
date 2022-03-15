# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.30
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
import json
import numpy as np
from sklearn.model_selection import train_test_split
import os


class PhysioDataLoader7525:
    def __init__(self, participant, n_days, weda, experiment, label, seed):
        self.participant = participant
        self.n_days = n_days
        self.weda = weda
        self.experiment = experiment
        self.physio_path = 'data/03_features/physio-feat/'
        self.temp_path = 'data/03_features/temporal-feat/'
        self.label = label
        self.seed = seed
        self.mood_labels = []

    def run(self):
        if self.experiment == 'percentage':
            x_train, x_test, y_train, y_test = self.percentage_experiment()
            return x_train, x_test, y_train, y_test
        elif self.experiment == 'lodo':
            self.leave_day_out_experiment()

    def clean_dataset(self, discard=0.1):

        # Obtain an accounting of hbiw many vectors there is for each emotion
        unique, counts = np.unique(self.mood_labels, return_counts=True)
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
        indx_del_arr = np.where(self.mood_labels == emo_del_arr)[0]
        assert indx_del_arr.shape[0] == sum(
            emo_del_dict.values()), "The amount of vectors to delete does not match! Check it."

        return indx_del_arr

    def percentage_experiment(self):
        # seeds = [42, 13, 45, 23, 54] #TODO: It may be an attribute as well
        physio_feat_ = []  # Lists to store both physio and temporal features and labels for each day
        physio_lab_ = []

        for day in range(1, self.n_days + 1):
            path_physio_path_aux = self.physio_path + "{}/{}/{}/features.json".format(self.weda, self.participant,
                                                                                      str(day))
            with open(path_physio_path_aux, 'r') as f_json:
                aux_physio_dict = json.load(f_json)
                aux_physio_feat_ = np.asarray(aux_physio_dict["features"])


            if self.label == 'mood':
                aux_physio_lab_ = np.asarray(aux_physio_dict["labels_m"])

            elif self.label == "happiness":
                aux_physio_lab_ = np.asarray(aux_physio_dict["labels_h"])

            else:
                aux_physio_lab_ = np.asarray(aux_physio_dict["labels_a"])

            moods_ = np.asarray(aux_physio_dict["labels_m"])
            if moods_.shape == ():
                continue
            else:
                self.mood_labels.append(moods_)

            physio_feat_.append(aux_physio_feat_)
            physio_lab_.append(aux_physio_lab_)

        physio_feat_ = np.vstack(physio_feat_)
        physio_lab_ = np.vstack(physio_lab_)

        self.mood_labels = np.vstack(self.mood_labels)

        feats_ = physio_feat_
        labels_ = physio_lab_

        assert feats_.shape[0] == labels_.shape[0], "ERROR! Number of total rows for features labels" \
                                                    "must be the same"

        #Clean data set according to mood labels
        idx_clean_mood = self.clean_dataset()
        feats_clean = np.delete(feats_, idx_clean_mood, axis=0)
        labels_clean = np.delete(labels_, idx_clean_mood, axis=0)

        x_train, x_test, y_train, y_test = train_test_split(feats_clean, labels_clean, test_size=0.25,
                                                            random_state=self.seed)

        return x_train, x_test, y_train, y_test

    def leave_day_out_experiment(self):
        if self.feat_type == 0:
            aux_feat_path = self.physio_path + '{}/{}/'.format(self.weda, self.participant)
        for day in range(1, self.n_days+1):
            train_days = []
            for i in range(1, self.n_days+1):
                if i == day:
                    test_day = i
                else:
                    train_days.append(i)
            aux_test_path = aux_feat_path + str(test_day) + '/features.json'
            with open(aux_test_path, 'r') as f_json:
                aux_test_dict = json.load(f_json)
                x_test = np.asarray(aux_test_dict["features"])
                y_test_mood = np.asarray(aux_test_dict["labels_m"])
                y_test_happ = np.asarray(aux_test_dict["labels_h"])
                y_test_acti = np.asarray(aux_test_dict["labels_a"])

            if x_test.shape == ():
                continue

            train_feat_ = []
            train_lab_mood_ = []
            train_lab_hap_ = []
            train_lab_act_ = []
            for day_t in train_days:
                aux_train_path = aux_feat_path + str(day_t) + '/features.json'
                with open(aux_train_path, 'r') as f_json:
                    aux_train_dict = json.load(f_json)
                    x_train = np.asarray(aux_train_dict["features"])
                    y_train_mood = np.asarray(aux_train_dict["labels_m"])
                    y_train_happ = np.asarray(aux_train_dict["labels_h"])
                    y_train_acti = np.asarray(aux_train_dict["labels_a"])

                if x_train.shape == ():
                    continue
                else:
                    train_feat_.append(x_train)
                    train_lab_mood_.append(y_train_mood)
                    train_lab_hap_.append(y_train_happ)
                    train_lab_act_.append(y_train_acti)

            x_train = np.vstack(train_feat_)
            y_train_mood = np.vstack(train_lab_mood_)
            y_train_happ = np.vstack(train_lab_hap_)
            y_train_acti = np.vstack(train_lab_act_)

            dataset_dict = {
                "x_train": x_train.tolist(),
                "y_train_mood": y_train_mood.tolist(),
                "y_train_happ": y_train_happ.tolist(),
                "y_train_acti": y_train_acti.tolist(),
                "x_test": x_test.tolist(),
                "y_test_mood": y_test_mood.tolist(),
                "y_test_happ": y_test_happ.tolist(),
                "y_test_acti": y_test_acti.tolist()
            }
            if self.feat_type == 0:
                out_path = 'data/04_models/sets/physio/loso/{}/{}/{}/'.format(self.weda, self.participant, day)
            elif self.feat_type == 1:
                out_path = 'data/04_models/sets/tempo/loso/{}/{}/{}/'.format(self.weda, self.participant, day)
            else:
                out_path = 'data/04_models/sets/physio-tempo/loso/{}/{}/{}/'.format(self.weda, self.participant, day)
            out_filename = out_path + 'dataset.json'

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with open(out_filename, 'w') as json_f:
                json.dump(dataset_dict, json_f)
