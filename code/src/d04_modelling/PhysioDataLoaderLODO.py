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
import os

class PhysioDataLoaderLODO:
    def __init__(self, participant, n_days, weda, label):
        self.participant = participant
        self.n_days = n_days
        self.weda = weda
        self.experiment = 'lodo'
        self.physio_path = 'data/03_features/physio-feat/'
        self.label = label
        self.mood_labels = []

    def run(self):
        #self.leave_day_out_experiment()
        self.leave_day_out_experiment_clean()


    def leave_day_out_experiment(self):
        aux_feat_path = self.physio_path + '{}/{}/'.format(self.weda, self.participant)

        for day in range(1, self.n_days+1):
            train_days = []
            for i in range(1, self.n_days+1):
                if i == day:
                    test_day = i
                    aux_test_path = aux_feat_path + str(test_day) + '/features.json'
                else:
                    train_days.append(i)

            with open(aux_test_path, 'r') as f_json:
                aux_test_dict = json.load(f_json)
                x_test = np.asarray(aux_test_dict["features"])

                if self.label == 'mood':
                    y_test = np.asarray(aux_test_dict["labels_m"])
                elif self.label == 'happiness':
                    y_test = np.asarray(aux_test_dict["labels_h"])
                else:
                    y_test = np.asarray(aux_test_dict["labels_a"])

            if x_test.shape == ():
                continue


            train_feat_ = []
            train_lab_ = []
            for day_t in train_days:
                aux_train_path = aux_feat_path + str(day_t) + '/features.json'
                with open(aux_train_path, 'r') as f_json:
                    aux_train_dict = json.load(f_json)
                    x_train_aux = np.asarray(aux_train_dict["features"])

                    if self.label == 'mood':
                        y_train_aux = np.asarray(aux_train_dict["labels_m"])
                    elif self.label == 'happiness':
                        y_train_aux = np.asarray(aux_train_dict["labels_h"])
                    else:
                        y_train_aux = np.asarray(aux_train_dict["labels_a"])

                if x_train_aux.shape == ():
                    continue
                else:
                    train_feat_.append(x_train_aux)
                    train_lab_.append(y_train_aux)

            x_train = np.vstack(train_feat_)
            y_train = np.vstack(train_lab_)

            dataset_dict = {
                "x_train": x_train.tolist(),
                "x_test": x_test.tolist(),
                "y_train": y_train.tolist(),
                "y_test": y_test.tolist()
            }

            out_path = 'data/04_models/sets/physio/lodo/{}/{}/{}/'.format(self.label, self.weda, self.participant)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            out_filename = out_path + 'dataset{}.json'.format(day)

            with open(out_filename, 'w') as json_f:
                json.dump(dataset_dict, json_f)

    def leave_day_out_experiment_clean(self):
        aux_feat_path = self.physio_path + '{}/{}/'.format(self.weda, self.participant)

        for day in range(1, self.n_days + 1):
            train_days = []
            for i in range(1, self.n_days + 1):
                if i == day:
                    test_day = i
                    aux_test_path = aux_feat_path + str(test_day) + '/features.json'
                else:
                    train_days.append(i)

            with open(aux_test_path, 'r') as f_json:
                aux_test_dict = json.load(f_json)
                x_test = np.asarray(aux_test_dict["features"])

                if self.label == 'mood':
                    y_test = np.asarray(aux_test_dict["labels_m"])
                    y_test_mood = np.asarray(aux_test_dict["labels_m"])
                elif self.label == 'happiness':
                    y_test = np.asarray(aux_test_dict["labels_h"])
                    y_test_mood = np.asarray(aux_test_dict["labels_m"])
                else:
                    y_test = np.asarray(aux_test_dict["labels_a"])
                    y_test_mood = np.asarray(aux_test_dict["labels_m"])

            if x_test.shape == ():
                dataset_dict = {
                    "x_train": -1,
                    "x_test": -1,
                    "y_train": -1,
                    "y_test": -1
                }

                out_path = 'data/04_models/sets/physio/lodo/{}/{}/{}/'.format(self.label, self.weda, self.participant)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                out_filename = out_path + 'dataset{}.json'.format(day)

                with open(out_filename, 'w') as json_f:
                    json.dump(dataset_dict, json_f)
                continue

            train_feat_ = []
            train_lab_ = []
            y_train_mood = []
            for day_t in train_days:
                aux_train_path = aux_feat_path + str(day_t) + '/features.json'
                with open(aux_train_path, 'r') as f_json:
                    aux_train_dict = json.load(f_json)
                    x_train_aux = np.asarray(aux_train_dict["features"])

                    if self.label == 'mood':
                        y_train_aux = np.asarray(aux_train_dict["labels_m"])
                        y_train_mood_aux = np.asarray(aux_train_dict["labels_m"])
                    elif self.label == 'happiness':
                        y_train_aux = np.asarray(aux_train_dict["labels_h"])
                        y_train_mood_aux = np.asarray(aux_train_dict["labels_m"])
                    else:
                        y_train_aux = np.asarray(aux_train_dict["labels_a"])
                        y_train_mood_aux = np.asarray(aux_train_dict["labels_m"])

                if x_train_aux.shape == ():
                    continue
                else:
                    train_feat_.append(x_train_aux)
                    train_lab_.append(y_train_aux)
                    y_train_mood.append(y_train_mood_aux)

            x_train = np.vstack(train_feat_)
            y_train = np.vstack(train_lab_)
            y_train_mood = np.vstack(y_train_mood)

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
            id_test = np.hstack(id_test)
            [id_train.append(np.where(y_train_mood == id)[0]) for id in list(emo_del_arr)]
            id_train = np.hstack(id_train)

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

            #x_train_clean = np.delete(x_train, id_train, axis=0)
            #y_train_clean = np.delete(y_train, id_train, axis=0)
            #x_test_clean = np.delete(x_test, id_test, axis=0)
            #y_test_clean = np.delete(y_test, id_test, axis=0)

            dataset_dict = {
                "x_train": x_train_clean.tolist(),
                "x_test": x_test_clean.tolist(),
                "y_train": y_train_clean.tolist(),
                "y_test": y_test_clean.tolist()
            }

            out_path = 'data/04_models/sets/physio/lodo/{}/{}/{}/'.format(self.label, self.weda, self.participant)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            out_filename = out_path + 'dataset{}.json'.format(day)

            with open(out_filename, 'w') as json_f:
                json.dump(dataset_dict, json_f)