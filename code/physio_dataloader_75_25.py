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
from src.d04_modelling.PhysioDataLoader7525 import PhysioDataLoader7525
import os
import json
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='This script filter and assign labels to each data point according to '
                                                 'weda parameter. It will be computed for the list of input patients')
    parser.add_argument('-p', '--patients', nargs="*", help='Patient indicator, e.g.: P1',
                        default=['P1', 'P2', 'P3', 'P4'], type=str, action='store')
    parser.add_argument('-d', '--days', nargs="*", help='Number of sampling days for the indicated patient.',
                        default=[9, 15, 13, 13], type=int, action='store')
    parser.add_argument('-w', '--weda', help='Window size to extend the emotion in time axis.', default=60, type=int)
    parser.add_argument('-l', '--label', help='Dimension to label vectors.', default='mood', type=str)

    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    days = args.days
    patients = args.patients
    weda = args.weda
    label = args.label
    #seeds = [42, 25, 56, 12, 69]
    seeds = [42, 25, 56]

    for patient, day in zip(patients, days):
        for idx, seed in enumerate(seeds, 1):

            out_path = 'data/04_models/sets/physio/75-25/{}/{}/{}/'.format(label, weda, patient)

            dl_obj = PhysioDataLoader7525(
                participant=patient,
                n_days=day,
                weda=weda,
                experiment='percentage',
                label=label,
                seed=seed
            )
            x_train, x_test, y_train, y_test = dl_obj.run()

            aux_dict = dict()
            aux_dict['x_train'] = x_train.tolist()
            aux_dict['x_test'] = x_test.tolist()
            aux_dict['y_train'] = y_train.tolist()
            aux_dict['y_test'] = y_test.tolist()

            out_filename = out_path + 'dataset{}.json'.format(idx)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with open(out_filename, 'w') as json_f:
                json.dump(aux_dict, json_f)


if __name__ == '__main__':
    main()

