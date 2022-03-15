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
from src.d04_modelling.TempDataLoader7525 import TempDataLoader7525
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
    #patients = ['P3']
    #ays = [13]

    for patient, day in zip(patients, days):

        dl_obj = TempDataLoader7525(
            participant=patient,
            n_days=day,
            weda=weda,
            label=label
        )

        dl_obj.run()
        #dl_obj.all_feat_sets()
        #dl_obj.selected_feat_sets()
        #dl_obj.stable_feat_sets()


        #dl_obj.physio_selected_feat_sets()
        #dl_obj.physio_stable_feat_sets()
            #aux_dict = dict()
            #aux_dict['x_train'] = x_train.tolist()
            #aux_dict['x_test'] = x_test.tolist()
            #aux_dict['y_train'] = y_train.tolist()
            #aux_dict['y_test'] = y_test.tolist()

            #out_filename = out_path + 'dataset{}.json'.format(idx)
            #if not os.path.exists(out_path):
            #    os.makedirs(out_path)

            #with open(out_filename, 'w') as json_f:
            #    json.dump(aux_dict, json_f)


if __name__ == '__main__':
    main()

