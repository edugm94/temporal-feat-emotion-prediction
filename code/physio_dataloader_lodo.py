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
from src.d04_modelling.PhysioDataLoaderLODO import PhysioDataLoaderLODO
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

    #patients = ['P4']
    #days = [13]
    #label = 'happiness'
    for patient, day in zip(patients, days):

        '''
        if feat_type == 0:
            out_path = 'data/04_models/sets/physio/75-25/{}/{}/{}/'.format(label, weda, patient)
        elif feat_type == 1:
            out_path = 'data/04_models/sets/tempo/75-25/{}/{}/{}/'.format(label, weda, patient)
        else:
            out_path = 'data/04_models/sets/physio-tempo/75-25/{}/{}/{}/'.format(label, weda, patient)
        '''

        dl_obj = PhysioDataLoaderLODO(
            participant=patient,
            n_days=day,
            weda=weda,
            label=label
        )
        dl_obj.run()


if __name__ == '__main__':
    main()
