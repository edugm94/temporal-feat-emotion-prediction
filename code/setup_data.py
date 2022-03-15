# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.23
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
from src.d02_intermediate.SetupDataset import SetupDataset
import argparse
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser(description='This script filter and assign labels to each data point according to '
                                                 'weda parameter. It will be computed for the list of input patients')
    parser.add_argument('-p', '--patients', nargs="*", help='Patient indicator, e.g.: P1',
                        default=['P1', 'P2', 'P3', 'P4'], type=str, action='store')
    parser.add_argument('-d', '--days', nargs="*", help='Number of sampling days for the indicated patient.',
                        default=[9, 15, 13, 13], type=str, action='store')
    parser.add_argument('-w', '--weda', help='Window size to extend the emotion in time axis.', default=60, type=int)

    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    days = args.days
    patients = args.patients
    weda = args.weda

    with tqdm(zip(days, patients), position=0) as t:
        for day, patient in t:
            t.set_description('Processing {} | number of days {}'.format(patient, day))
            ds_obj = SetupDataset(participant=patient, n_days=day, weda=weda)
            ds_obj.run()


if __name__ == '__main__':
    main()


