# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.11.23
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
from src.d03_features.TempFeatExtractor import TempFeatExtractor
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
    #patients = ['P3'] #TODO: Undo this when end debugging
    #days = [13]
    #weda = 60
    with tqdm(zip(days, patients), position=0) as t:
        for day, patient in t:
            t.set_description('Processing {} | number of days {} | Window: {}'.format(patient, str(day), str(weda)))
            f_obj = TempFeatExtractor(participant=patient, n_days=day, weda=weda, w_slide=60)
            f_obj.run()

    print("stop")


if __name__ == '__main__':
    main()