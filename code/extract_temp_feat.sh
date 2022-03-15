#!/bin/bash

python extract_temp_feat.py -p P4 -d 13 -w 60
python extract_temp_feat.py -p P1 P2 P3 P4 -d 9 15 13 13 -w 30
python extract_temp_feat.py -p P1 P2 P3 P4 -d 9 15 13 13 -w 120
