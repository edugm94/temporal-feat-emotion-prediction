# Wearable-based intelligent mental well-being monitoring of older adults during activities for daily life

This repository contains the code used to filter, preprocess and extract features to the released benchmark presented 
in paper: *"Wearable-based intelligent mental well-being monitoring of older adults during activities for daily life".*

This page shows a brief tutorial about how to use the code.

## Setup physiological signals
Initially, physiological signals are preprocessed, and it is assigned to each datapoint a label. Use `setup_data.py` to 
get signals filtered and labelled. And example of how to use this script is given below: 
  ```sh
  python setup_data.py -w 60 -p P1 -d 9
  ```
In the example above, it is prepared signals for participant P1 which has 9 days of sampling data. Also, labels 
(_happines_, _activeness_, and _mood_) will be extended within a window of 60 minutes (this is set by parameter `-w`).

The output of running this script consists of a set of .CSV files (one for each signal). The format of this .CSV file is
the following: 

| ts | HR | label_m | label_h | label_a |
| --- | --- | --- | --- |--- | 
| 1521197869.0 | 81.23 | -1 |	-1 | -1 |
1521197870.0 | 81.13 | -1 | -1 | -1
1521197871.0 | 81.03 | -1 | -1 | -1
1521197872.0 | 80.98 | -1 | -1 | -1
1521197873.0 | 80.88 | 2 | 3 | 3
1521197874.0 | 80.77 | 2 | 3 |3
1521197875.0 | 80.62 | 2 | 3 | 3
1521197876.0 | 80.43 | 2 | 3 | 3
1521197877.0 | 80.23 | 2 | 3 | 3

The example aboce corresponds to signal `HR`. The meaning of each column are the following: 
- **_ts_**: Timestamp corresponding to each data point. 
- **_HR_**: Signal value (in this example, the signal is Heart Rate (HR)).
- **_label_h_**: _Happiness_ level obtained from questionnaire. 
- _**label_a**_: _Activeness_ level obtained from questionnaire.
- **_label_m_**: Mood label interpolated from _happiness_ and _activeness_ levels. 
Value -1 represents lack of label for that data point.


**NOTE:** Remember to change the path to raw data. This path can be changed in: `src/d00_utils/constants.py`. Inside this 
file modify the variable `RAW_DATA_PATH`. 
## Extract Physiological and Temporal Features 


<!--
## Experiments Dataloaders 
 *TODO: To be defined whether dataloaders will be provided or not...*
-->
