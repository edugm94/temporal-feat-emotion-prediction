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
**NOTE:** You can input several participants with the argument ``-p``, e.g: `-p P1 P2 P3 P4`. In this case you need to 
specify a list of the corresponding number of sapling days for each participant `-d 9 15 13 13`, which order correspond 
with the list of participants, i.e. participant P1 has 9 sampling days; participant P2 has 15 sampling days, etc.

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
It is provided two scripts to extract physiological and temporal features. One can use these scripts as follow. 
To extract physiological features: 
  ```sh
  python extract_feat.py -w 60 -p P1 -d 9
  ```
To extract temporal features: 
  ```sh
  python extract_feat_temp.py -w 60 -p P1 -d 9
  ``` 
**NOTE:** You can input a list of participants and number of sampling days as explained before. 


Please, if you use the code provided and found relevant the work presented in this paper, cite the following reference in your work: 

```
@article{gutierrez2023wearable,
  title={Wearable-Based Intelligent Emotion Monitoring in Older Adults during Daily Life Activities},
  author={Gutierrez Maestro, Eduardo and De Almeida, Tiago Rodrigues and Schaffernicht, Erik and Martinez Mozos, {\'O}scar},
  journal={Applied Sciences},
  volume={13},
  number={9},
  pages={5637},
  year={2023},
  publisher={MDPI}
}
```

<!--
## Experiments Dataloaders 
 *TODO: To be defined whether dataloaders will be provided or not...*
-->
