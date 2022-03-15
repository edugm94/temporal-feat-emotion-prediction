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
In the example above, it is prepared signals for participant P1 which has 9 days of sampling data. The pare  
## Extract Physiological and Temporal Features 


## Experiments Dataloaders 
 *TODO: To be defined whether dataloaders will be provided or not...*

