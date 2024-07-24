## creating the datasets

```
python ./r_latplan_start_training.py -h
    positional arguments:
    {r_latplan,vanilla}   if vanilla or r-latplan
    {create_clean_traces,create_exp_data_sym,create_exp_data_im} type of task to be performed
    {hanoi,blocks,sokoban} domain name
    {complete,partial}    completness of the dataset, i.e. if missing transitions or not
    {clean,noisy}         if the dataset should be clean or noisy
    {erroneous,faultless} if the dataset should contain some mislabeled transitions
    {True,False}          if to use the transition identifier
    extra_label           add an extra label to the repo

    optional arguments:
    -h, --help            show this help message and exit
```

## training
```
python ./r_latplan_start_training.py -h

    usage: r_latplan_start_training.py [-h] [--pb_folder PB_FOLDER] {r_latplan,vanilla} {hanoi,blocks,sokoban}dataset_folder
    A script to train R-latplan for a specific experiment
    positional arguments:
    {r_latplan,vanilla}   if vanilla or r-latplan
    {hanoi,blocks,sokoban} domain name
    dataset_folder        folder where the images are
    optional arguments:
    -h, --help            show this help message and exit
    --pb_folder PB_FOLDER REQUIRED for PARTIAL

```
## testing
```
python ./r_latplan_testing.py
```