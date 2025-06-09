import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.tree import _tree
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data



path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/sokoban/sokoban_complete_clean_faultless_withoutTI_N25/data.p"

# load dataset for the specific experiment
loaded_data = load_dataset(path_to_dataset)
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

all_high_lvl_actions_unique = loaded_data["all_high_lvl_actions_unique"]


# GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION
dico_transitions_per_high_lvl_actions = {}

print(len(train_set_no_dupp))
print(train_set_no_dupp[0][2])
# ele[1] = low level index
# ele[2] = high level index


dico_numberHighLvl_name = {}


# loop over the train set (without dupplicate)
# AND group the transitions into their respective High Level Actions
# [all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]]
for ii, ele in enumerate(train_set_no_dupp):

    # if the looped transition high level action is not a key of dico_transitions_per_high_lvl_actions
    if all_high_lvl_actions_unique[int(np.argmax(ele[2]))] not in dico_transitions_per_high_lvl_actions:

        # add the key (the high lvl action index)
        dico_transitions_per_high_lvl_actions[all_high_lvl_actions_unique[int(np.argmax(ele[2]))]] = {}

        dico_numberHighLvl_name[int(np.argmax(ele[2]))] = all_high_lvl_actions_unique[int(np.argmax(ele[2]))]

    # if the hash is not in the keys of the dico of the high lvl action
    # i.e. if the transition was not added as a transition for this high lvl action
    if int(np.argmax(ele[1])) not in dico_transitions_per_high_lvl_actions[all_high_lvl_actions_unique[int(np.argmax(ele[2]))]]:
        
        dico_transitions_per_high_lvl_actions[all_high_lvl_actions_unique[int(np.argmax(ele[2]))]][int(np.argmax(ele[1]))] = {
            #"preprocessed" : ele[0],
            "preprocessed": "aaaa"
            # "reduced" : train_set_no_dupp_orig[ii][0],
            # "onehot": ele[2],
            #"max_diff_in_bits_": ele[2]
        }

# print(dico_transitions_per_high_lvl_actions[0].keys())



with open("highLvlLowlvl.json", "w") as json_file:
    json.dump(dico_transitions_per_high_lvl_actions, json_file, indent=4) 

with open("highLvlLowlvlNames.json", "w") as json_file:
    json.dump(dico_numberHighLvl_name, json_file, indent=4) 


# 
exit()
