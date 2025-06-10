# Algo that retrieve all transitions from the data.p
# and put the departing/arriving images on same row, along 
# with the action's number
# save everything as a numpy array in a pickle file


import numpy as np
import pandas as pd
import re
from collections import defaultdict
import os
import sys
from itertools import combinations, islice
import time
import itertools
from tqdm import tqdm
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import math
from typing import List
from math import comb
from itertools import islice
import argparse





import subprocess
import os
import sys
import latplan
import latplan.model
from latplan.util import *
from latplan.util.planner import *
from latplan.util.plot import *
import latplan.util.stacktrace
import os.path
import keras.backend as K
import tensorflow as tf
import math
import time
import json





parser = argparse.ArgumentParser(description="A script cluster low level actions of R-latplan")

parser.add_argument('--base_dir', default=None, type=str, help='Optional: Base path of the current experiment', required=False)
parser.add_argument('--data_folder', default=None, type=str, help='Optional: Base path of the current experiment data', required=False)
parser.add_argument('--exp_folder', default=None, type=str, help='Optional: Base path of the current experiment', required=False)


args = parser.parse_args()

base_dir = args.base_dir
data_folder = args.data_folder
exp_folder = args.exp_folder


def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def autoencode_image(image):

    state = sae.encode(np.array([image]))[0].round().astype(int)

    image_rec = sae.decode(np.array([state]))[0]
    
    return state, image_rec



def deenhance(enhanced_image):
    temp_image = enhanced_image - 0.5
    temp_image = temp_image / 3
    original_image = temp_image + 0.5
    return original_image

def denormalize(normalized_image, original_min, original_max):
    if original_max == original_min:
        return normalized_image + original_min
    else:
        return (normalized_image * (original_max - original_min)) + original_min

def unnormalize_colors(normalized_images, mean, std): 
    return (normalized_images*std)+mean


def save_image(image, sae, name_, is_soko=False):
    init_unorm_color = unnormalize_colors(image, sae.parameters["mean"], sae.parameters["std"])
    if not is_soko:
        init_dee = deenhance(init_unorm_color)
        init_unorm_color = denormalize(init_dee, sae.parameters["orig_min"], sae.parameters["orig_max"])
    init_denorm = np.clip(init_unorm_color, 0, 1)
    plt.imsave(name_+".png", init_denorm)
    plt.close()
    return



path_to_dataset = data_folder + "/data.p"
loaded_data = load_dataset(path_to_dataset)
# train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, all_high_lvl_actions_one_hot, mean_all, std_all, 
# all_actions_unique, all_high_lvl_actions_unique, orig_max, orig_min, train_set_no_dupp_processed, train_set_no_dupp_orig, all_traces_pair_and_action
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

print(path_to_dataset)

sae = latplan.model.load(exp_folder,allow_failure=True)


rows = []

total = len(train_set_no_dupp)

dico_lowlvl_highlvl = {} 
for ii, ele in enumerate(train_set_no_dupp):

    print("ii: {}/{}".format(ii, total))

    # print(len(ele))
    # print(type(ele[0])) # images, processed
    # print(type(ele[0])) # all_action_one_hot
    # ele[1]  all_actions_one_hot
    # ele[2] all_high_lvl_actions_one_hot

    init_processed, goal_processed = ele[0]

    state0, init_rec = autoencode_image(init_processed)
    state1, goal_rec = autoencode_image(goal_processed)

    # save_image(init_rec, sae, "INITT", is_soko=False)
    # save_image(goal_rec, sae, "GOAAL", is_soko=False)

    # print("state0.shape")
    # print(state0)
    # print(state1)
    # print(np.argmax(ele[2]))

    #result = np.concatenate((state0, state1, np.array([np.argmax(ele[2])])))

    result = np.concatenate((state0, state1, np.array([8])))

    if not any(np.array_equal(result, existing) for existing in rows):
        rows.append(result)




result = np.array(rows)
print(result.shape) # (1469, 33)


filename = "all_transis_and_action.p"
with open(exp_folder+"/"+filename, mode="wb") as f:
    pickle.dump(result, f)

#     init, goal = init_goal_misc(p, 1, noise=None, image_path=problem_dir, is_soko=True)


#     if np.argmax(ele[1]) not in dico_lowlvl_highlvl:
#         dico_lowlvl_highlvl[np.argmax(ele[1])] = np.argmax(ele[2])

1