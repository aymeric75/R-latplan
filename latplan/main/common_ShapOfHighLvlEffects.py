import os
import os.path
import glob
import itertools
import numpy as np

import latplan.util.stacktrace
from latplan.util.tuning import simple_genetic_search, parameters, nn_task, reproduce, load_history
from latplan.util        import curry
import sys
import json
import random
from math import factorial as fact

from multiprocessing import Pool
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split



################################################################
# globals

args     = None
sae_path = None

################################################################
# command line parsing

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# 
# RawDescriptionHelpFormatter
# ArgumentDefaultsHelpFormatter


parser.add_argument(
    "mode",
    help=(
        "A string which contains mode substrings."
        "\nRecognized modes are:"
        "\n" 
        "\n   learn     : perform the training with a hyperparameter tuner. Results are stored in samples/[experiment]/logs/[hyperparameter]."
        "\n               If 'learn' is not specified, it attempts to load the stored weights."
        "\n   plot      : produce visualizations"
        "\n   dump      : dump the csv files necessary for producing the PDDL models"
        "\n   summary   : perform extensive performance evaluations and collect the statistics, store the result in performance.json"
        "\n   debug     : debug training limited to epoch=2, batch_size=100. dataset is truncated to 200 samples"
        "\n   reproduce : train the best hyperparameter so far three times with different random seeds. store the best results."
        "\n   iterate   : iterate plot/dump/summary commands above over all hyperparmeters that are already trained and stored in logs/ directory."
        "\n"
        "\nFor example, learn_plot_dump contains 'learn', 'plot', 'dump' mode."
        "\nThe separater does not matter because its presense is tested by python's `in` directive, i.e., `if 'learn' in mode:` ."
        "\nTherefore, learnplotdump also works."))


subparsers = parser.add_subparsers(
    title="subcommand",
    metavar="subcommand",
    required=True,
    description=(
        "\nA string which matches the name of one of the dataset functions in latplan.main module."
        "\n"
        "\nEach task has a different set of parameters, e.g.,"
        "\n'puzzle' has 'type', 'width', 'height' where 'type' should be one of 'mnist', 'spider', 'mandrill', 'lenna',"
        "\nwhile 'lightsout' has 'type' being either 'digital' and 'twisted', and 'size' being an integer."
        "\nSee subcommand help."))

def add_common_arguments(subparser,task,objs=False):
    subparser.set_defaults(task=task)
    subparser.add_argument(
        "num_examples",
        default=5000,
        type=int,
        help=(
            "\nNumber of data points to use. 90%% of this number is used for training, and 5%% each for validation and testing."
            "\nIt is assumed that the user has already generated a dataset archive in latplan/puzzles/,"
            "\nwhich contains a larger number of data points using the setup-dataset script provided in the root of the repository."))
    subparser.add_argument(
        "aeclass",
        help=
        "A string which matches the name of the model class available in latplan.model module.\n"+
        "It must be one of:\n"+
        "\n".join([ " "*4+name for name, cls in vars(latplan.model).items()
                    if type(cls) is type and \
                    issubclass(cls, latplan.network.Network) and \
                    cls is not latplan.network.Network
                ])
    )
    if objs:
        subparser.add_argument("location_representation",
                               nargs='?',
                               choices=["bbox","coord","binary","sinusoidal","anchor"],
                               default="coord",
                               help="A string which specifies how to convert/encode the location in the dataset. See documentations for normalize_transitions_objects")
        subparser.add_argument("randomize_location",
                               nargs='?',
                               type=bool,
                               default=False,
                               help="A boolean which specifies whether we randomly translate the environment globally. See documentations for normalize_transitions_objects")
    subparser.add_argument("comment",
                           nargs='?',
                           default="",
                           help="A string which is appended to the directory name to label each experiment.")


    subparser.add_argument("--dataset_folder",
                           nargs='?',
                           default="",
                           help="folder path of where the images are")

    subparser.add_argument("--type",
                           nargs='?',
                           default="",
                           help="if vanilla or r_latplan")
    
    return




def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data


def main(parameters={}):
    import latplan.util.tuning
    latplan.util.tuning.parameters.update(parameters)

    import sys
    global args, sae_path
    args = parser.parse_args()
    task = args.task

    delattr(args,"task")
    latplan.util.tuning.parameters.update(vars(args))

    print("'sys.argvsys.argvsys.argv")
    print(sys.argv[2])

    print(sys.argv)
    
    if sys.argv[-1] == 'vanilla':
        import latplan.modelVanilla
    else:
        import latplan.model


    if(sys.argv[2]=="puzzle"):
        sae_path = "_".join(sys.argv[2:9])
    if(sys.argv[2]=="blocks"):
        sae_path = "_".join(sys.argv[2:7])
    if(sys.argv[2]=="lightsout"):
        sae_path = "_".join(sys.argv[2:8])
    if(sys.argv[2]=="sokoban"):
        sae_path = "_".join(sys.argv[2:7])
    if(sys.argv[2]=="hanoi"):
        sae_path = "_".join(sys.argv[2:7])


    
    print(vars(args))
    # latplan.util.tuning.parameters.update(vars(args))
    # sae_path = "_".join(sys.argv[2:])

    print("SAE PATH")
    print(sae_path)


    try:
        task(args)
    except:
        latplan.util.stacktrace.format()


################################################################
# procedures for each mode

def plot_autoencoding_image(ae,transitions,label):
    if 'plot' not in args.mode:
        return

    if hasattr(ae, "plot_transitions"):
        transitions = transitions[:6]
        ae.plot_transitions(transitions, ae.local(f"transitions_{label}"),verbose=True)
    else:
        transitions = transitions[:3]
        states = transitions.reshape((-1,*transitions.shape[2:]))
        ae.plot(states, ae.local(f"states_{label}"),verbose=True)

    return


def dump_all_actions(ae,configs,trans_fn,name = "all_actions.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    l     = len(configs)
    batch = 5000
    loop  = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                pre, suc    = transitions[0], transitions[1]
                pre_b       = ae.encode(pre,batch_size=1000).round().astype(int)
                suc_b       = ae.encode(suc,batch_size=1000).round().astype(int)
                actions     = np.concatenate((pre_b,suc_b), axis=1)
                np.savetxt(f,actions,"%d")


def dump_actions(ae,transitions,name = "actions.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    print(ae.local(name))
    ae.dump_actions(transitions,batch_size = 1000)


def dump_all_states(ae,configs,states_fn,name = "all_states.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    l     = len(configs)
    batch = 5000
    loop  = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                states   = states_fn(configs[begin:end])
                states_b = ae.encode(states,batch_size=1000).round().astype(int)
                np.savetxt(f,states_b,"%d")


def dump_states(ae,states,name = "states.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            np.savetxt(f,ae.encode(states,batch_size = 1000).round().astype(int),"%d")


def dump_code_unused():
    # This code is not used. Left here for copy-pasting in the future.
    if False:
        dump_states      (ae,all_states,"all_states.csv")
        dump_all_actions (ae,all_transitions_idx,
                          lambda idx: all_states[idx.flatten()].reshape((len(idx),2,num_objs,-1)).transpose((1,0,2,3)))


def train_val_test_split(x):
    train = x[:int(len(x)*0.9)]
    val   = x[int(len(x)*0.9):int(len(x)*0.95)]
    test  = x[int(len(x)*0.95):]
    return train, val, test




def coalitions_profits(distances, max_distance):
    return 100 - ( distances / max_distance ) * 100



def compute_coalitions_weights(coals, pp):
    weights = []
    fact_p = fact(pp)
    weights = [ (len(c)*fact(pp - len(c) - 1)) / fact_p for c in coals]
    return weights


def compute_distances(images1, images2):
    """
    Computes the Euclidean distance between each pair of images from two numpy arrays.

    Parameters:
    images1 (numpy array): The first array of images of shape (N, 45, 45, 3)
    images2 (numpy array): The second array of images of shape (N, 45, 45, 3)

    Returns:
    numpy array: An array of Euclidean distances for each pair of images.
    """
    # Ensure both inputs are numpy arrays
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)

    # Check if the shapes match and have the correct format
    if images1.shape != images2.shape:
        raise ValueError("The two image arrays must have the same shape.")
    
    # if len(images1.shape) != 4 or images1.shape[1:] != (45, 45, 3):
    #     raise ValueError("Each image array must have shape (N, 45, 45, 3).")

    # Flatten the images along the last three dimensions to convert each image to a vector
    images1_flat = images1.reshape(images1.shape[0], -1)
    images2_flat = images2.reshape(images2.shape[0], -1)

    # Compute the squared differences and then sum across the vectors
    squared_diff = np.sum((images1_flat - images2_flat) ** 2, axis=1)

    # Take the square root to get the Euclidean distances
    distances = np.sqrt(squared_diff)

    return distances








# The function now accepts additional arguments needed for processing
def process_effect(coun, eff, effects_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key):
    all_effs_without_one = effects_list.copy()
    all_effs_without_one.remove(eff)
    all_coallitions_without_one = []
    all_coallitions_with_eff = []

    for r in range(0, len(all_effs_without_one) + 1):
        combinaisons = list(itertools.combinations(all_effs_without_one, r))
        combinaisons = [set(combb) for combb in combinaisons]
        all_coallitions_without_one.extend(combinaisons)
        for comb in combinaisons:
            list_ = list(comb)
            list_.append(eff)
            all_coallitions_with_eff.append(set(list_))

    coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
    coals_weights = np.array(coals_weights)

    mask_for_the_effect_without_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)
    mask_for_the_effect_with_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)

    for jjj, coal in enumerate(all_coallitions_):
        if set(coal) not in all_coallitions_without_one:
            mask_for_the_effect_without_it_trans_by_coals[0][jjj] = False
        if set(coal) not in all_coallitions_with_eff:
            mask_for_the_effect_with_it_trans_by_coals[0][jjj] = False
    
    reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(1 * nber_of_coals)
    distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]

    reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(1 * nber_of_coals)
    distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

    max_distance_the_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_without_one), axis=0)
    max_distance_the_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_with_eff), axis=0)

    profits_of_the_without_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_without_one)
    profits_of_the_with_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_with_one)

    substraction = profits_of_the_with_eff_coalition_for_the_transition - profits_of_the_without_eff_coalition_for_the_transition
    the_coalition_the_transition = coals_weights * substraction

    the_coalition_the_transition_sum_over_coals = np.sum(the_coalition_the_transition, axis=0)
    
    mean_shap_value_over_the_transition = np.mean(the_coalition_the_transition_sum_over_coals)
    shap_value = mean_shap_value_over_the_transition

    return coun, eff, shap_value




def run(path,transitions,extra=None):








    import hashlib
    def hash_two_images(image1: np.ndarray, image2: np.ndarray, hash_function: str = 'sha256') -> str:
        """
        Hashes two NumPy colored images using the specified hash function and returns a unique hash.

        Parameters:
        image1 (np.ndarray): A NumPy array representing the first colored image (H x W x 3).
        image2 (np.ndarray): A NumPy array representing the second colored image (H x W x 3).
        hash_function (str): The hash function to use ('md5', 'sha1', 'sha256', etc.).

        Returns:
        str: The resulting unique hash string for both images combined.
        """
        # Ensure both inputs are NumPy arrays
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise TypeError("Both images must be NumPy arrays")

        # Optionally, you can ensure both images have the same shape. If not, you could resize or pad them.
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same shape for hashing")

        # Concatenate the two images along a new axis to combine them
        combined_images = np.concatenate((image1, image2), axis=-1)

        # Convert the combined image to a byte array
        combined_bytes = combined_images.tobytes()

        # Create the hash object
        hash_obj = hashlib.new(hash_function)

        # Update the hash object with the combined image bytes
        hash_obj.update(combined_bytes)

        # Return the hexadecimal digest
        return hash_obj.hexdigest()


    def unnormalize_colors(normalized_images, mean, std): 
        # Reverse the normalization process
        # unnormalized_images = normalized_images * (std + 1e-6) + mean
        # return np.round(unnormalized_images).astype(np.uint8)
        return (normalized_images*std)+mean


    def deenhance(enhanced_image):
        # Reverse the final shift by subtracting 0.5
        temp_image = enhanced_image - 0.5
        
        # Reverse the clipping: Since clipping limits values, we cannot fully recover original values if they were outside the [-0.5, 0.5] range. 
        # However, for values within the range, we can reverse the scale by dividing by 3.
        # We assume that the enhanced image has values within the range that the clip function allowed.
        temp_image = temp_image / 3
        
        # Reverse the initial shift by adding 0.5 back
        original_image = temp_image + 0.5
        
        return original_image

    def denormalize(normalized_image, original_min, original_max):
        if original_max == original_min:
            return normalized_image + original_min
        else:
            return (normalized_image * (original_max - original_min)) + original_min

    def denorm(theim, mean_to_use_, std_to_use_, orig_min_, orig_max_):

        # last_current_time = time.time()
        # time_spent = time.time() - last_current_time
        # print("THE TIME spent start is ", str(time_spent))
        unorm = unnormalize_colors(theim, mean_to_use_, std_to_use_)

        # time_spent =  time.time() - last_current_time
        # print("THE TIME spent 00 is ", str(time_spent))
        # last_current_time = time.time()
        dehanced = deenhance(unorm)

        # time_spent =  time.time() - last_current_time
        # print("THE TIME spent 11 is ", str(time_spent))
        # last_current_time = time.time()

        denormalized = denormalize(dehanced, orig_min_, orig_max_)
        # time_spent =  time.time() - last_current_time
        # print("THE TIME spent 22 is ", str(time_spent))
        # last_current_time = time.time()

        ret = np.clip(denormalized, 0, 1)
        # time_spent =  time.time() - last_current_time
        # print("THE TIME spent 33 is ", str(time_spent))
        # last_current_time = time.time()

        return ret

    ### 

    ###


    #### former Vanilla Latplan code
    train, val, test = train_val_test_split(transitions)

    # learning: un base_aux_json
    #                   puis le exp_aux_json
    #
    #
    #       testing depuis le exp_aux_json

    # 
    dataset_fold = None

    if args.type == "vanilla":
        dataset_fold = "r_vanilla_latplan_datasets"
    else:
        dataset_fold = "r_latplan_datasets"

    # DATASETS FOLDERS
    dataset_aux_json_folder_base = dataset_fold+"/"+sys.argv[2]
    dataset_aux_json_folder_exp = dataset_fold+"/"+sys.argv[2] + "/" + args.dataset_folder

    # EXPERIMENTS FOLDERS
    exp_aux_json_folder = None
    if args.type == "vanilla":
        exp_aux_json_folder = "r_vanilla_latplan_exps/" +sys.argv[2] + "/" + args.dataset_folder
    else:
        exp_aux_json_folder = "r_latplan_exps/" +sys.argv[2] + "/" + args.dataset_folder
    
    # 
    path_to_dataset = dataset_aux_json_folder_exp +"/data.p"


    # load dataset for the specific experiment
    loaded_data = load_dataset(path_to_dataset)
    
    train_set = loaded_data["train_set"] 
    test_val_set = loaded_data["test_val_set"] 
    all_pairs_of_images_reduced_orig = loaded_data["all_pairs_of_images_reduced_orig"] 
    all_actions_one_hot = loaded_data["all_actions_one_hot"]
    all_high_lvl_actions_one_hot = loaded_data["all_high_lvl_actions_one_hot"]
    mean_all = loaded_data["mean_all"] 
    std_all = loaded_data["std_all"] 
    all_actions_unique = loaded_data["all_actions_unique"] 
    orig_max = loaded_data["orig_max"] 
    orig_min = loaded_data["orig_min"] 
    #

    train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]
    train_set_no_dupp_orig = loaded_data["train_set_no_dupp_orig"]


    if args.type == "vanilla":
        train_set_ = []
        for tr in train_set:
            train_set_.append(tr[0])
        train_set = np.array(train_set_)
        test_val_set_ = []
        for tr in test_val_set:
            test_val_set_.append(tr[0])
        test_val_set = np.array(test_val_set_)

    # elif args.type == "r_latplan":
    #     train_set = np.array(train_set)
    #     test_val_set = np.array(test_val_set)

    #     train_set_ = []
    #     for tr in train_set:
    #         print("TR IS ")
    #         print(tr)
    #         exit()
    #         #train_set_.append(tr[0])
    #     # train_set = np.array(train_set_)
    #     # test_val_set_ = []
    #     # for tr in test_val_set:
    #     #     test_val_set_.append(tr[0])
    #     # test_val_set = np.array(test_val_set_)



    if 'compute_effects_shap_values_per_action' in args.mode:


        print(os.path.join(exp_aux_json_folder,"aux.json"))
        if os.path.isfile(os.path.join(exp_aux_json_folder,"aux.json")):
            with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
                data = json.load(f)
        

        # PREAMBULE
        data['parameters']["loss_effs"] = "no"
        data['parameters']["loss_precs_stat"] = "no"
        data['parameters']["loss_precs_dyn"] = "no"
        data['parameters']["newloss_starting_epoch__AND__newloss_ending_epoch"] = [1, 5]
        data['parameters']["loss_effs_k_dist"] = 1.8 #"1. #0.08
        data['parameters']["loss_effs_k_var"] = 8
        data['parameters']["loss_precs_stat_k_dist"] = 1.8
        data['parameters']["loss_precs_stat_k_var"] = 8
        data['parameters']["loss_precs_dyn_k_dist"] = 0.04
        data['parameters']["loss_precs_dyn_k_var"] = 2.5
        data['parameters']["ama_w"] = 1
        data['parameters']["sae_w"] = 1

        
        # LOAD THE MODEL
        net = latplan.model.load(exp_aux_json_folder, allow_failure=True)


        # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION

        dico_transitions_per_high_lvl_actions = {}

        # will be like
        #   {  
        #       0: {
        #               'hash1' : [ image1, image2 ],
        #               'hash2' : [image1, image2],
        #               ....
        #           }
        #       1: {
        #               ......
        #           }
        #}
        


        # loop over the train set (without dupplicate)
        # AND group the transitions into their respective High Level Actions
        for ii, ele in enumerate(train_set_no_dupp):


            # if the looped transition high level action is not a key of dico_transitions_per_high_lvl_actions
            if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:

                # add the key (the high lvl action index)
                dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}

            # create a unique hash for the current transition
            #current_images_hash = hash_two_images(ele[0][0], ele[0][1])
            

            # if the hash is not in the keys of the dico of the high lvl action
            # i.e. if the transition was not added as a transition for this high lvl action
            if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
                
                # add the transition i.e.:
                # the two preprocessed images
                # the two <=> reduced images (not preprocessed)
                # the onehot repr of the high lvl action
                # the max_diff_in_bits (not used here)
                dico_transitions_per_high_lvl_actions[np.argmax(ele[2])][np.argmax(ele[1])] = {
                    "preprocessed" : ele[0],
                    "reduced" : train_set_no_dupp_orig[ii][0],
                    "onehot": ele[2],
                    #"max_diff_in_bits_": ele[2]
                }

        #save_with_pickle(os.getcwd(), dico_transitions_per_high_lvl_actions)



        # add_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"action_add4.csv"), delimiter=' ', dtype=int)
        # del_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"action_del4.csv"), delimiter=' ', dtype=int)

        add_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"add_effs_aligned.csv"), delimiter=' ', dtype=int)
        del_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"del_effs_aligned.csv"), delimiter=' ', dtype=int)


        dir_shap_vals_effects_no_touching = os.path.join(exp_aux_json_folder, 'shap_vals_effects_no_touching')
        if not os.path.isdir(dir_shap_vals_effects_no_touching):
            os.mkdir(dir_shap_vals_effects_no_touching)

        
        dir_shap_vals_persisting_effects_removed = os.path.join(exp_aux_json_folder, 'shap_vals_persisting_effects_removed')
        if not os.path.isdir(dir_shap_vals_persisting_effects_removed):
            os.mkdir(dir_shap_vals_persisting_effects_removed)


        last_current_time = time.time()


        # including the empty set !!!!

        lecounter = 0
        print("d2222")
        # for each high level action
        for high_lvl_ac_index, transitions in dico_transitions_per_high_lvl_actions.items():


            # if high_lvl_ac_index in [5, 11]:
            #     continue

            # 
            action_main_effects = []
            with open(dir_shap_vals_persisting_effects_removed+"/action_"+str(high_lvl_ac_index)+"_main_effects_withEmptySet_corrected.txt", "r") as file:
                for line in file:
                    action_main_effects.append(line.strip())
            

            print("action_main_effectsaction_main_effectsaction_main_effects")
            print(len(action_main_effects))
            

            #######

            indices_of_transitions = np.array(list(transitions.keys()))
            add_effects_for_the_action = add_effects[indices_of_transitions]
            del_effects_for_the_action = del_effects[indices_of_transitions]

            print("there are {} transitions".format(str(len(transitions))))

            # NO USED HERE

            # add_effects_for_the_action_ = []
            # for trans_effects in add_effects_for_the_action:
            #     add_effects_for_the_action__ = np.where(trans_effects == 1)[0]

            #     for eff in add_effects_for_the_action__:
            #         if (eff not in add_effects_for_the_action_):
            #             add_effects_for_the_action_.append(eff)
            # add_effects_for_the_action = add_effects_for_the_action_

            # del_effects_for_the_action_ = []
            # for trans_effects in del_effects_for_the_action:
            #     del_effects_for_the_action__ = np.where(trans_effects == 1)[0]
            #     for eff in del_effects_for_the_action__:
            #         if (eff not in del_effects_for_the_action_):
            #             del_effects_for_the_action_.append(eff)
            # del_effects_for_the_action = del_effects_for_the_action_

            # # make a list of effects "add_"+i, "del_"+i
            # effects_list = []
            # for add_eff in add_effects_for_the_action:
            #     effects_list.append("add_"+str(add_eff))
            # for del_eff in del_effects_for_the_action:
            #     effects_list.append("del_"+str(del_eff))

    
            effects_list = action_main_effects
   
   

            # LIST OF ALL POSSIBLE COALLITIONS OF THE EFFECTS
            all_coallitions_ = []
            for r in range(0, len(effects_list) + 1):
                combinaisons = list(itertools.combinations(effects_list, r))
                all_coallitions_.extend(combinaisons)
                    
            nber_of_coals = len(all_coallitions_)

            print("NBER OF COALS !!!") # 1 048 575

            time_spent =  time.time() - last_current_time
            print("time spent 11 is ", str(time_spent))
            last_current_time = time.time()

            all_preproc_im1 = []
            all_preproc_im2 = []
            all_reduced_im1 = []
            all_reduced_im2 = []


            for val in transitions.values():

                preproc_im1, preproc_im2 = val["preprocessed"]
                all_preproc_im1.append(preproc_im1)
                all_preproc_im2.append(preproc_im2)

                reduced_im1, reduced_im2 = val["reduced"]
                all_reduced_im1.append(reduced_im1)
                all_reduced_im2.append(reduced_im2)

            all_preproc_im1 = np.array(all_preproc_im1)
            all_preproc_im2 = np.array(all_preproc_im2)

            all_reduced_im1 = np.array(all_reduced_im1)
            all_reduced_im2 = np.array(all_reduced_im2)



            time_spent =  time.time() - last_current_time
            print("time spent 22 is ", str(time_spent))
            last_current_time = time.time()


            # PRE LATENT STATES (encoded)
            zs_pre_s = net.encode(all_preproc_im1)
            zs_sucs_s = np.squeeze(zs_pre_s.copy())

            # SUCC LATENT STATES (predicted)
            zs_sucs_s[:, add_effects_for_the_action] = 1
            zs_sucs_s[:, del_effects_for_the_action] = 0
     

            nber_of_coals = len(all_coallitions_)

            # mask of add / del (also (nber_coals x 50))
            add_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)
            del_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)

            time_spent =  time.time() - last_current_time
            print("time spent 33 is ", str(time_spent))
            last_current_time = time.time()


            # MASKS of the ADD and the DEL EFFECTS for each coalitions
            for iii, coal in enumerate(all_coallitions_):
                adds = [int(command.split('_')[1]) for command in coal if command.startswith("add")]
                add_masks_all_coals[iii][adds] = 1
                dels = [int(command.split('_')[1]) for command in coal if command.startswith("del")]
                del_masks_all_coals[iii][dels] = 1


            time_spent =  time.time() - last_current_time
            print("time spent 44 is ", str(time_spent))
            last_current_time = time.time()

            # super TENSOR OF: coals / transitions / latent
            # i.e. each item is a coalition of a transition 
            # contains, for each coalition, for each transition, the z_pre state applied to the coalition
            coals_transis_latent = np.zeros((nber_of_coals, len(transitions), 50), dtype=int)
            for iiii, (add_m, del_m) in enumerate(zip(add_masks_all_coals, del_masks_all_coals)):
                zs_pre_s_ = zs_pre_s.copy()
                zs_pre_s_[:,np.where(add_m == 1)[0]] = 1
                zs_pre_s_[:,np.where(del_m == 1)[0]] = 0
  
                coals_transis_latent[iiii] = zs_pre_s_
                del zs_pre_s_


            time_spent =  time.time() - last_current_time
            print("time spent 55 is ", str(time_spent))
            last_current_time = time.time()

            transis_coals_latent = np.transpose(coals_transis_latent, (1, 0, 2))
            shape = transis_coals_latent.shape


            # coals_transis_latent becomes of shape (#transis x #coals, #latent)
            transis_coals_latent_two_dims = transis_coals_latent.reshape(shape[0] * shape[1], shape[2])


            del transis_coals_latent
            # decoding and denorming of all the coals applied to the z_pre of each transition
            transis_coals_latent_two_dims_decoded = net.decode(transis_coals_latent_two_dims)
            transis_coals_latent_two_dims_denormed = denorm(np.squeeze(transis_coals_latent_two_dims_decoded), mean_all, std_all, orig_min, orig_max)

            time_spent =  time.time() - last_current_time
            print("time spent 66 is ", str(time_spent))
            last_current_time = time.time()
            # 
            all_reduced_im2_transi_by_coals = np.repeat(all_reduced_im2, repeats=nber_of_coals, axis=0)

            all_preproc_im2_denormed = denorm(np.squeeze(all_preproc_im2), mean_all, std_all, orig_min, orig_max)
            all_preproc_im2_transi_by_coals_denormed = np.repeat(all_preproc_im2_denormed, repeats=nber_of_coals, axis=0)

            time_spent =  time.time() - last_current_time
            print("time spent 77 is ", str(time_spent))
            last_current_time = time.time()


            dists = compute_distances(transis_coals_latent_two_dims_denormed, all_preproc_im2_transi_by_coals_denormed)
            reshaped_dists = dists.reshape(len(transitions), nber_of_coals)
            max_distance_for_each_transition  = np.max(reshaped_dists, axis=1)
            p = len(effects_list)

            time_spent =  time.time() - last_current_time
            print("time spent 88 is ", str(time_spent))
            last_current_time = time.time()



            for coun, eff in enumerate(effects_list):
                    
                time_spent =  time.time() - last_current_time
                print(" (starting an effect) time spent 06 is ", str(time_spent))
                last_current_time = time.time()

                # if coun > 2:
                #     break

                # need some lists with and without the effects
                all_effs_without_one = effects_list.copy()
                all_effs_without_one.remove(eff)
                all_coallitions_without_one = []
                all_coallitions_with_eff = []

                for r in range(0, len(all_effs_without_one) + 1):
                    combinaisons = list(itertools.combinations(all_effs_without_one, r))
                    combinaisons = [set(combb) for combb in combinaisons]
                    all_coallitions_without_one.extend(combinaisons)
                    for comb in combinaisons:
                        list_ = list(comb)
                        list_.append(eff)
                        all_coallitions_with_eff.append(set(list_))
                        #all_coallitions_with_eff.append(tuple(list_))

                del all_effs_without_one

                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 07 is ", str(time_spent))
                last_current_time = time.time()

                # doit être le array des weights de ttes les coals SANS le player p

                coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
                coals_weights = np.array(coals_weights)


                # # tells for each transition, and for any coal, of the coal 
                # # belongs to the group of all_coallitions_without_one (or of all_coallitions_with_eff)
                # mask_for_the_effect_without_it_trans_by_coals = np.full((len(transitions), len(all_coallitions_)), True)
                # mask_for_the_effect_with_it_trans_by_coals = np.full((len(transitions), len(all_coallitions_)), True)
                # for iii in range(len(transitions)):
                #     thecount = 0
                #     for jjj, coal in enumerate(all_coallitions_):
                #         if set(coal) not in all_coallitions_without_one:
                #             mask_for_the_effect_without_it_trans_by_coals[iii][jjj] = False
                #         else:
                #             thecount += 1
                #         if set(coal) not in all_coallitions_with_eff:
                #             mask_for_the_effect_with_it_trans_by_coals[iii][jjj] = False


                # Convert list of sets to an array of objects to use numpy masking
                coal_array = np.array([set(coal) for coal in all_coallitions_], dtype=object)
                all_coals_without_array = np.array([set(coal) for coal in all_coallitions_without_one], dtype=object)
                all_coals_with_eff_array = np.array([set(coal) for coal in all_coallitions_with_eff], dtype=object)

                # Create masks by checking membership in all_coallitions_without_one and all_coallitions_with_eff
                mask_for_the_effect_without_it = np.isin(coal_array, all_coals_without_array).reshape(1, -1)
                mask_for_the_effect_with_it = np.isin(coal_array, all_coals_with_eff_array).reshape(1, -1)

                # Broadcast masks over transitions
                mask_for_the_effect_without_it_trans_by_coals = np.tile(mask_for_the_effect_without_it, (len(transitions), 1))
                mask_for_the_effect_with_it_trans_by_coals = np.tile(mask_for_the_effect_with_it, (len(transitions), 1))


                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 08 is ", str(time_spent))
                last_current_time = time.time()
                #print(" all_coallitions_ {}".format(len(all_coallitions_)))

                
                reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(len(transitions)*nber_of_coals)
                distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]
                
                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 08 BIS 0 is ", str(time_spent))
                last_current_time = time.time()

                reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(len(transitions)*nber_of_coals)
                distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 08 BIS 1 is ", str(time_spent))
                last_current_time = time.time()

                # faut max_distance_each_transition_repeated (#transitions, #coals)
                max_distance_each_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_each_transition, repeats=len(all_coallitions_without_one), axis=0)
                # (#transitions x #coals_without_one)
                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 08 BIS 2 is ", str(time_spent))
                last_current_time = time.time()

                max_distance_each_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_each_transition, repeats=len(all_coallitions_with_eff), axis=0)

                time_spent =  time.time() - last_current_time
                print(" (in an effect) time spent 09 is ", str(time_spent))
                last_current_time = time.time()

                profits_of_each_without_eff_coalition_for_each_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_each_transition_repeated_for_all_coallitions_without_one)
                profits_of_each_with_eff_coalition_for_each_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_each_transition_repeated_for_all_coallitions_with_one)

                profits_of_each_without_eff_coalition_for_each_transition = profits_of_each_without_eff_coalition_for_each_transition.reshape(len(transitions), len(all_coallitions_without_one))
                profits_of_each_with_eff_coalition_for_each_transition = profits_of_each_with_eff_coalition_for_each_transition.reshape(len(transitions), len(all_coallitions_with_eff))



                substraction = profits_of_each_with_eff_coalition_for_each_transition - profits_of_each_without_eff_coalition_for_each_transition
                each_coalition_each_transition = coals_weights * substraction

    
                # print("np.sum(each_coalition_each_transition)")
                each_coalition_each_transition_sum_over_coals = np.sum(each_coalition_each_transition, axis=1)
                
                mean_shap_value_over_transitions = np.mean(each_coalition_each_transition_sum_over_coals)
                
                # écrire chaque ele comme une ligne
                #with open(dir_shap_vals_effects_no_touching+"/action_"+str(high_lvl_ac_index)+".txt", "a+") as file:
                with open(dir_shap_vals_persisting_effects_removed+"/action_"+str(high_lvl_ac_index)+"_ShapOfHighLvlEffect_withEmptySet_corrected.txt", "a+") as file:
                    file.write("for effect "+str(eff)+"\n")
                    #with open('array_elements.txt', 'w') as file:
                    for element in each_coalition_each_transition_sum_over_coals:
                        file.write(f"{element}\n")
                    file.write("\n")

                shap_value = mean_shap_value_over_transitions

                 
                time_spent =  time.time() - last_current_time
                print(" (ending effect) time spent 10 is ", str(time_spent))
                last_current_time = time.time()



                print("shap_value for action {} and eff: {} is {}".format(str(high_lvl_ac_index), eff, str(shap_value)))
  
        exit()

    # load the json file from the base domain folder (in order to update and copy/save it in the exp subfolder)
    
    if 'learn' in args.mode:
        print("LI1")
        print(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json"))

        if os.path.isfile(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json")):
            with open(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json"),"r") as f:
                data = json.load(f)

    # elif 'dump' in args.mode:
    #     print("LI2")

    #     print(os.path.join(exp_aux_json_folder,"aux.json"))
    #     if os.path.isfile(os.path.join(exp_aux_json_folder,"aux.json")):
    #         with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
    #             data = json.load(f)


    # # Step 2: Replace 'mean' and 'std' in the dictionary
    # data['parameters']['mean'] = mean_all.tolist()
    # data['parameters']['std'] = std_all.tolist()

    # data['parameters']['orig_max'] = orig_max
    # data['parameters']['orig_min'] = orig_min
    # data["parameters"]["time_start"] = ""

    # if args.type == "vanilla" and 'dump' in args.mode:
    #     data["parameters"]["beta_z_and_beta_d"] = [1, 1000]
    #     data["parameters"]["pdiff_z1z2_z0z3"] = [1, 1000]



    # if args.type == "vanilla":
    #     data['parameters']['A'] = 6000
    # else:
    #     data['parameters']['A'] = len(all_actions_unique)

    # if args.type == "vanilla":

    #     aaa = [2]
    #     aaa.extend(train_set[0][0].shape)
    #     data["input_shape"] = aaa

    # else:

    #     aaa = [2]
    #     aaa.extend(train_set[0][0][0].shape)
    #     data["input_shape"] = aaa


    # print("ON EST LAAAAAAAAAA")
    # print("dataset_aux_json_folder_exp {}".format(dataset_aux_json_folder_exp))
    # print()
    # print("exp_aux_json_folder {}".format(exp_aux_json_folder))
    # print()
    # # save the updated aux.json into the exp subfolder of the dataset folder
    # with open(os.path.join(dataset_aux_json_folder_exp,"aux.json"),"w") as f:
    #     json.dump(data, f, indent=4)

    # # save the updated aux.json into the exp folder (in r_latplan_exps)
    # with open(os.path.join(exp_aux_json_folder,"aux.json"),"w") as f:
    #     json.dump(data, f, indent=4)
    

    # finally, read the saved exp aux.json (see above)
    with open(os.path.join(dataset_aux_json_folder_exp,"aux.json"),"r") as f:
        parameters = json.load(f)["parameters"]


    parameters["mean"] = mean_all
    parameters["std"] = std_all

    val_set = test_val_set[:len(test_val_set)//2]

    random.shuffle(val_set)



    def postprocess(ae):
        show_summary(ae, train, test)
        plot_autoencoding_image(ae,train,"train")
        plot_autoencoding_image(ae,test,"test")
        dump_actions(ae,transitions)
        return ae


    def report(net,eval):
        try:
            postprocess(net)
            if extra:
                extra(net)
        except:
            latplan.util.stacktrace.format()
        return



    if 'dump' in args.mode:

        # print(self.local(name))

        # finally, read the saved exp aux.json (see above)
        with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
            parameters = json.load(f)["parameters"]


        print(args)
        print("HGTFRDEA!!!!!")
        # prob ici c'est que ça load 
        #args.dataset_folder is hanoi_complete_clean_faultless_withoutTI
        print("exp_aux_json_folder is {}".format(exp_aux_json_folder))
        print("dataset_aux_json_folder_exp is {}".format(dataset_aux_json_folder_exp))

        parameters["time_start"] = ""

        # beta_z_and_beta_d
        parameters["epoch"] = 1
        #parameters["A"] = 6000

        parameters["beta_ama_recons"] = 1
        parameters["beta_z_and_beta_d"] = [1, 1000]
        parameters["pdiff_z1z2_z0z3"] = [1, 1000]
        print("theparameters")
        print(parameters)
        print("parametersss")
        #exit()
        # print("exp_aux_json_folderexp_aux_json_folder")
        # print(exp_aux_json_folder)
        # exit()
        if sys.argv[-1] == 'vanilla':
            net = latplan.modelVanilla.load(exp_aux_json_folder, allow_failure=False)
        else:
            net = latplan.model.load(exp_aux_json_folder, allow_failure=False)

        # 
        print(type(transitions)) # 
        #print(transitions.shape) # (5000, 2, 48, 48, 1)
        print("AAAAASQQQQQQ")
        print(len(transitions))
        
        # print(np.array(transitions).shape)

        if args.type == "vanilla":
            #dump_actions(net, [transitions, actions_transitions], name = "actions.csv", repeat=1)
            dump_actions(net, train_set, name = "actions.csv", repeat=1)
        
        else:

            thetransarray=[]
            theactionarray=[]
            for trr in train_set:
                thetransarray.append(trr[0])
                theactionarray.append(trr[1])
            

            #dump_actions(net, [transitions, actions_transitions], name = "actions.csv", repeat=1)
            dump_actions(net, [thetransarray, theactionarray], name = "actions.csv", repeat=1)
            # alors...




    if 'aggregate_effects_preconds' in args.mode:
        print("ENFIN")
        exit()



    if 'learn' in args.mode:


        parameters["epoch"] = 10000

        parameters["load_sae_weights"] = False
        
        parameters["use_wandb"] = True


        # train_set = [train_set[0]]
        # val_set = [val_set[0]]


        parameters["the_exp_path"] = exp_aux_json_folder
        # parameters["beta_z_and_beta_d"] = [10, 1000]
        # parameters["N"] = 300
        parameters["beta_z_and_beta_d"] = [1, 100]
        parameters["N"] = 50
        # parameters["pdiff_z1z2_z0z3"] = 0
        parameters["type"] = args.type
            
        with open(os.path.join(dataset_aux_json_folder_exp,"aux.json"),"w") as f:
            json.dump(data, f, indent=4)

        with open(os.path.join(exp_aux_json_folder,"aux.json"),"w") as f:
            json.dump(data, f, indent=4)
        print("exp_aux_json_folderexp_aux_json_folder is {}".format(exp_aux_json_folder))
        # return 1
        # exit()
        if args.type == "vanilla":
            task = curry(nn_task, latplan.modelVanilla.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
            task()
        else:
            task = curry(nn_task, latplan.model.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
            task()

        # simple_genetic_search(
        #     curry(nn_task, latplan.model.get(parameters["aeclass"]),
        #           path,
        #           train, train, val, val), # noise data is used for tuning metric
        #     parameters,
        #     path,
        #     limit              = 100,
        #     initial_population = 100,
        #     population         = 100,
        #     report             = report,
        # )

    if 'resume' in args.mode:
        simple_genetic_search(
            lambda parameters: nn_task(latplan.model.get(parameters["aeclass"]), path, train, train, val, val, parameters, resume=True),
            parameters,
            path,
            limit              = 100,
            initial_population = 100,
            population         = 100,
            report             = report,
        )

    if 'debug' in args.mode:
        print("debug run. removing past logs...")
        for _path in glob.glob(os.path.join(path,"*")):
            if os.path.isfile(_path):
                os.remove(_path)
        parameters["epoch"]=1
        parameters["batch_size"]=100
        train, val = train[:200], val[:200]
        simple_genetic_search(
            curry(nn_task, latplan.model.get(parameters["aeclass"]),
                  path,
                  train, train, val, val), # noise data is used for tuning metric
            parameters,
            path,
            limit              = 1,
            initial_population = 1,
            population         = 1,
            report             = report,
        )

    if 'reproduce' in args.mode:   # reproduce the best result from the grid search log
        reproduce(
            curry(nn_task, latplan.model.get(parameters["aeclass"]),
                  path,
                  train, train, val, val), # noise data is used for tuning metric
            path,
            report      = report,
        )

    if 'iterate' in args.mode:
        open_list, _ = load_history(path)
        topk = open_list[:10]
        topk_dirnames = [
            os.path.join(path,"logs",elem[1]["hash"])
            for elem in topk
        ]
        print(f"iterating mode {args.mode} for all weights stored under logs")
        for path in topk_dirnames:
            postprocess(latplan.model.load(path))



def show_summary(ae,train,test):
    if 'summary' in args.mode:
        ae.summary()
        ae.report(train, test_data = test, train_data_to=train, test_data_to=test)


