import os
import os.path
import glob
import itertools
import numpy as np
import h5py
import latplan.util.stacktrace
from latplan.util.tuning import simple_genetic_search, parameters, nn_task, reproduce, load_history
from latplan.util import curry
import sys
import json
import random
from math import factorial as fact

import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Array, Value, Lock, Manager
import time

import pickle

from tqdm import tqdm

from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt

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

    subparser.add_argument("--action_id",
                           nargs='?',
                           default="",
                           help="which high lvl action id to train on",
                           required = False
                           )
    

    

    return





def save_dataset(dire, to_save, filename):
    data = {
        "saved": to_save
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)


def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data



def compute_combinations(args):
    """Compute combinations for a given range of r values."""
    preconds_list, r = args
    return list(itertools.combinations(preconds_list, r))

def parallel_combinations(preconds_list):
    """Compute all combinations of preconds_list in parallel with progress tracking."""
    all_coallitions_ = []

    # Prepare arguments for the pool
    args = [(preconds_list, r) for r in range(0, len(preconds_list) + 1)]

    # Initialize progress bar
    with tqdm(total=len(args), desc="Processing combinations") as pbar:
        
        # Custom callback to update progress bar
        def update_progress(*_):
            pbar.update()

        # Use multiprocessing to compute combinations in parallel
        with Pool(cpu_count()) as pool:
            results = []
            for result in pool.imap(compute_combinations, args):
                results.append(result)
                update_progress()

    # Flatten the list of results
    for result in results:
        all_coallitions_.extend(result)

    return all_coallitions_




# def compute_combinations(args):
#     """Compute combinations for a given range of r values."""
#     preconds_list, r = args
#     return list(itertools.combinations(preconds_list, r))

# def parallel_combinations(preconds_list):
#     """Compute all combinations of preconds_list in parallel."""
#     all_coallitions_ = []
    
#     # Prepare arguments for the pool
#     args = [(preconds_list, r) for r in range(0, len(preconds_list) + 1)]

#     # Use multiprocessing to compute combinations in parallel
#     with Pool(cpu_count()) as pool:
#         results = pool.map(compute_combinations, args)

#     # Flatten the list of results
#     for result in results:
#         all_coallitions_.extend(result)

#     return all_coallitions_




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


# def compute_distances(images1, images2):
#     """
#     Computes the Euclidean distance between each pair of images from two numpy arrays.

#     Parameters:
#     images1 (numpy array): The first array of images of shape (N, 45, 45, 3)
#     images2 (numpy array): The second array of images of shape (N, 45, 45, 3)

#     Returns:
#     numpy array: An array of Euclidean distances for each pair of images.
#     """
#     # Ensure both inputs are numpy arrays
#     images1 = np.asarray(images1)
#     images2 = np.asarray(images2)

#     # Check if the shapes match and have the correct format
#     if images1.shape != images2.shape:
#         raise ValueError("The two image arrays must have the same shape.")
    
#     # if len(images1.shape) != 4 or images1.shape[1:] != (45, 45, 3):
#     #     raise ValueError("Each image array must have shape (N, 45, 45, 3).")

#     # Flatten the images along the last three dimensions to convert each image to a vector
#     images1_flat = images1.reshape(images1.shape[0], -1)
#     images2_flat = images2.reshape(images2.shape[0], -1)

#     # Compute the squared differences and then sum across the vectors
#     squared_diff = np.sum((images1_flat - images2_flat) ** 2, axis=1)

#     # Take the square root to get the Euclidean distances
#     distances = np.sqrt(squared_diff)

#     print("distancesdistances")
#     print(distances.shape)
#     print(type(distances))
#     print(distances[:2])
#     print(type(distances[0]))

#     # (512,)
#     # <class 'numpy.ndarray'>
#     # [1.53527 1.53527]
#     # <class 'numpy.float64'>
    

#     exit()

#     return distances



def compute_distances_worker(args):
    """Worker function to compute distances for a chunk of images."""
    images1_chunk, images2_chunk = args
    def compute_distance_single(image1, image2):
        diff = image1 - image2
        return np.sqrt(np.sum(diff ** 2))

    return [compute_distance_single(img1, img2) for img1, img2 in zip(images1_chunk, images2_chunk)]


def compute_distances(images1, images2):
    """Main function to compute distances using multiprocessing."""
    # Ensure both inputs are numpy arrays
    images1 = np.asarray(images1)
    images2 = np.asarray(images2)

    # Check if the shapes match and have the correct format
    if images1.shape != images2.shape:
        raise ValueError("The two image arrays must have the same shape.")

    # Flatten the images along the last three dimensions to convert each image to a vector
    images1_flat = images1.reshape(images1.shape[0], -1)
    images2_flat = images2.reshape(images2.shape[0], -1)

    # Determine the number of CPUs to use
    num_cpus = min(cpu_count(), len(images1_flat))  # Ensure CPUs don't exceed the number of tasks

    # Split the images into chunks for parallel processing
    chunk_size = len(images1_flat) // num_cpus + (len(images1_flat) % num_cpus > 0)
    chunks1 = [images1_flat[i:i + chunk_size] for i in range(0, len(images1_flat), chunk_size)]
    chunks2 = [images2_flat[i:i + chunk_size] for i in range(0, len(images2_flat), chunk_size)]

    # Prepare arguments for each worker
    args = list(zip(chunks1, chunks2))

    # Initialize the progress bar
    with Pool(num_cpus) as pool:
        with tqdm(total=len(args)) as pbar:
            results = []
            for result in pool.map(compute_distances_worker, args):
                results.extend(result)  # Maintain order by appending in sequence
                pbar.update(1)  # Update progress bar for each completed chunk

    # Convert the results to a numpy array
    retour = np.array(results)
    return retour



##### BON, TU FAIS LA MËME CHOSE POUR LES PRECONDITIONS..

###### pour etre plus précis, tu veux: 
######              identifie ce qui prend le plus de temps dans le code 
####################    ensuite dans ce TRUC, identifie la loop  !!!!!  et c'est ça que tu note !!!

# # returns the SHAP value of an effect of a transition
# def process_effect(coun, eff, effects_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key):
#     all_effs_without_one = effects_list.copy()
#     all_effs_without_one.remove(eff)
#     all_coallitions_without_one = []
#     all_coallitions_with_eff = []

#     for r in range(1, len(all_effs_without_one) + 1):
#         combinaisons = list(itertools.combinations(all_effs_without_one, r))
#         combinaisons = [set(combb) for combb in combinaisons]
#         all_coallitions_without_one.extend(combinaisons)
#         for comb in combinaisons:
#             list_ = list(comb)
#             list_.append(eff)
#             all_coallitions_with_eff.append(set(list_))

#     coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
#     coals_weights = np.array(coals_weights)

#     mask_for_the_effect_without_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)
#     mask_for_the_effect_with_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)

#     for jjj, coal in enumerate(all_coallitions_):
#         if set(coal) not in all_coallitions_without_one:
#             mask_for_the_effect_without_it_trans_by_coals[0][jjj] = False
#         if set(coal) not in all_coallitions_with_eff:
#             mask_for_the_effect_with_it_trans_by_coals[0][jjj] = False
    
#     reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(1 * nber_of_coals)
#     distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]

#     reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(1 * nber_of_coals)
#     distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

#     max_distance_the_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_without_one), axis=0)
#     max_distance_the_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_with_eff), axis=0)

#     profits_of_the_without_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_without_one)
#     profits_of_the_with_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_with_one)

#     substraction = profits_of_the_with_eff_coalition_for_the_transition - profits_of_the_without_eff_coalition_for_the_transition
#     the_coalition_the_transition = coals_weights * substraction

#     the_coalition_the_transition_sum_over_coals = np.sum(the_coalition_the_transition, axis=0)
    
#     mean_shap_value_over_the_transition = np.mean(the_coalition_the_transition_sum_over_coals)
#     shap_value = mean_shap_value_over_the_transition

#     return coun, eff, shap_value


def process_coalitions(start, end, all_coallitions_, all_coallitions_without_one, all_coallitions_with_eff):
    mask_for_the_effect_without_it = [True] * (end - start)
    mask_for_the_effect_with_it = [True] * (end - start)

    for idx, coal in enumerate(all_coallitions_[start:end]):
        if set(coal) not in all_coallitions_without_one:
            mask_for_the_effect_without_it[idx] = False
        if set(coal) not in all_coallitions_with_eff:
            mask_for_the_effect_with_it[idx] = False

    return mask_for_the_effect_without_it, mask_for_the_effect_with_it

def parallelize_coalition_check(all_coallitions_, all_coallitions_without_one, all_coallitions_with_eff, num_processes):
    chunk_size = len(all_coallitions_) // num_processes
    processes = []
    results = []
    
    manager = mp.Manager()
    mask_for_the_effect_without_it_trans_by_coals = manager.list([True] * len(all_coallitions_))
    mask_for_the_effect_with_it_trans_by_coals = manager.list([True] * len(all_coallitions_))

    with mp.Pool(num_processes) as pool:
        tasks = []
        for i in range(num_processes):
            start = i * chunk_size
            # Ensure the last chunk includes the remaining elements
            end = len(all_coallitions_) if i == num_processes - 1 else (i + 1) * chunk_size

            tasks.append(pool.apply_async(
                process_coalitions,
                (start, end, all_coallitions_, all_coallitions_without_one, all_coallitions_with_eff)
            ))

        # Collect results from all processes
        for task in tasks:
            result_without_it, result_with_it = task.get()
            results.append((result_without_it, result_with_it))

    # Combine results
    final_mask_without_it = []
    final_mask_with_it = []
    for res in results:
        final_mask_without_it.extend(res[0])
        final_mask_with_it.extend(res[1])

    return final_mask_without_it, final_mask_with_it



# Function to check the conditions for a single coalition
def check_coalition(index_and_coal):
    index, coal = index_and_coal
    without_it = set(coal) not in all_coallitions_without_one
    with_it = set(coal) not in all_coallitions_with_eff
    return index, without_it, with_it



def process_effect(coun, eff, effects_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key):
    #global all_coallitions_
    
    # The main computation logic remains the same.
    all_effs_without_one = effects_list.copy()
    all_effs_without_one.remove(eff)
    global all_coallitions_without_one
    global all_coallitions_with_eff
    all_coallitions_without_one = []
    all_coallitions_with_eff = []

    last_current_time = time.time()
    time_spent = time.time() - last_current_time
    #print("In process_effect start is ", str(time_spent))


    for r in range(1, len(all_effs_without_one) + 1):
        combinaisons = list(itertools.combinations(all_effs_without_one, r))
        combinaisons = [set(combb) for combb in combinaisons]
        all_coallitions_without_one.extend(combinaisons)
        for comb in combinaisons:
            list_ = list(comb)
            list_.append(eff)
            all_coallitions_with_eff.append(set(list_))

    time_spent =  time.time() - last_current_time
    #print("In process_effect 01 is ", str(time_spent))
    last_current_time = time.time()

    coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
    coals_weights = np.array(coals_weights)

    time_spent =  time.time() - last_current_time
    #print("In process_effect 02 is ", str(time_spent))
    last_current_time = time.time()

    # mask_for_the_effect_without_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)
    # mask_for_the_effect_with_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)

    # for jjj, coal in enumerate(all_coallitions_):
    #     if set(coal) not in all_coallitions_without_one:
    #         mask_for_the_effect_without_it_trans_by_coals[0][jjj] = False
    #     if set(coal) not in all_coallitions_with_eff:
    #         mask_for_the_effect_with_it_trans_by_coals[0][jjj] = False

    print("starting computing computing masks")

    num_coalitions = len(all_coallitions_)

    # Shared arrays for masks
    mask_for_the_effect_without_it_trans_by_coals = np.full((1, num_coalitions), True)
    mask_for_the_effect_with_it_trans_by_coals = np.full((1, num_coalitions), True)

    # Create a pool of workers
    with Pool() as pool:
        # Use tqdm for progress tracking
        results = list(tqdm(pool.imap(check_coalition, enumerate(all_coallitions_)), total=num_coalitions))

    # Process results
    for index, without_it, with_it in results:
        mask_for_the_effect_without_it_trans_by_coals[0][index] = not without_it
        mask_for_the_effect_with_it_trans_by_coals[0][index] = not with_it




    time_spent =  time.time() - last_current_time
    #print("In process_effect 04 is ", str(time_spent))
    last_current_time = time.time()

    reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(1 * nber_of_coals)
    distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]

    reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(1 * nber_of_coals)
    distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

    time_spent =  time.time() - last_current_time
    #print("In process_effect 05 is ", str(time_spent))
    last_current_time = time.time()

    max_distance_the_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_without_one), axis=0)
    max_distance_the_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_with_eff), axis=0)

    time_spent =  time.time() - last_current_time
    #print("In process_effect 06 is ", str(time_spent))
    last_current_time = time.time()

    profits_of_the_without_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_without_one)
    profits_of_the_with_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_with_one)

    time_spent =  time.time() - last_current_time
    #print("In process_effect 07 is ", str(time_spent))
    last_current_time = time.time()

    substraction = profits_of_the_with_eff_coalition_for_the_transition - profits_of_the_without_eff_coalition_for_the_transition
    the_coalition_the_transition = coals_weights * substraction

    time_spent =  time.time() - last_current_time
    #print("In process_effect 08 is ", str(time_spent))
    last_current_time = time.time()

    the_coalition_the_transition_sum_over_coals = np.sum(the_coalition_the_transition, axis=0)
    
    mean_shap_value_over_the_transition = np.mean(the_coalition_the_transition_sum_over_coals)
    shap_value = mean_shap_value_over_the_transition

    time_spent =  time.time() - last_current_time
    #print("In process_effect 09 is ", str(time_spent))
    last_current_time = time.time()

    return coun, eff, shap_value




def unnormalize_colors(normalized_images, mean, std): 
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



def denorm_worker(args):
    """Worker function to process a chunk of images."""
    theims_chunk, mean_to_use_, std_to_use_, orig_min_, orig_max_ = args

    def denorm_single(theim):
        unorm = unnormalize_colors(theim, mean_to_use_, std_to_use_)
        dehanced = deenhance(unorm)
        denormalized = denormalize(dehanced, orig_min_, orig_max_)
        return np.clip(denormalized, 0, 1)

    return [denorm_single(theim) for theim in theims_chunk]




def denorm(theims, mean_to_use_, std_to_use_, orig_min_, orig_max_, optional_str=""):
    print("Begin Denorm")
    """Main function to process images using multiprocessing."""
    #print("in denorm 0")
    # Determine the number of CPUs to use
    num_cpus = cpu_count()
    print("num_cpus is {}".format(str(num_cpus)))

    # Split the images into chunks for parallel processing
    chunk_size = len(theims) // num_cpus + (len(theims) % num_cpus > 0)

    # theims = 514  et num_cpu = 256
    #    2 + 2 = 4
    #             *
    print("in denorm 1")
    chunks = [theims[i:i + chunk_size] for i in range(0, len(theims), chunk_size)]
    print("in denorm 2")
    # Prepare arguments for each worker
    args = [(chunk, mean_to_use_, std_to_use_, orig_min_, orig_max_) for chunk in chunks]
    print("in denorm 3")

    # Function to track progress
    def track_progress(index, result):
        print(f"\rin {optional_str}, chunk {index + 1}/{len(chunks)} processed.", end='', flush=False)

    # Use Pool to process the chunks in parallel
    with Pool(num_cpus) as pool:
        results = []
        for i, result in enumerate(pool.map(denorm_worker, args)):
            print("in pool, i is {}".format(str(i)))
            track_progress(i, result)
            results.append(result)


    print("in denorm 4")
    # Flatten the list of results
    return [item for sublist in results for item in sublist]


def repeat_chunk(data, repeats):
    # This function will handle a chunk of the repeat task
    return np.repeat(np.expand_dims(data, axis=0), repeats=repeats, axis=0)

def parallel_repeat(array, n_repeats):

    n_jobs = cpu_count() // 2

    # Calculate the chunk size for each process
    chunk_size = n_repeats // n_jobs
    extra = n_repeats % n_jobs

    # Create the tasks (repeats for each chunk)
    chunks = [chunk_size + 1 if i < extra else chunk_size for i in range(n_jobs)]

    # Create a shared memory array to store the results
    result_shape = (n_repeats, *array.shape)
    shared_result = Array('d', int(np.prod(result_shape)))  # Using 'd' for double (float64)

    # Shared variables for tracking progress
    progress_counter = Value('i', 0)  # Shared integer to track completed chunks
    progress_lock = Lock()  # Lock to ensure thread-safe updates

    # Function to handle writing to the shared memory and tracking progress
    def worker_task(start_idx, repeats):
        sub_result = repeat_chunk(array, repeats)
        np_array = np.frombuffer(shared_result.get_obj()).reshape(result_shape)
        end_idx = start_idx + sub_result.shape[0]
        np_array[start_idx:end_idx] = sub_result

        # Update progress
        with progress_lock:
            progress_counter.value += 1
            print(f"Progress: {progress_counter.value}/{len(chunks)} chunks computed")

    # Calculate start indices for each process
    start_indices = []
    start = 0
    for chunk in chunks:
        start_indices.append(start)
        start += chunk

    # Create the process pool
    processes = []
    for i, start_idx in enumerate(start_indices):
        repeats = chunks[i]
        process = mp.Process(target=worker_task, args=(start_idx, repeats))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Retrieve the final result from shared memory
    final_result = np.frombuffer(shared_result.get_obj()).reshape(result_shape)
    return final_result





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


    # def unnormalize_colors(normalized_images, mean, std): 
    #     # Reverse the normalization process
    #     # unnormalized_images = normalized_images * (std + 1e-6) + mean
    #     # return np.round(unnormalized_images).astype(np.uint8)
    #     return (normalized_images*std)+mean


    # def deenhance(enhanced_image):
    #     # Reverse the final shift by subtracting 0.5
    #     temp_image = enhanced_image - 0.5
        
    #     # Reverse the clipping: Since clipping limits values, we cannot fully recover original values if they were outside the [-0.5, 0.5] range. 
    #     # However, for values within the range, we can reverse the scale by dividing by 3.
    #     # We assume that the enhanced image has values within the range that the clip function allowed.
    #     temp_image = temp_image / 3
        
    #     # Reverse the initial shift by adding 0.5 back
    #     original_image = temp_image + 0.5
        
    #     return original_image

    # def denormalize(normalized_image, original_min, original_max):
    #     if original_max == original_min:
    #         return normalized_image + original_min
    #     else:
    #         return (normalized_image * (original_max - original_min)) + original_min

    # def denorm(theim, mean_to_use_, std_to_use_, orig_min_, orig_max_):

    #     # last_current_time = time.time()
    #     # time_spent = time.time() - last_current_time
    #     # print("THE TIME spent start is ", str(time_spent))
    #     unorm = unnormalize_colors(theim, mean_to_use_, std_to_use_)

    #     # time_spent =  time.time() - last_current_time
    #     # print("THE TIME spent 00 is ", str(time_spent))
    #     # last_current_time = time.time()
    #     dehanced = deenhance(unorm)

    #     # time_spent =  time.time() - last_current_time
    #     # print("THE TIME spent 11 is ", str(time_spent))
    #     # last_current_time = time.time()

    #     denormalized = denormalize(dehanced, orig_min_, orig_max_)
    #     # time_spent =  time.time() - last_current_time
    #     # print("THE TIME spent 22 is ", str(time_spent))
    #     # last_current_time = time.time()

    #     ret = np.clip(denormalized, 0, 1)
    #     # time_spent =  time.time() - last_current_time
    #     # print("THE TIME spent 33 is ", str(time_spent))
    #     # last_current_time = time.time()

    #     return ret

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


    #path_to_dataset = dataset_aux_json_folder_exp +"/dataPartialLast.p"


    # r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p
    loaded_data = load_dataset(path_to_dataset)

    
    all_actions_unique = loaded_data["all_actions_unique"] 
    #

    
    train_set = loaded_data["train_set"] 
    test_val_set = loaded_data["test_val_set"] 
    all_pairs_of_images_reduced_orig = loaded_data["all_pairs_of_images_reduced_orig"] 
    all_actions_one_hot = loaded_data["all_actions_one_hot"]
    #all_high_lvl_actions_one_hot = loaded_data["all_high_lvl_actions_one_hot"]
    
    mean_all = loaded_data["mean_all"] 
    std_all = loaded_data["std_all"] 
    all_actions_unique = loaded_data["all_actions_unique"] 
    #all_high_lvl_actions_unique = loaded_data["all_high_lvl_actions_unique"]
    orig_max = loaded_data["orig_max"] 
    orig_min = loaded_data["orig_min"] 
    #





    # for i in range(0, len(train_set), 500):
    #     string = all_actions_unique[np.argmax(train_set[i][1])]

    #     #string_high_lvl = all_high_lvl_actions_unique[np.argmax(train_set[i][2])]

    #     im1, im2 = train_set[i][0]

    #     denormalized = unnormalize_colors(im1, mean_all, std_all) 
    #     if(sys.argv[2]!="sokoban"):
    #         dehanced = deenhance(denormalized)
    #         denormalized = denormalize(dehanced, orig_min, orig_max)
    #     image1 = np.clip(denormalized, 0, 1)

    #     denormalized = unnormalize_colors(im2, mean_all, std_all) 
    #     if(sys.argv[2]!="sokoban"):
    #         dehanced = deenhance(denormalized)
    #         denormalized = denormalize(dehanced, orig_min, orig_max)
    #     image2 = np.clip(denormalized, 0, 1)


    #     combined_image = np.hstack((image1, image2))

    #     plt.imsave(str(i)+"_.png", combined_image)
    #     plt.close()




    # exit()



    # print(train_set[0][1])
    # print(train_set[0][2])
    # exit()
   


    # train_set ,  [all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]]


    # print("all_actions_unique") 
    # print(all_actions_unique[0]) # ['d3d1d2d4', 'd3E[d2d1]d4']

    # print(len(train_set)) # 51102

    # print(train_set[0][1])
    # print(train_set[0][2])


    # print(all_actions_unique[np.argmax(train_set[0][1])])


    # for i in range(0, len(train_set), 500):
    #     string = all_actions_unique[np.argmax(train_set[i][1])]

    #     string_high_lvl = all_high_lvl_actions_unique[np.argmax(train_set[i][2])]

    #     im1, im2 = train_set[i][0]

    #     denormalized = unnormalize_colors(im1, mean_all, std_all) 
    #     if(sys.argv[2]!="sokoban"):
    #         dehanced = deenhance(denormalized)
    #         denormalized = denormalize(dehanced, orig_min, orig_max)
    #     image1 = np.clip(denormalized, 0, 1)

    #     denormalized = unnormalize_colors(im2, mean_all, std_all) 
    #     if(sys.argv[2]!="sokoban"):
    #         dehanced = deenhance(denormalized)
    #         denormalized = denormalize(dehanced, orig_min, orig_max)
    #     image2 = np.clip(denormalized, 0, 1)


    #     combined_image = np.hstack((image1, image2))

    #     plt.imsave(str(i)+"_"+str(string_high_lvl)+".png", combined_image)
    #     plt.close()





    ##### OK

    #       1) régénère le PDDL, et vérifie i) si le nbre d'actions avant génération (des csv) est bien de 1469
    #               et ii) si le nbre d'après est 1459 (donc 10 de moins)

    #                   iii) vérifie QUELLES action ont été supprimés

    #                   iv) choisi soit de prendre en compte iii SOIT  de forcer à ne pas supprimer les 10 actions


    # train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]
    # train_set_no_dupp_orig = loaded_data["train_set_no_dupp_orig"]

    train_set_no_dupp = []
    train_set_no_dupp_orig = []

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






    # # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION
    # dico_transitions_per_high_lvl_actions = {}

    # for ii, ele in enumerate(train_set_no_dupp):
    #     if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:
    #         # add the key (the high lvl action index)
    #         dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}
    #     # if the hash is not in the keys of the dico of the high lvl action
    #     # i.e. if the transition was not added as a transition for this high lvl action
    #     if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
    #         # add the transition i.e.:
    #         # the two preprocessed images
    #         # the two <=> reduced images (not preprocessed)
    #         # the onehot repr of the high lvl action
    #         # the max_diff_in_bits (not used here)
    #         dico_transitions_per_high_lvl_actions[np.argmax(ele[2])][np.argmax(ele[1])] = {
    #             "preprocessed" : ele[0],
    #             "reduced" : train_set_no_dupp_orig[ii][0],
    #             "onehot": ele[2],
    #             #"max_diff_in_bits_": ele[2]
    #         }



    # weights_each_hl_action = {}

    # for kkkk, vvvv in dico_transitions_per_high_lvl_actions.items():
    #     weights_each_hl_action[kkkk] = len(vvvv) / len(all_actions_unique)




















    if 'compute_pos_preconds_shap_values_per_action' in args.mode or 'compute_neg_preconds_shap_values_per_action' in args.mode:

        type_of_preconds = ""

        if 'compute_pos_preconds_shap_values_per_action' in args.mode:
            type_of_preconds = "pos"
        elif 'compute_neg_preconds_shap_values_per_action' in args.mode:
            type_of_preconds = "neg"

        # train_set_no_dupp[0] is like all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]

        if os.path.isfile(os.path.join(exp_aux_json_folder,"aux.json")):
            with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
                data = json.load(f)


        last_current_time = time.time()



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

        # loop over the train set (without dupplicate)
        # AND group the transitions into their respective High Level Actions
        # [all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]]
        for ii, ele in enumerate(train_set_no_dupp):


            # if the looped transition high level action is not a key of dico_transitions_per_high_lvl_actions
            if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:

                # add the key (the high lvl action index)
                dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}

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







        # save_dataset(os.getcwd(), dico_transitions_per_high_lvl_actions, "dico_transitions_per_high_lvl_actions.p")

        #pos_preconds = np.loadtxt(os.path.join(exp_aux_json_folder,"pos_preconds_aligned.csv"), delimiter=' ', dtype=int)
        #neg_preconds = np.loadtxt(os.path.join(exp_aux_json_folder,"neg_preconds_aligned.csv"), delimiter=' ', dtype=int)

        _preconds = np.loadtxt(os.path.join(exp_aux_json_folder, str(type_of_preconds)+"_preconds_aligned.csv"), delimiter=' ', dtype=int)

        dir_shap_vals_preconds = os.path.join(exp_aux_json_folder, "shap_vals_"+str(type_of_preconds)+"_preconds")
        if not os.path.isdir(dir_shap_vals_preconds):
            os.mkdir(dir_shap_vals_preconds)

        max_number_of_preconds = 0


        # for each high level action
        for high_lvl_ac_index, transitions in dico_transitions_per_high_lvl_actions.items():


            print("Now processing action {}".format(str(high_lvl_ac_index)))

            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR (starting a new ACTION) ", str(time_spent))
            last_current_time = time.time()

            # we put all (preprocessed) transitions images of the action in arrays
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

            # we encode the "SUCC" images
            zs_succ_s = net.encode(all_preproc_im2) # zs_pre_s[i] would be the encoded PRE 
                                                   # STATE OF THE ith transition of the action


            # FOR EACH TRANSITION, we retrieve the NEG/POS PRECONDITIONS
            # and we compute the COALITIONS 

            for ij, (key, trans) in enumerate(transitions.items()):

                time_spent = time.time() - last_current_time
                #print("BEGIINING TIME ", str(time_spent))
                last_current_time = time.time()


                _preconds_for_the_transition = _preconds[key]
                #neg_preconds_for_the_transition = neg_preconds[key]
     
                _preconds_for_the_transition = np.where(_preconds_for_the_transition == 1)[0]
                #neg_preconds_for_the_transition = np.where(neg_preconds_for_the_transition == 1)[0]
                
                nine_in_preconds = np.where(_preconds_for_the_transition == 9)[0]
                #nine_in_neg = np.where(neg_preconds_for_the_transition == 9)[0]

                # nines means there was no effects for the transition/action at hand, so we do not compute SHAP values
                # (we dont compute the SHAP neither for the preconditions)
                if len(nine_in_preconds) > 0:
                    continue


                # make a list of preconds "pos_"+i, "neg_"+i
                # (actually, since there are way too much preconds 
                # and their combinations is exponential, we just consider the positive 
                # preconds (and then eventually, separately, the negative ones))
                preconds_list = []
                for _precond in _preconds_for_the_transition:
                    preconds_list.append("z_"+str(_precond))

                # for neg_precond in neg_preconds_for_the_transition:
                #     preconds_list.append("not_z"+str(neg_precond))

      
                p = len(preconds_list)

                print("p is {}".format(p))

                if p == 0:
                    print("did not compute coz p is 0")
                    continue

                if p > max_number_of_preconds:
                    max_number_of_preconds = p
                

                #print(preconds_list)



                # # 
                # # LIST OF ALL POSSIBLE COALLITIONS OF THE PRECONDS
                # all_coallitions_ = []
                # for r in range(0, len(preconds_list) + 1):
                #     combinaisons = list(itertools.combinations(preconds_list, r))
                #     all_coallitions_.extend(combinaisons)

                # we use parallelization, coz too much preconds (compared to effects)
                all_coallitions_ = parallel_combinations(preconds_list)


                dire = "/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/"
                save_dataset(dire, all_coallitions_, "all_coallitions_action_"+str(high_lvl_ac_index)+"_trans_"+str(key)+"_"+str(type_of_preconds)+"_preconds.p")

                #all_coallitions_ = load_dataset(dire+"all_coallitions_.p")["saved"]

                # #
                # print("ici0")
                # print(net.decoder.summary())

                #Trainable params: 1,524,039
                #Non-trainable params: 56,612

                # from keras.utils.layer_utils import count_params
                # trainable_params = sum(count_params(layer) for layer in net.decoder.trainable_weights)
                # non_trainable_params = sum(count_params(layer) for layer in net.decoder.non_trainable_weights)

                # print("trainable_params :  {}".format(str(trainable_params)))
                # print("non_trainable_params :  {}".format(str(non_trainable_params)))


                #all_coallitions_ = all_coallitions_[:5000]

                nber_of_coals = len(all_coallitions_)
                # mask of pos / neg (also (nber_coals x 50))

                _masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)
                ##neg_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)

                print("all_coallitions_[:5]")
                print(all_coallitions_[:5])

                # MASKS of the (pos or neg) PRECONDS for each coalitions
                for iii, coal in enumerate(all_coallitions_):
                    poss_or_negs = [int(command.split('_')[1]) for command in coal if command.startswith("z")]
                    _masks_all_coals[iii][poss_or_negs] = 1
                    # negs = [int(command.split('_')[1]) for command in coal if command.startswith("del")]
                    # del_masks_all_coals[iii][negs] = 1

                # 1) perint le nbre MAX d'effet PARMIS TOUT !!!!!


                print("ici1")
                # super TENSOR OF: coals / latent
                # i.e. each item is a coalition of the transition (being looper over) 
                # contains, for each coalition, for the transition, the coalition applied to the z_pre state
                coals_latent = np.zeros((nber_of_coals, 50), dtype=int)
                for iiii, (pos_m) in enumerate(_masks_all_coals):
                    zs_succ_s_ = zs_succ_s.copy()
                    zs_succ_s_[ij][np.where(pos_m == 1)[0]] = 1
                    coals_latent[iiii] = zs_succ_s_[ij]
                    del zs_succ_s_
                    
                coals_latent_decoded = net.decode(coals_latent)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 0: ", str(time_spent))
                last_current_time = time.time()

                unorm = unnormalize_colors(np.squeeze(coals_latent_decoded), mean_all, std_all)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-0: ", str(time_spent))
                last_current_time = time.time()

                dehanced = deenhance(unorm)
            
                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-1: ", str(time_spent))
                last_current_time = time.time()

                denormalized = denormalize(dehanced, orig_min, orig_max)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-2: ", str(time_spent))
                last_current_time = time.time()

                #coals_latent_denormed = denorm(np.squeeze(coals_latent_decoded), mean_all, std_all, orig_min, orig_max, "FIRST DENORM")
                coals_latent_denormed = np.clip(denormalized, 0, 1)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-3: ", str(time_spent))
                last_current_time = time.time()



                # Taking the second image of the transition and dupplicating it (np.repeat)
                #current_preproc_im1_transi_by_coals = np.repeat(np.expand_dims(all_preproc_im1[ij], axis=0), repeats=nber_of_coals, axis=0)
                current_preproc_im1_transi_by_coals = parallel_repeat(all_preproc_im1[ij], nber_of_coals)
          
                time_spent = time.time() - last_current_time
                print("TIME AFTER parallel repeat: ", str(time_spent))
                last_current_time = time.time()



                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-0: ", str(time_spent))
                last_current_time = time.time()

                unorm = unnormalize_colors(np.squeeze(current_preproc_im1_transi_by_coals), mean_all, std_all)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-1: ", str(time_spent))
                last_current_time = time.time()

                dehanced = deenhance(unorm)
            
                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-2: ", str(time_spent))
                last_current_time = time.time()

                denormalized = denormalize(dehanced, orig_min, orig_max)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-3: ", str(time_spent))
                last_current_time = time.time()

                current_preproc_im1_transi_by_coals_denormed = np.clip(denormalized, 0, 1)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-4: ", str(time_spent))
                last_current_time = time.time()


                print("Starting Computing Distances")
            
                # compute the distance btween the decoded coals applied to first state and the second image
                dists = compute_distances(coals_latent_denormed, current_preproc_im1_transi_by_coals_denormed)


                time_spent = time.time() - last_current_time
                print("Computed distances: ", str(time_spent))
                last_current_time = time.time()


                # take the max distance as a reference for computing the SHAPs
                max_distance_for_the_transition  = np.max(dists, axis=0)

                p = len(preconds_list)




                results = []

                tasks = [(coun, precond, preconds_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key) for coun, precond in enumerate(preconds_list)]

                # we use process_effect instead of process_preconds coz calculations are the same
                for task in tasks:
                    tmp_result = process_effect(*task)
                    results.append(tmp_result)

                time_spent = time.time() - last_current_time
                print("Finished Processing SHAPs for the transition ", str(time_spent))
                last_current_time = time.time()





                # # Prepare the list of inputs to process, and other required global variables
                # tasks = [(coun, precond, preconds_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key) for coun, precond in enumerate(preconds_list)]
                # # Use Pool.starmap to process the tasks in parallel
                # # we use process_effect fct here since it s same calculation for the preconds as for the effects....
                # with Pool() as pool:
                #     results = pool.starmap(process_effect, tasks)

                # # coun, eff, shap_value

                with open(dir_shap_vals_preconds+"/action_"+str(high_lvl_ac_index)+"_withEmptySet.txt", "a+") as file:
                    file.write("transition "+str(key)+"\n")

                    # Print the results
                    for coun, precond, shap_value in results:
                        print("shap_value for action {}, transition {} and precond: {} is {}".format(
                            str(high_lvl_ac_index), str(key), precond, str(shap_value)
                        ))
                        file.write(f"{precond} {shap_value}\n")

                        
                    file.write("\n")

        #print("max_number_of_preconds is {}".format(str(max_number_of_preconds)))




    if 'compute_effects_shap_values_per_action' in args.mode:



        last_current_time = time.time()


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

        # "aligned" ? coz i think i removed dupplicates in the effects (afterwhich there are less effects than preconds)
        add_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"add_effs_aligned.csv"), delimiter=' ', dtype=int)
        del_effects = np.loadtxt(os.path.join(exp_aux_json_folder,"del_effs_aligned.csv"), delimiter=' ', dtype=int)



        dir_shap_vals_effects_no_touching = os.path.join(exp_aux_json_folder, 'shap_vals_effects_no_touching')
        if not os.path.isdir(dir_shap_vals_effects_no_touching):
            os.mkdir(dir_shap_vals_effects_no_touching)

        
        dir_shap_vals_persisting_effects_removed = os.path.join(exp_aux_json_folder, 'shap_vals_persisting_effects_removed')
        if not os.path.isdir(dir_shap_vals_persisting_effects_removed):
            os.mkdir(dir_shap_vals_persisting_effects_removed)

        lecounter = 0

        # for each high level action


        # max_p = 0

        for high_lvl_ac_index, transitions in dico_transitions_per_high_lvl_actions.items():

            # 

            print("Now processing action {}".format(str(high_lvl_ac_index)))

            # we now compute for each transition, all the SHAP values of all the effects

            # FOR THIS WE NEED:

            # 1) loop over transitions, loop over the effects of the transition

            #           JUST BEFORE the 2nd loop SET SHAP_value_of_effect_for_transition TO 0

            # time_spent = time.time() - last_current_time
            # print("time spent start is ", str(time_spent))
            # last_current_time = time.time()

            # we put all (preprocessed) transitions images of the action in arrays
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

            # we encode the "PRE" images
            zs_pre_s = net.encode(all_preproc_im1) # zs_pre_s[i] would be the encoded PRE 
                                                   # STATE OF THE ith transition of the action


            # FOR EACH TRANSITION, we retrieve the DEL/ADD effects
            # and we compute the COALITIONS 
            for ij, (key, trans) in enumerate(transitions.items()):

                time_spent = time.time() - last_current_time
                #print("BEGIINING TIME ", str(time_spent))
                last_current_time = time.time()


                #print("transition {} / {}".format(str(ij), len(transitions.items())))

                add_effects_for_the_transition = add_effects[key]
                del_effects_for_the_transition = del_effects[key]

                add_effects_for_the_transition = np.where(add_effects_for_the_transition == 1)[0]
                del_effects_for_the_transition = np.where(del_effects_for_the_transition == 1)[0]
     
                
                nine_in_add = np.where(add_effects_for_the_transition == 9)[0]
                nine_in_del = np.where(del_effects_for_the_transition == 9)[0]

                # nines means there was no effects for the transition/action at hand, so we do not compute SHAP values
                if len(nine_in_add) > 0 or len(nine_in_del) > 0:
                    continue

                # make a list of effects "add_"+i, "del_"+i
                effects_list = []
                for add_eff in add_effects_for_the_transition:
                    effects_list.append("add_"+str(add_eff))
                for del_eff in del_effects_for_the_transition:
                    effects_list.append("del_"+str(del_eff))

                p = len(effects_list)

                print("p is {}".format(p))

           
                # 
                # LIST OF ALL POSSIBLE COALLITIONS OF THE EFFECTS
                #   (starting with r=0 to include the empty set!)
                all_coallitions_ = []
                for r in range(0, len(effects_list) + 1):
                    combinaisons = list(itertools.combinations(effects_list, r))
                    all_coallitions_.extend(combinaisons)

                        
                nber_of_coals = len(all_coallitions_)
                # mask of add / del (also (nber_coals x 50))
                add_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)
                del_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)

                # MASKS of the ADD and the DEL EFFECTS for each coalitions
                for iii, coal in enumerate(all_coallitions_):
                    adds = [int(command.split('_')[1]) for command in coal if command.startswith("add")]
                    add_masks_all_coals[iii][adds] = 1
                    dels = [int(command.split('_')[1]) for command in coal if command.startswith("del")]
                    del_masks_all_coals[iii][dels] = 1


                # super TENSOR OF: coals / latent
                # i.e. each item is a coalition of the transition (being looper over) 
                # contains, for each coalition, for the transition, the coalition applied to the z_pre state
                coals_latent = np.zeros((nber_of_coals, 50), dtype=int)
                for iiii, (add_m, del_m) in enumerate(zip(add_masks_all_coals, del_masks_all_coals)):
                    zs_pre_s_ = zs_pre_s.copy()
                    
                    zs_pre_s_[ij][np.where(add_m == 1)[0]] = 1
                    zs_pre_s_[ij][np.where(del_m == 1)[0]] = 0
              
                    coals_latent[iiii] = zs_pre_s_[ij]
                    del zs_pre_s_
        

                time_spent = time.time() - last_current_time
                print("TIME 1: ", str(time_spent))
                last_current_time = time.time()

                # decoding and denorming of all the coals applied to the z_pre of the transition
                coals_latent_decoded = net.decode(coals_latent)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 0: ", str(time_spent))
                last_current_time = time.time()

                unorm = unnormalize_colors(np.squeeze(coals_latent_decoded), mean_all, std_all)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-0: ", str(time_spent))
                last_current_time = time.time()

                dehanced = deenhance(unorm)
            
                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-1: ", str(time_spent))
                last_current_time = time.time()

                denormalized = denormalize(dehanced, orig_min, orig_max)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-2: ", str(time_spent))
                last_current_time = time.time()

                #coals_latent_denormed = denorm(np.squeeze(coals_latent_decoded), mean_all, std_all, orig_min, orig_max, "FIRST DENORM")
                coals_latent_denormed = np.clip(denormalized, 0, 1)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 1-3: ", str(time_spent))
                last_current_time = time.time()

                # with h5py.File('H5_coals_latent_denormed.h5', 'w') as hf:
                #     hf.create_dataset('H5_coals_latent_denormed', data=coals_latent_denormed)


                # Taking the second image of the transition and dupplicating it (np.repeat)
                current_preproc_im2_transi_by_coals = np.repeat(np.expand_dims(all_preproc_im2[ij], axis=0), repeats=nber_of_coals, axis=0)

                #current_preproc_im2_transi_by_coals_denormed = denorm(np.squeeze(current_preproc_im2_transi_by_coals), mean_all, std_all, orig_min, orig_max, "SECOND DENORM")

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-0: ", str(time_spent))
                last_current_time = time.time()

                unorm = unnormalize_colors(np.squeeze(current_preproc_im2_transi_by_coals), mean_all, std_all)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-1: ", str(time_spent))
                last_current_time = time.time()

                dehanced = deenhance(unorm)
            
                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-2: ", str(time_spent))
                last_current_time = time.time()

                denormalized = denormalize(dehanced, orig_min, orig_max)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-3: ", str(time_spent))
                last_current_time = time.time()

                current_preproc_im2_transi_by_coals_denormed = np.clip(denormalized, 0, 1)

                time_spent = time.time() - last_current_time
                print("TIME Begin denorm 2-4: ", str(time_spent))
                last_current_time = time.time()

                # with h5py.File('H5_current_preproc_im2_transi_by_coals_denormed.h5', 'w') as hf:
                #     hf.create_dataset('H5_current_preproc_im2_transi_by_coals_denormed', data=current_preproc_im2_transi_by_coals_denormed)


                print("Starting Computing Distances")

                # compute the distance btween the decoded coals applied to first state and the second image
                dists = compute_distances(coals_latent_denormed, current_preproc_im2_transi_by_coals_denormed)
            

                # with h5py.File('H5_coals_latent_denormed.h5', 'w') as hf:
                #     hf.create_dataset('H5_coals_latent_denormed', data=coals_latent_denormed)


                time_spent = time.time() - last_current_time
                print("Computed distances: ", str(time_spent))
                last_current_time = time.time()

                # take the max distance as a reference for computing the SHAPs
                max_distance_for_the_transition  = np.max(dists, axis=0)

                p = len(effects_list)


                # time_spent =  time.time() - last_current_time
                # print("THE TIME spent starting effects ", str(time_spent))
                # last_current_time = time.time()


                # Prepare the list of inputs to process, and other required global variables
                #tasks = [(coun, eff, effects_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key) for coun, eff in enumerate(effects_list)]

                # # Use Pool.starmap to process the tasks in parallel
                # with Pool() as pool:
                #     results = pool.starmap(process_effect, tasks)

                # si 20 effets par ex:
                #
                #
                #
                #           Mieux de faire effet par effet MAIS de départer à 'lintérieur de process_effect

                #

                # with Manager() as manager:
                #     # Shared counter for progress tracking
                #     progress_counter = manager.Value('i', 0)
                #     with Pool() as pool:
                #         results = pool.starmap(process_effect, [(task + (progress_counter,)) for task in tasks])


                results = []
                tasks = [(coun, eff, effects_list, all_coallitions_, p, dists, max_distance_for_the_transition, nber_of_coals, high_lvl_ac_index, key) for coun, eff in enumerate(effects_list)]

                for task in tasks:
                    tmp_result = process_effect(*task)
                    results.append(tmp_result)

                time_spent = time.time() - last_current_time
                print("Finished Processing SHAPs for the transition ", str(time_spent))
                last_current_time = time.time()

                # coun, eff, shap_value

                with open(dir_shap_vals_persisting_effects_removed+"/action_"+str(high_lvl_ac_index)+"_withEmptySet.txt", "a+") as file:
                    file.write("transition "+str(key)+"\n")

                    # Print the results
                    for coun, eff, shap_value in results:
                        print("shap_value for action {}, transition {} and eff: {} is {}".format(
                            str(high_lvl_ac_index), str(key), eff, str(shap_value)
                        ))
                        file.write(f"{eff} {shap_value}\n")

                        
                    file.write("\n")


                # time_spent =  time.time() - last_current_time
                # print("THE TIME spent END effectS", str(time_spent))
                # last_current_time = time.time()


                # # FOR EACH EFFECT OF THE CURRENT TRANSITION (of the current high lvl action)
                # for coun, eff in enumerate(effects_list):
                        
                #     # time_spent =  time.time() - last_current_time
                #     # print(" (starting an effect) time spent 06 is ", str(time_spent))
                #     # last_current_time = time.time()

                #     # if coun > 2:
                #     #     break

                #     # need some lists with and without the effects
                #     all_effs_without_one = effects_list.copy()
                #     all_effs_without_one.remove(eff)
                #     all_coallitions_without_one = []
                #     all_coallitions_with_eff = []

                #     for r in range(1, len(all_effs_without_one) + 1):
                #         combinaisons = list(itertools.combinations(all_effs_without_one, r))
                #         combinaisons = [set(combb) for combb in combinaisons]
                #         all_coallitions_without_one.extend(combinaisons)
                #         for comb in combinaisons:
                #             list_ = list(comb)
                #             list_.append(eff)
                #             all_coallitions_with_eff.append(set(list_))
                #             #all_coallitions_with_eff.append(tuple(list_))

                #     del all_effs_without_one

                #     # time_spent =  time.time() - last_current_time
                #     # print(" (in an effect) time spent 07 is ", str(time_spent))
                #     # last_current_time = time.time()

                #     # doit être le array des weights de ttes les coals SANS le player p

                #     coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
                #     coals_weights = np.array(coals_weights)


                #     # tells for each transition, and for any coal, of the coal 
                #     # belongs to the group of all_coallitions_without_one (or of all_coallitions_with_eff)
                #     mask_for_the_effect_without_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)
                #     mask_for_the_effect_with_it_trans_by_coals = np.full((1, len(all_coallitions_)), True)

  

                #     thecount = 0
                #     for jjj, coal in enumerate(all_coallitions_):
                #         if set(coal) not in all_coallitions_without_one:
                #             mask_for_the_effect_without_it_trans_by_coals[0][jjj] = False
                #         else:
                #             thecount += 1
                #         if set(coal) not in all_coallitions_with_eff:
                #             mask_for_the_effect_with_it_trans_by_coals[0][jjj] = False

                #     #print("the count is {}".format(thecount))

                    
                #     reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(1*nber_of_coals)
                #     distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]
    
                #     reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(1*nber_of_coals)
                #     distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

        

                #     # faut max_distance_each_transition_repeated (#transitions, #coals)
                #     max_distance_the_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_without_one), axis=0)
                #     # (#transitions x #coals_without_one)
                #     max_distance_the_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_the_transition, repeats=len(all_coallitions_with_eff), axis=0)

                #     # time_spent =  time.time() - last_current_time
                #     # print(" (in an effect) time spent 09 is ", str(time_spent))
                #     # last_current_time = time.time()

                #     profits_of_the_without_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_without_one)
                #     profits_of_the_with_eff_coalition_for_the_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_the_transition_repeated_for_all_coallitions_with_one)

                #     # print("finalement")
                #     # print(profits_of_the_without_eff_coalition_for_the_transition.shape)


                #     substraction = profits_of_the_with_eff_coalition_for_the_transition - profits_of_the_without_eff_coalition_for_the_transition
                #     the_coalition_the_transition = coals_weights * substraction

        
                #     # print("np.sum(each_coalition_each_transition)")
                #     the_coalition_the_transition_sum_over_coals = np.sum(the_coalition_the_transition, axis=0)
                    
                #     mean_shap_value_over_the_transition = np.mean(the_coalition_the_transition_sum_over_coals)
            
                #     shap_value = mean_shap_value_over_the_transition


                    
                #     # time_spent =  time.time() - last_current_time
                #     # print(" (ending effect) time spent 10 is ", str(time_spent))
                #     # last_current_time = time.time()

                #     print("shap_value for action {}, transition {} and eff: {} is {}".format(str(high_lvl_ac_index), str(key), eff, str(shap_value)))


                # time_spent =  time.time() - last_current_time
                # print("THE TIME spent END effectS", str(time_spent))
                # last_current_time = time.time()


                # exit()
            lecounter += 1


        # print("MAX P IS {}".format(str(max_p)))
        # exit()
        #     indices_of_transitions = np.array(list(transitions.keys()))
        #     add_effects_for_the_action = add_effects[indices_of_transitions]
        #     del_effects_for_the_action = del_effects[indices_of_transitions]

        #     print("there are {} transitions".format(str(len(transitions))))

        #     # # loop over the array of transitions effects (each array is a effect's 0/1 array)
        #     # # for each, retrieve the effect number (indices)
        #     # # and add it IF not already in the list
        #     # add_effects_for_the_action_ = []
        #     # for trans_effects in add_effects_for_the_action:
        #     #     add_effects_for_the_action__ = np.where(trans_effects == 1)[0]
        #     #     for eff in add_effects_for_the_action__:
        #     #         if eff not in add_effects_for_the_action_:
        #     #             add_effects_for_the_action_.append(eff)
        #     # add_effects_for_the_action = add_effects_for_the_action_


        #     time_spent =  time.time() - last_current_time
        #     print("time spent 00 is ", str(time_spent))
        #     last_current_time = time.time()




        #     # del_effects_for_the_action_ = []
        #     # for trans_effects in del_effects_for_the_action:
        #     #     del_effects_for_the_action__ = np.where(trans_effects == 1)[0]
        #     #     for eff in del_effects_for_the_action__:
        #     #         if eff not in del_effects_for_the_action_:
        #     #             del_effects_for_the_action_.append(eff)
        #     # del_effects_for_the_action = del_effects_for_the_action_





        #     time_spent =  time.time() - last_current_time
        #     print("time spent 11 is ", str(time_spent))
        #     last_current_time = time.time()

        #     all_preproc_im1 = []
        #     all_preproc_im2 = []
        #     all_reduced_im1 = []
        #     all_reduced_im2 = []

        #     for val in transitions.values():

        #         preproc_im1, preproc_im2 = val["preprocessed"]
        #         all_preproc_im1.append(preproc_im1)
        #         all_preproc_im2.append(preproc_im2)

        #         reduced_im1, reduced_im2 = val["reduced"]
        #         all_reduced_im1.append(reduced_im1)
        #         all_reduced_im2.append(reduced_im2)

        #     all_preproc_im1 = np.array(all_preproc_im1)
        #     all_preproc_im2 = np.array(all_preproc_im2)

        #     all_reduced_im1 = np.array(all_reduced_im1)
        #     all_reduced_im2 = np.array(all_reduced_im2)



        #     time_spent =  time.time() - last_current_time
        #     print("time spent 22 is ", str(time_spent))
        #     last_current_time = time.time()


        #     # PRE LATENT STATES (encoded)
        #     zs_pre_s = net.encode(all_preproc_im1)
        #     zs_sucs_s = np.squeeze(zs_pre_s.copy())

        #     # SUCC LATENT STATES (predicted)
        #     zs_sucs_s[:, add_effects_for_the_action] = 1
        #     zs_sucs_s[:, del_effects_for_the_action] = 0
     

        #     nber_of_coals = len(all_coallitions_)
        #     # mask of add / del (also (nber_coals x 50))
        #     add_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)
        #     del_masks_all_coals = np.zeros((nber_of_coals, 50), dtype=int)

        #     # MASKS of the ADD and the DEL EFFECTS for each coalitions
        #     for iii, coal in enumerate(all_coallitions_):
        #         adds = [int(command.split('_')[1]) for command in coal if command.startswith("add")]
        #         add_masks_all_coals[iii][adds] = 1
        #         dels = [int(command.split('_')[1]) for command in coal if command.startswith("del")]
        #         del_masks_all_coals[iii][dels] = 1

        #     # super TENSOR OF: coals / transitions / latent
        #     # i.e. each item is a coalition of a transition 
        #     # contains, for each coalition, for each transition, the z_pre state applied to the coalition
        #     coals_transis_latent = np.zeros((nber_of_coals, len(transitions), 50), dtype=int)
        #     for iiii, (add_m, del_m) in enumerate(zip(add_masks_all_coals, del_masks_all_coals)):
        #         zs_pre_s_ = zs_pre_s.copy()
        #         zs_pre_s_[:,np.where(add_m == 1)[0]] = 1
        #         zs_pre_s_[:,np.where(del_m == 1)[0]] = 0
  
        #         coals_transis_latent[iiii] = zs_pre_s_
        #         del zs_pre_s_

        #     transis_coals_latent = np.transpose(coals_transis_latent, (1, 0, 2))
        #     shape = transis_coals_latent.shape


        #     # coals_transis_latent becomes of shape (#transis x #coals, #latent)
        #     transis_coals_latent_two_dims = transis_coals_latent.reshape(shape[0] * shape[1], shape[2])
        #     del transis_coals_latent
        #     # decoding and denorming of all the coals applied to the z_pre of each transition
        #     transis_coals_latent_two_dims_decoded = net.decode(transis_coals_latent_two_dims)
        #     transis_coals_latent_two_dims_denormed = denorm(np.squeeze(transis_coals_latent_two_dims_decoded), mean_to_use, std_to_use, orig_min, orig_max)


        #     # 
        #     all_preproc_im2_transi_by_coals = np.repeat(all_preproc_im2, repeats=nber_of_coals, axis=0)
        #     all_reduced_im2_transi_by_coals = np.repeat(all_reduced_im2, repeats=nber_of_coals, axis=0)
        #     all_preproc_im2_transi_by_coals_denormed = denorm(np.squeeze(all_preproc_im2_transi_by_coals), mean_to_use, std_to_use, orig_min, orig_max)

        #     dists = compute_distances(transis_coals_latent_two_dims_denormed, all_preproc_im2_transi_by_coals_denormed)
        #     reshaped_dists = dists.reshape(len(transitions), nber_of_coals)
        #     max_distance_for_each_transition  = np.max(reshaped_dists, axis=1)
        #     p = len(effects_list)

        #     #exit()

        #     for coun, eff in enumerate(effects_list):
                    
        #         time_spent =  time.time() - last_current_time
        #         print(" (starting an effect) time spent 06 is ", str(time_spent))
        #         last_current_time = time.time()

        #         # if coun > 2:
        #         #     break

        #         # need some lists with and without the effects
        #         all_effs_without_one = effects_list.copy()
        #         all_effs_without_one.remove(eff)
        #         all_coallitions_without_one = []
        #         all_coallitions_with_eff = []

        #         for r in range(1, len(all_effs_without_one) + 1):
        #             combinaisons = list(itertools.combinations(all_effs_without_one, r))
        #             combinaisons = [set(combb) for combb in combinaisons]
        #             all_coallitions_without_one.extend(combinaisons)
        #             for comb in combinaisons:
        #                 list_ = list(comb)
        #                 list_.append(eff)
        #                 all_coallitions_with_eff.append(set(list_))
        #                 #all_coallitions_with_eff.append(tuple(list_))

        #         del all_effs_without_one

        #         time_spent =  time.time() - last_current_time
        #         print(" (in an effect) time spent 07 is ", str(time_spent))
        #         last_current_time = time.time()

        #         # doit être le array des weights de ttes les coals SANS le player p

        #         coals_weights = compute_coalitions_weights(all_coallitions_without_one, p)
        #         coals_weights = np.array(coals_weights)


        #         # tells for each transition, and for any coal, of the coal 
        #         # belongs to the group of all_coallitions_without_one (or of all_coallitions_with_eff)
        #         mask_for_the_effect_without_it_trans_by_coals = np.full((len(transitions), len(all_coallitions_)), True)
        #         mask_for_the_effect_with_it_trans_by_coals = np.full((len(transitions), len(all_coallitions_)), True)

        #         #for iii, transi in enumerate(transitions):
        #         for iii in range(len(transitions)):
        #             thecount = 0
        #             for jjj, coal in enumerate(all_coallitions_):
        #                 if set(coal) not in all_coallitions_without_one:
        #                     mask_for_the_effect_without_it_trans_by_coals[iii][jjj] = False
        #                 else:
        #                     thecount += 1
        #                 if set(coal) not in all_coallitions_with_eff:
        #                     mask_for_the_effect_with_it_trans_by_coals[iii][jjj] = False

        #             #print("the count is {}".format(thecount))

        #         time_spent =  time.time() - last_current_time
        #         print(" (in an effect) time spent 08 is ", str(time_spent))
        #         last_current_time = time.time()
        #         #print(" all_coallitions_ {}".format(len(all_coallitions_)))

                
        #         reshaped_mask_for_the_effect_without_it_trans_by_coals = mask_for_the_effect_without_it_trans_by_coals.reshape(len(transitions)*nber_of_coals)
        #         distances_of_the_without_effects_coals = dists[np.where(reshaped_mask_for_the_effect_without_it_trans_by_coals == 1)[0]]
   
        #         reshaped_mask_for_the_effect_with_it_trans_by_coals = mask_for_the_effect_with_it_trans_by_coals.reshape(len(transitions)*nber_of_coals)
        #         distances_of_the_with_effects_coals = dists[np.where(reshaped_mask_for_the_effect_with_it_trans_by_coals == 1)[0]]

        #         # faut max_distance_each_transition_repeated (#transitions, #coals)
        #         max_distance_each_transition_repeated_for_all_coallitions_without_one = np.repeat(max_distance_for_each_transition, repeats=len(all_coallitions_without_one), axis=0)
        #         # (#transitions x #coals_without_one)
        #         max_distance_each_transition_repeated_for_all_coallitions_with_one = np.repeat(max_distance_for_each_transition, repeats=len(all_coallitions_with_eff), axis=0)

        #         time_spent =  time.time() - last_current_time
        #         print(" (in an effect) time spent 09 is ", str(time_spent))
        #         last_current_time = time.time()

        #         profits_of_each_without_eff_coalition_for_each_transition = coalitions_profits(distances_of_the_without_effects_coals, max_distance_each_transition_repeated_for_all_coallitions_without_one)
        #         profits_of_each_with_eff_coalition_for_each_transition = coalitions_profits(distances_of_the_with_effects_coals, max_distance_each_transition_repeated_for_all_coallitions_with_one)

        #         profits_of_each_without_eff_coalition_for_each_transition = profits_of_each_without_eff_coalition_for_each_transition.reshape(len(transitions), len(all_coallitions_without_one))
        #         profits_of_each_with_eff_coalition_for_each_transition = profits_of_each_with_eff_coalition_for_each_transition.reshape(len(transitions), len(all_coallitions_with_eff))



        #         substraction = profits_of_each_with_eff_coalition_for_each_transition - profits_of_each_without_eff_coalition_for_each_transition
        #         each_coalition_each_transition = coals_weights * substraction

    
        #         # print("np.sum(each_coalition_each_transition)")
        #         each_coalition_each_transition_sum_over_coals = np.sum(each_coalition_each_transition, axis=1)
                
        #         mean_shap_value_over_transitions = np.mean(each_coalition_each_transition_sum_over_coals)
                
        #         # écrire chaque ele comme une ligne
        #         #with open(dir_shap_vals_effects_no_touching+"/action_"+str(high_lvl_ac_index)+".txt", "a+") as file:
        #         with open(dir_shap_vals_persisting_effects_removed+"/action_"+str(high_lvl_ac_index)+".txt", "a+") as file:
        #             file.write("for effect "+str(eff)+"\n")
        #             #with open('array_elements.txt', 'w') as file:
        #             for element in each_coalition_each_transition_sum_over_coals:
        #                 file.write(f"{element}\n")
        #             file.write("\n")

        #         shap_value = mean_shap_value_over_transitions


                 
        #         time_spent =  time.time() - last_current_time
        #         print(" (ending effect) time spent 10 is ", str(time_spent))
        #         last_current_time = time.time()

        #         print("shap_value for action {} and eff: {} is {}".format(str(high_lvl_ac_index), eff, str(shap_value)))


        # exit()

    # load the json file from the base domain folder (in order to update and copy/save it in the exp subfolder)
    


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



    if 'gen_invs' in args.mode:

        print("icii9999")
        exit()


    if 'dump' in args.mode:

        # print(self.local(name))

        # # finally, read the saved exp aux.json (see above)
        # with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
        #     parameters = json.load(f)["parameters"]

        ###   r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI   ????????????????

 
        # 1) load aux json from exp_aux_json_folder
        if os.path.isfile(os.path.join(exp_aux_json_folder,"aux.json")):
            with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
                data = json.load(f)
                parameters = data["parameters"]




        if args.type == "vanilla":
            data['parameters']['A'] = 6000
        else:
            data['parameters']['A'] = len(all_actions_unique)

        if args.type == "vanilla":
            aaa = [2]
            aaa.extend(train_set[0][0].shape)
            data["input_shape"] = aaa

        else:
            aaa = [2]
            aaa.extend(train_set[0][0][0].shape)
            data["input_shape"] = aaa


        ### LE PROB ???? 
        #### FAUT QUE TU UTILISE LE MEME JSON QUE QUAND TU TRAINE 


        data['parameters']['N'] = 50
        parameters["N"] = 50
        print(args)
        print("HGTFRDEA!!!!!")
        # prob ici c'est que ça load 
        #args.dataset_folder is hanoi_complete_clean_faultless_withoutTI
        print("exp_aux_json_folder is {}".format(exp_aux_json_folder))
        print("dataset_aux_json_folder_exp is {}".format(dataset_aux_json_folder_exp))

        parameters["time_start"] = ""
        data['parameters']['time_start'] = ""

        # beta_z_and_beta_d
        parameters["epoch"] = 1
        data['parameters']['epoch'] = 1
        #parameters["A"] = 6000

        parameters["beta_ama_recons"] = 1
        parameters["beta_z_and_beta_d"] = [1, 1000]
        parameters["pdiff_z1z2_z0z3"] = [1, 1000]
        print("theparameters")
        print(parameters["A"])
        print(data['parameters']["A"])
        print("exp_aux_json_folderexp_aux_json_folder")
        print(exp_aux_json_folder)

        with open(os.path.join(exp_aux_json_folder,"aux.json"),"w") as f:
            json.dump(data, f, indent=4)



        if sys.argv[-1] == 'vanilla':
            net = latplan.modelVanilla.load(exp_aux_json_folder, allow_failure=False)
        else:
            net = latplan.model.load(exp_aux_json_folder, allow_failure=False)


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


    def train_fn(config=None):

        import wandb

        with wandb.init(config=config, group="Hanoi-4-4-HIGH_Lvl_Jaccard_N16_", resume=True):
            
            #path=path_to_json
            config = wandb.config

            parameters["jaccard_on"] = config.jaccard_on
            #parameters["denominator"] = config.denominator
            parameters["newloss_starting_epoch__AND__newloss_ending_epoch"] = config.newloss_starting_epoch__AND__newloss_ending_epoch

            parameters["epoch"] = 1200 #1500
            #parameters["time_start"] = args.hash

            data['parameters'] = parameters
            # # Step 3: Write the modified dictionary back to the JSON file
            # if os.path.isfile(os.path.join(path_to_json,"aux.json")):
            #     with open(os.path.join(path_to_json,"aux.json"),"w") as f:
            #         json.dump(data, f, indent=4)
            # là, ya model compilation ET training
            task = curry(nn_task, latplan.model.get(parameters["aeclass"]), path, train_set, train_set, val_set, val_set, parameters, False) 
            task()

    if 'learn' in args.mode:





        if args.action_id != "" and args.action_id != None:

            print("train_set")
            print(len(train_set[0]))

            #all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]
            train_set_one_action = []
            val_set_one_action = []

            for sample in train_set:
                if np.argmax(sample[-1]) == int(args.action_id):
                    sample[1] = np.array([1])
                    train_set_one_action.append(sample)
            
            for sample in val_set:
                if np.argmax(sample[-1]) == int(args.action_id):
                    sample[1] = np.array([1])
                    val_set_one_action.append(sample)


            train_set = train_set_one_action
            val_set = val_set_one_action



        import wandb
        
        wandb.login(key="2eec5f6bab880cdbda5c825881bbd45b4b3819d9")

        dataset_aux_json_folder_base = dataset_fold+"/"+sys.argv[2]
        dataset_aux_json_folder_exp = dataset_fold+"/"+sys.argv[2] + "/" + args.dataset_folder


        
        
        print(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json"))

        # 1) load aux json from dataset_aux_json_folder_base
        if os.path.isfile(os.path.join(dataset_aux_json_folder_base,"aux.json")):
            with open(os.path.join(dataset_aux_json_folder_base,"aux.json"),"r") as f:
                data = json.load(f)

        # 2) update some data using the loaded stuffs (img size, means etc, voir +  haut)
        # Step 2: Replace 'mean' and 'std' in the dictionary
        data['parameters']['mean'] = mean_all.tolist()
        data['parameters']['std'] = std_all.tolist()

        data['parameters']['orig_max'] = orig_max
        data['parameters']['orig_min'] = orig_min
        data["parameters"]["time_start"] = ""


        print("len(all_actions_unique)")
        print(len(all_actions_unique))
        if args.type == "vanilla":
            data['parameters']['A'] = 6000
        else:
            data['parameters']['A'] = len(all_actions_unique)


        if args.type == "vanilla":
            aaa = [2]
            aaa.extend(train_set[0][0].shape)
            data["input_shape"] = aaa

        else:
            aaa = [2]
            aaa.extend(train_set[0][0][0].shape)
            data["input_shape"] = aaa


        # action_id

        parameters["action_id"] = args.action_id

        parameters["epoch"] = 10000

        parameters["load_sae_weights"] = False
        
        parameters["use_wandb"] = True


        parameters["the_exp_path"] = exp_aux_json_folder
        # parameters["beta_z_and_beta_d"] = [10, 1000]
        # parameters["N"] = 300
        parameters["beta_z_and_beta_d"] = [1, 100]
        parameters["N"] = 50 #25
        data['parameters']["N"] = 50 #25
        # parameters["pdiff_z1z2_z0z3"] = 0
        parameters["type"] = args.type
        data['parameters']["type"] = args.type

        parameters["A"] = len(train_set[0][1])


        #parameters["weights_each_hl_action"] = weights_each_hl_action

        parameters["newloss_starting_epoch__AND__newloss_ending_epoch"] = [0, 700]

        parameters["jaccard_on"] = "both"

        parameters["use_temperature"] = False

        # 3) sauve la data updatée dans json dans le EXP     folder !!!!
        with open(os.path.join(exp_aux_json_folder,"aux.json"),"w") as f:
            json.dump(data, f, indent=4)





        ########################################  NORMAL TRAINING ########################################

        # import wandb
        # wandb.login(key="2eec5f6bab880cdbda5c825881bbd45b4b3819d9")
        parameters["use_wandb"] = True
        with wandb.init(project="my-Latplan", group="SinglerunsHANOI", name="R-Latplan-N50-HANOI-OneAction", resume=False):

            if args.type == "vanilla":
                task = curry(nn_task, latplan.modelVanilla.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
                task()
            else:

                task = curry(nn_task, latplan.model.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
                task()

    

        exit()

        ##################################### HYPER PARAMS SEARCH ########################################


        sweep_configuration = {
            #"method": "bayes",
            "method": "grid",
            "metric": {
                "goal": "minimize", 
                "name": "elbo"
            },
            "parameters": {
                'jaccard_on' : {"values": ["effects", "preconds", "both"]},
                #'denominator': {"values": [10, 2]},
                "newloss_starting_epoch__AND__newloss_ending_epoch" : {"values":
                [
                    [500, 700], [900, 1199]
                ],
                },

                # 'beta_d' : {"values": [ 100, 1000, 10000 ]},
                # 'beta_z' : {"values": [ 10, 100 ]},
                # "conv_channel" : {"values": [16, 32]},
                # "conv_kernel" : {"values": [5, 10]},
                # "conv_pooling" : {"values": [1, 2]},
                # "aae_width" : {"values": [500, 1000]},
                # "aae_depth" : {"values": [2, 4]},
                # "fc_depth" : {"values": [2, 3]},

            },
        }

        parameters["use_wandb"] = True
        parameters["epoch"] = 2000



        # for kk, vv in parameters.items():
        #     if is_jsonable(vv):
        #         data['parameters'][kk] = vv

        # # Step 3: Write the modified dictionary back to the JSON file
        # if os.path.isfile(os.path.join(path_to_json,"aux.json")):
        #     with open(os.path.join(path_to_json,"aux.json"),"w") as f:
        #         json.dump(data, f, indent=4)

        #
        sweep_id = wandb.sweep(sweep_configuration, project="my-Latplan")
        #sweep_id = "aymeric-b/my-Latplan/x3eyoaf5"
        wandb.agent(sweep_id, function=train_fn, count=6)

        exit()











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


