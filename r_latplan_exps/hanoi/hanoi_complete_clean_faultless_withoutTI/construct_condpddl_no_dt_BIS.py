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
import os
import sys
from itertools import combinations, islice
from multiprocessing import Pool, cpu_count, Array
from concurrent.futures import ProcessPoolExecutor
import time

from tqdm import tqdm



def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data


# from Negated Atom stuff
# returns add_smth

def friendly_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "(not (z"+str(integer)+"))"
    else:
        transformed_name += "(z"+str(integer)+")"
    return transformed_name


from typing import List

def intersection_of_lists(lists: List[List[str]]) -> List[str]:
    if not lists:
        return []
    
    # Start with the first list as the base set
    common_elements = set(lists[0])
    
    # Intersect with each subsequent list
    for lst in lists[1:]:
        common_elements.intersection_update(lst)
    
    return list(common_elements)

# # Example usage:
# lists = [[1, 2, 3, 4], [2, 3, 5], [2, 3, 6, 7]]
# result = intersection_of_lists(lists)
# print(result)  # Output: [2, 3]

def remove_from(to_update, to_remove):

    for i, ele in enumerate(to_update):
        if type(ele) is not list:
 
            if ele in to_remove:
                while ele in to_update:
                    to_update.remove(ele)
        else:
            remove_from(ele, to_remove)
            to_update = [ell for ell in to_update if ell]

    return to_update






from math import comb

def combinations_slice(args):
    """Worker function to generate a slice of combinations."""
    indices_U, r, start, end = args
    return list(islice(combinations(indices_U, r), start, end))

def GenerateCombinations(indices_U, maxSize):

    n = len(indices_U)
    total_combos = comb(n, maxSize)  # Total number of combinations
    max_workers = 64
    num_workers = min(cpu_count(), max_workers)
    chunk_size = total_combos // num_workers

    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = total_combos if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((indices_U, maxSize, start, end))

    results = []
    with Pool(processes=num_workers) as pool:
        for partial in tqdm(pool.imap_unordered(combinations_slice, tasks),
                            total=len(tasks),
                            desc="Generating combinations"):
            results.extend(partial)

    return results


def create_comb_row(index_comb):
    i, comb = index_comb
    row = np.zeros(n, dtype=int)
    row[list(comb)] = 1
    return i, row




########
#######



###### SelectBestCombination(candidates, targets)

######       ça calcul pour chaque candidat, sa couverture par rapport aux targets 
######
######
######          CAD : le nbre de targets pour lesquelles tous les bits à 1 dans le candidat SONT aussi à 1 dans la target



def SelectBestCombination(candidates, targets, final_set_):
    """
    Returns a 1D array of length num_candidates,
    where each value is the number of targets that entail that candidate.
    """

    num_candidates = candidates.shape[0]
    num_targets = targets.shape[0]

    candidates = candidates.astype(bool) 

    true_indices = np.array([])

    if len(final_set_) > 0:

        entails = np.full(len(candidates), False, dtype=bool)

        for ele in final_set_:

            matrix_ele = np.tile(ele, (num_candidates, 1)).astype(bool) 

            temp =  np.all(matrix_ele == candidates, axis=1)

            entails = np.logical_or(entails, temp)

        true_indices = np.where(entails)[0] # indices of the combinations candidates that we want to ignore


    # Expand dims for broadcasting: (num_targets, 1, 32) & (1, num_candidates, 32)
    # Resulting shape: (num_targets, num_candidates, 32)
    comparison = (targets[:, None, :] & candidates[None, :, :]) == candidates[None, :, :]

    # Now check for all bits matching across the last axis
    entailed_matrix = np.all(comparison, axis=2)  # shape: (num_targets, num_candidates)

    # Count for each candidate how many targets entail it
    count_per_candidate = np.sum(entailed_matrix, axis=0)  # shape: (num_candidates,)

    if len(true_indices) > 0:
        count_per_candidate[true_indices] = 0

    # sorted_arr = np.sort(count_per_candidate)[::-1]
    # print("sorted_arrsorted_arr")
    # print(sorted_arr[:5])
    # print(count_per_candidate[np.argmax(count_per_candidate)])

    if max(count_per_candidate) <= 0:
        return None

    else:
        return np.argmax(count_per_candidate)




# def SelectBestCombination(candidates, targets, fine_grained = False):
    
#     # BEST COMBI BORDEL 

#     #### MOUAIS C EST PAS BON CA POUR MOI , utilis el' a
#     #### Best combi c'eset cobui qu'est entailed par le plus grand nbre de targets FILS DE PUTE


#     candidates = candidates.astype(np.uint8)
#     targets = targets.astype(np.uint8)

#     # Broadcast candidates and targets
#     cand_exp = candidates[:, None, :]  # (35960, 1, 32)
#     targ_exp = targets[None, :, :]     # (1, 192, 32)

#     # Compute where candidate i "violates" target j
#     coverage_mask = (cand_exp & ~targ_exp) == 0  # (35960, 192, 32)
#     covered = coverage_mask.all(axis=2)          # (35960, 192)

#     # Mask out candidates that are all zeros
#     non_zero_mask = candidates.any(axis=1)       # (35960,)
#     covered[~non_zero_mask] = 0

#     if fine_grained:
#         # Compute number of active features in each candidate
#         candidate_weights = candidates.sum(axis=1)   # (35960,)
#         # Broadcast weights to match covered matrix shape
#         weighted_coverage = covered * candidate_weights[:, None]  # (35960, 192)
#         covered = weighted_coverage

#     # Total number of targets covered by each candidate
#     total_coverage = covered.sum(axis=1)  # (35960,)

#     # overall_coverage = covered.sum()
#     # overall_weighted_coverage = covered.sum()

#     return np.argmax(total_coverage), total_coverage[np.argmax(total_coverage)]













def find_entailed_candidates_and_check_disjunction(candidates, targets):

    candidates = candidates.astype(np.uint8)
    targets = targets.astype(np.uint8)

    # print("in find_entailed_candidates_and_check_disjunction")

    # print("targetstargets")
    # print(targets)

    # print(targets[2])
    # print(targets[4])

    # print("candidatescandidates")
    # print(candidates)


    # [0 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0]
    # [0 0 1 0 1 0 0 0 0 1 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0]


    # [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]

    # [0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]


    # # print(candidates.shape)
    # new_row = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    # candidates = np.vstack([candidates, new_row])

    # print("candidates after")
    # candidates = np.empty((0, 32), dtype=int)

    # # Loop to create and stack each row
    # for i in range(32):
    #     new_row = np.zeros((1, 32), dtype=int)
    #     new_row[0, i] = 1  # Set the 1 at the ith position
    #     candidates = np.vstack([candidates, new_row])


    summ = 0

    for i, target in enumerate(targets):




        # New condition: candidate & target == candidate
        entailed_mask = np.all((candidates & target) == candidates, axis=1)
        entailed_candidates = candidates[entailed_mask]

        if len(entailed_candidates) == 0:
            entails_back = False
        else:
            # Disjunction over entailed candidates
            disjunction = np.bitwise_or.reduce(entailed_candidates, axis=0)
            # Check if this disjunction entails the target
            entails_back = np.all((disjunction & target) == target)

        # results.append({
        #     'target_index': i,
        #     'entailed_candidate_indices': np.where(entailed_mask)[0],
        #     'entails_back': entails_back
        # })

        if entails_back:
            summ += 1


    # print("SUMMM is ")
    # print(summ)
    # exit()

    return summ


def checkfordupplicates(arr):


    # Convert each row to a tuple so we can use numpy's unique function
    _, idx, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)

    # Find duplicates (count > 1)
    duplicate_rows = arr[np.where(counts > 1)]
    has_duplicates = len(duplicate_rows) > 0

    
    if has_duplicates:
        

        print("Duplicate rows:")
        print(duplicate_rows)
        exit()

    else:
        print("no dupplicates")
    return


def TotalCoverage(candidates, targets):



    # candidates = np.array(
    #     [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 1., 1., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
    #     0., 0., 0., 1., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 1.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 1., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 1., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #     0., 0., 0., 0., 0., 0., 0., 0.]]
        
    # )


    if len(candidates) == 0:
        # print("CANTTT  BE HERE")
        # exit()
        return 0


    num_of_covered_targets = find_entailed_candidates_and_check_disjunction(candidates, targets)

    # print("targets ")
    # print(targets)

    # print("num_of_covered_targetsnum_of_covered_targetsnum_of_covered_targets")
    # print(num_of_covered_targets)
    # exit()

    return num_of_covered_targets # / len(targets)








# # candidates: final_set
# # targets: binary_S
# # Return the percentage of
# #  preconditions (binary_S stuffs)
# #       that "are covered" by the disjunction of candidates
# #           i.e. the number of targets (in targets) for which : all indices where there is a one, then 
# #
# #                   for this same indices, there is a one also in the conjunction of candidates (the super_candidate)


# def TotalCoverage(candidates, targets):
    
#     # HERE "candidates" IS THE final_set, i.e. set of all groups in the "OR"
#     candidates = candidates.astype(np.uint8)
#     targets = targets.astype(np.uint8)

#     # if len(candidates) == 0:
#     #     return 0

#     # Step 1: Combine all candidates using bitwise OR (disjunction)
#     # super_candidate = np.bitwise_or.reduce(candidates, axis=0)  # shape: (32,)

#     super_candidate = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    
#     #super_candidate = np.ones((32,)).astype(np.uint8)


#     print("super_candidatesuper_candidatesuper_candidatesuper_candidate")
#     print(super_candidate)


#     # targets[0] = targets[0]*0
#     # targets[0][0] = 1
#     print()
#     print("targets")
#     print(targets)
#     # Step 2: Compute coverage of super_candidate with each target
#     # Coverage: all bits set in super_candidate must also be set in target

#     print("TYYYYYYYYYYYYYYPES")
#     targets = targets.astype(bool)
#     super_candidate = super_candidate.astype(bool)

#     mask = (targets & ~super_candidate).sum(axis=1) == 0
#     num_entailed = np.sum(mask)
#     print("num_entailednum_entailednum_entailednum_entailed")
#     print(num_entailed)

#     exit()
  

#     ### for all indices where value is 1 in targets example THEN in super_candidates (for these same indices) the value MUST ALSO BE ONE



#     return (num_entailed / len(targets)) * 100










# exit()

# def friendly_name(literal):       
#     integer = int(''.join(x for x in str(literal) if x.isdigit()))
#     transformed_name = ""
#     if "Negated" in str(literal):
#         transformed_name += "del_"+str(integer)
#     else:
#         transformed_name += "add_"+str(integer)
#     return transformed_name


#############       GROUP THE TRANS IDS PER HIGH LVL ACTION      ############# 

path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"

loaded_data = load_dataset(path_to_dataset) # load dataset for the specific experiment
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

dico_lowlvl_highlvl = {} # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION

for ii, ele in enumerate(train_set_no_dupp): # loop over the train set (without dupplicate) # AND group the transitions into their respective High Level Actions
    if np.argmax(ele[1]) not in dico_lowlvl_highlvl:
        dico_lowlvl_highlvl[np.argmax(ele[1])] = np.argmax(ele[2])






# 1) construct the dico: key=lowlvlid , value=highlvlid         DONE (dico_lowlvl_highlvl)


# 2) constriuct the dico: key=highlvlid, value=dico where each ele
#       is a dico with key=lowlvlid and value is a list with 2 eles (preconds , effects)
#           

dico_highlvlid_lowlvlactions = {}



domainfile="domain_ORIGINAL_NO_DUPP.pddl"
problemfile="problem.pddl"
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
sys.path.insert(0, translate_path)

try:
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars
    task = pddl_pars.open(
    domain_filename=domainfile, task_filename=problemfile) # options.task

    # print(task.indices_actions_no_effects)
    # exit()
    nb_total_actions = len(task.actions) + len(task.indices_actions_no_effects) # including actions with no effects

    cc_normal_actions = 0

    for trans_id in range(nb_total_actions):

        if trans_id in task.indices_actions_no_effects:
            pass 

        else:

            
            act = task.actions[cc_normal_actions]

            highlvlid = dico_lowlvl_highlvl[trans_id]

            if highlvlid not in dico_highlvlid_lowlvlactions:
                dico_highlvlid_lowlvlactions[highlvlid] = {}

            if trans_id not in dico_highlvlid_lowlvlactions[highlvlid]:
                dico_highlvlid_lowlvlactions[highlvlid][trans_id] = {"preconds": [], "effects": []}


            for precond in list(act.precondition.parts):
                f_name_precond = friendly_name(precond)
                dico_highlvlid_lowlvlactions[highlvlid][trans_id]["preconds"].append(f_name_precond)

            for eff in act.effects:
                f_name_eff = friendly_name(eff.literal)
                dico_highlvlid_lowlvlactions[highlvlid][trans_id]["effects"].append(f_name_eff)

            # friendly_name
            cc_normal_actions += 1


finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)



print(dico_highlvlid_lowlvlactions[0][19])



##############################################################################
###########################  WRITE THE PDDL   ################################
#######################  and LEARN THE SECOND TREE ###########################
##############################################################################
with open("domainCondBIS-NODT-BIS.pddl", "w") as f:


    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(16):
        f.write("(z"+str(i)+" )\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")


    atoms = []
    for i in range(16):
        atoms.append(f"(z{i})")
    for i in range(16):
        atoms.append(f"(not (z{i}))")

    # print(atoms)
    # exit()

    dico_disjunctions_preconds_dt = {}

    for num_action in range(0, 22):

        


        dico_disjunctions_preconds_dt[num_action] = []
 
        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")

        # f.write("   :precondition ()\n") # IF NOT PRECONDITIONS

        precondition_part = "" 
        intersection_part = ""
        or_clause = ""


        finat_set = np.array([])
        S = []
        maxSize_in_S = 0
        U = []
        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():
            S.append(dico_vals["preconds"])
            if len(dico_vals["preconds"]) > maxSize_in_S:
                maxSize_in_S = len(dico_vals["preconds"])
            for atom in dico_vals["preconds"]:
                U.append(atom)
        U = list(set(U))

   


        # 1 -- taking the intersection of all the preconditions set of the high lvl action
        I = intersection_of_lists(S)

        # 3-4 --
        for s in S:
            for i in I:
                if i in s:
                    s.remove(i)

        binary_S = np.zeros((len(S), 32), dtype=np.uint8)
        binary_S = pd.DataFrame(binary_S)
        binary_S.columns = atoms
        for idx, row in binary_S.iterrows():
            for col in S[idx]:
                if col in binary_S.columns:
                    binary_S.at[idx, col] = 1


        binary_S = binary_S.to_numpy()



        
        # 0) if there is an intersection we build an AND clause

        if len(I) > 0:
            for inter in I:
                intersection_part += inter + " "


        indices_U = [i for i in range(len(atoms)) if atoms[i] in U]

        maxSize_in_S = 8 #7
        # 

        final_set = np.empty((0, 32))  #

        binary_S_copy = binary_S.copy()


        # final_set, binary_S_copy

        print("final_set")
        print(final_set)
        print()
        print("binary_S_copy")
        print(binary_S_copy.shape) # (192, 32)

        thecounter = 0

        while True:

            print("final_set SENT is")
            print(final_set)
            print("thecounter is {}".format(str(thecounter)))
            # TotalCoverage ( candidates, targets )
            #   TotalCoverage ( final_set, binary_S )
            #
            # print()
            # print("binary_S_copy")
            # print(binary_S_copy)
            # np.savetxt("binary_S_copy.txt", binary_S_copy, fmt='%d')
            # print()
            coverage = TotalCoverage(final_set, binary_S_copy)

            # if "thecounter" == 1:
            #     exit()




            print(f"Current coverage: {coverage}")
            if coverage >= 100:
                break


            max_ones = np.max(np.sum(binary_S, axis=1))

            maxSize_in_S = max(1, min(maxSize_in_S - 1, max_ones))

            print("maxSize_in_S is {}".format(str(maxSize_in_S)))

            # maxSize_in_S part d'une valeur puis

            ####  maxSize_in_S = maxSize_in_S - 1

            ######   ou max_ones

            #### max(1, )


            last_current_time = time.time()
            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 0", str(time_spent))
            last_current_time = time.time()
    
            # print("maxSize_in_S is {}".format(str(maxSize_in_S))) # 6

            # print("len indices_U {}".format(len(indices_U))) # 32

            # 6 parmis 32 = 906192
            combinationsIndices = GenerateCombinations(indices_U, maxSize_in_S)

            # print("len combinationsIndices")
            # print(len(combinationsIndices))

            # print(combinationsIndices[:10])

            # print()

            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 1", str(time_spent))
            last_current_time = time.time()


            n_features = len(indices_U)


            matrix_shape = (len(combinationsIndices), n_features)

            # Shared memory array
            shared_array = Array('i', matrix_shape[0] * matrix_shape[1], lock=False)
            combinationsMatrix = np.frombuffer(shared_array, dtype='int32').reshape(matrix_shape)

            def worker(args):
                idx, comb = args
                for j in comb:
                    combinationsMatrix[idx, j] = 1
                return idx  # Just to track progress


            with Pool(processes=64) as pool:
                # Wrap the iterable with tqdm for progress bar
                for _ in tqdm(pool.imap(worker, enumerate(combinationsIndices)), total=len(combinationsIndices)):
                    pass  # tqdm handles progress tracking


            # print("laaaa")
            # print(combinationsIndices[:5])
            # print()
            # print(combinationsMatrix[:5])

            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 2", str(time_spent))
            last_current_time = time.time()

        
            best_combi_index = SelectBestCombination(combinationsMatrix, binary_S, final_set)            



            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 3", str(time_spent))
            last_current_time = time.time()



            # print(size_covered)
            best = combinationsMatrix[best_combi_index]
            # print("best")
            # print(best)
            # print("binary_Sbinary_S")
            # print(binary_S)


            ### TOU

            # def a entails b, a ^ b = a 

            counter_entailments = 0

            indices = []

            for iii, s in enumerate(binary_S):
                # print("s is ")
                # print(s)
                # print("best is ")
                # print(best)
                #if np.all((s & best) == s):
                if np.array_equal(best & s, best):
                    counter_entailments += 1
                    indices.append(iii)




            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 4", str(time_spent))
            last_current_time = time.time()




            # print("counter_entailments")
            # print(counter_entailments)


            # print("indicesindicesindicesindices")
            # print(indices)

            # [2, 4, 6, 16, 22, 23, 25, 26, 33, 35, 37, 39, 41, 44, 47, 48, 49, 54, 58, 59, 60, 66, 67, 69, 70, 72, 73, 74, 76, 82, 84, 87, 92, 95, 112, 113, 115, 116, 118, 120, 130, 132, 135, 138, 145, 146, 147, 151, 161, 166, 171, 175, 177, 179, 191]


            checkfordupplicates(final_set)



            time_spent =  time.time() - last_current_time
            print("TIME SPENT SO FAR 5", str(time_spent))
            last_current_time = time.time()




            final_set = np.vstack([final_set, best])

            # --- l 11-13

            # best = np.array([0,0,0,1])

            # binary_S = np.array([
            #     [1,0,1,1],
            #     [1,0,0,0],
            #     [0,0,1,1]
            # ])
        
            matrix_best = np.tile(best, (len(binary_S), 1))

            # print("matrix_best IS ")
            # print(matrix_best)
            entails = np.all((binary_S & matrix_best) == matrix_best, axis=1)
            # print("entails is ")
            # print(entails)

            binary_S[entails] = binary_S[entails] - matrix_best[entails]

            # print("VBINARY S MODIFIED ")
            # print(binary_S)


            #### 1) indices of entailments

            ### 2) where best = truc, only for indices of entailements 

            ###             zero out

            #best_matrix = 



            # Zero out positions in examples that entail best, but only where best is True
        

            thecounter += 1


        # time_spent =  time.time() - last_current_time
        # print("TIME SPENT SO FAR 3", str(time_spent))
        # last_current_time = time.time()


        # [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]

        exit()


        #     


        # C is the candidate

        # s is the "target"

        #  s |= C if whatever is in 

        # I have two binary numpy array, the candidates of shape (35960, 32) which hold values for 35960 candidates
        # about 32 binary features

        # and the "target" of shape (192, 32) which holds values for 192 "targets" about 32 binary features

        # I want to compute, for each candidate, its coverage with regard to the targets, i.e. 


        #       candidate c_i covers target t_i iff for all the features that are true in c_i then they are

        #       also true in t_i  , in such case the coverage is 1, otherwise it is 0. Then, I want the total 

        #           coverage regarding the targets, which is the sum of each individual coverage

        #
        #
        #

        #    OTHER VERSION: 

        #
        #
        #           now I want to add another constraint for c_i to cover t_i, namely, instead of adding +1
        #
        #               I want to add +Number_elements_of_c_i

        exit()

        # def GenerateCombinations(indices_U, maxSize):
        #     return combinations(indices_U, maxSize)  # This is a generator

        # #indices_U = [0, 1, 2, 3]
        # maxSize = 16
        # counter = 0
        # for comb in GenerateCombinations(indices_U, maxSize):
        #     # process comb without storing all of them
        #     print(comb)
        #     counter += 1
        #     if counter > 13:
        #         break
        #     pass

        # exit()
        # each comb is a set of indices

        # 


        # exit()
   

        # # exit()


        # while not TotalCoverage(binary_S, finat_set):

        #     # 1) remove from S and from remaining_atoms whatever atom that is in Best (or in intersection)
        #     print("S BEFORE")
        #     print(len(S))

        #     S = remove_from(S, I)
      

        #     print("S AFTER")


        #     # 2) take the max size among all sets in S and construct all possible combinations of sets of this max size, using the atoms in remaining_atoms

        #     max_size = 0    
        #     for s in S:
        #         if len(s) > max_size:
        #             max_size = len(s)

        #     print(max_size)


            

        #     exit()


        #     # 3)





        # CASE : precond is an OR of the set of all high lvl preconds 
        f.write("   :precondition (OR ")

        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():

            print("FFFFFFF")
            print(dico_vals["preconds"])
            exit()
            two_tabs_space  = "         "
            tmp_str = ""

            if len(dico_vals["preconds"]) > 0:
                
                #tmp_str += two_tabs_space+str(lowlvlkey)+" (when "
                #tmp_str += two_tabs_space

                #tmp_str += two_tabs_space+str(lowlvlkey)+" (when "

                tmp_str += "(and "  
                ### adding the preconds            
                for pre in dico_vals["preconds"]:
                    tmp_str += " "+pre

                tmp_str += ")\n"  


                # tmp_str += ")\n"  

                f.write(tmp_str+" ") 
                

        f.write(")\n") # closing the OR




        f.write("   :effect (and\n")

        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():


            two_tabs_space  = "         "
            tmp_str = ""

            if len(dico_vals["preconds"]) > 0:

                #tmp_str += two_tabs_space+str(lowlvlkey)+" (when "
                tmp_str += two_tabs_space+" (when "

                tmp_str += "(and "  
                ### adding the preconds            
                for pre in dico_vals["preconds"]:
                    tmp_str += " "+pre

                tmp_str += ")\n"  

                

            tmp_str += two_tabs_space
            # adding the effects
            
            if len(dico_vals["effects"]) == 1:
                tmp_str += " "+eff
            
            elif len(dico_vals["effects"]) > 1: 
                tmp_str += "(and "
                for eff in dico_vals["effects"]:
                    tmp_str += " "+eff
                tmp_str += ")"
            if len(dico_vals["preconds"]) > 0:
                tmp_str += ")\n"   


            f.write(tmp_str)

        f.write("   )\n")
        f.write(")\n")

    f.write(")\n")