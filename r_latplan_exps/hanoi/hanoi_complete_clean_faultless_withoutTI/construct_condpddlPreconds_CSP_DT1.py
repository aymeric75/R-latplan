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
from collections import defaultdict
import os
import sys
from itertools import combinations, islice
from multiprocessing import Pool, cpu_count, Array
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import copy
from tqdm import tqdm


base_dir = "/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/"


filter_out_dt1 = False 
factorize_dt1 = False

precondition_only_entailing = True # tells if we consider, in the precondition clause, only the sets that entail other sets
filter_out_condEffects = True # tells if we filter out from the preconditions of the conditional effects, the atoms that are already
# present in the precondition clause


# False False
# False True
# True False
# True True

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

def format_precond(precond, reverse=False):
    splits = precond.split('_')
    if reverse:
        if int(precond[-1]) == 0:
            return '('+str(splits[0])+str(splits[1])+')'
        elif int(precond[-1]) == 1:
            return '(not ('+str(splits[0])+str(splits[1])+'))'
    else:
        if int(precond[-1]) == 0:
            return '(not ('+str(splits[0])+str(splits[1])+'))'
        elif int(precond[-1]) == 1:
            return '('+str(splits[0])+str(splits[1])+')'
            

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




def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            sample_count = tree_.n_node_samples[node]  # Get the number of samples at this leaf
            class_label = np.argmax(tree_.value[node])
            path += [f"class is: {class_label}, samples: {sample_count}"]
            paths += [path]
            
    recurse(0, path, paths)
    return paths


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


def GenerateCombinationsRange(indices_U, maxSize):
    max_workers = 64
    num_workers = min(cpu_count(), max_workers)
    tasks = []

    #for r in range(1, maxSize + 1):
    for r in range(1, maxSize + 1):
        total_combos = comb(len(indices_U), r)
        chunk_size = total_combos // num_workers

        for i in range(num_workers):
            start = i * chunk_size
            end = total_combos if i == num_workers - 1 else (i + 1) * chunk_size
            tasks.append((indices_U, r, start, end))

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






def factorize_dict(input_dict, num_action=None):



    effects_for_all = []
    for keyy, listes in input_dict.items():
        if len(listes) == 1:
            if len(listes[0]) == 1:
                if isinstance(listes[0][0], int):
                    effects_for_all.append(keyy)



    list_to_keys = defaultdict(set)
    new_dict = defaultdict(list)
    
    # Reverse mapping from lists to keys
    for key, lists in input_dict.items():
        # if not lists:
        #     new_dict[(key,)] = []  # Preserve empty lists with single key as tuple

        if key not in effects_for_all:
            for lst in lists:
                tuple_lst = tuple(sorted(lst[:-1]))  # Convert list to tuple for immutability and sorting for consistency
                # print("tuple_lsttuple_lsttuple_lst")
                # print(tuple_lst)
                # print(key)
                # exit()
                list_to_keys[tuple_lst].add(key)
        
                for sup in effects_for_all:
                    list_to_keys[tuple_lst].add(sup)

    # Construct the new factorized dictionary
    processed_keys = set()
    # pour chaque combi de preconds (lst) abd corresponding effects (keys)
    #
    for lst, keys in list_to_keys.items():
        keys_tuple = tuple(sorted(keys))
        new_dict[keys_tuple].append(list(lst))
        processed_keys.update(keys)
    
    # Convert keys tuples to appropriate format
    factorized_dict = {}
    for keys, lists in new_dict.items():
        factorized_dict[tuple(keys)] = lists  # Ensure all keys are stored as tuples

    return factorized_dict



########
#######



###### nberOfEntailingTargets(candidates, targets)

######       ça calcul pour chaque candidat, sa couverture par rapport aux targets 
######
######
######          CAD : le nbre de targets pour lesquelles tous les bits à 1 dans le candidat SONT aussi à 1 dans la target



def transfo_precond_effect_name_to_pddl(eff_name):

    if eff_name.split('_')[0] == 'del':
        eff_name = "(not (z"+eff_name.split('_')[1]+"))"
    else:
        eff_name = "(z"+eff_name.split('_')[1]+")"
    return eff_name

# list like [['(z11)', '(z13)', '(z3)'], ['(not (z11))', '(not (z6))', '(z13)', '(z3)']]
def logic_factorization(list_of_clauses):

    # Convert lists to sets for easier set operations
    clause_sets = [set(clause) for clause in list_of_clauses]

    # print()
    # print("clause_sets")
    # print(clause_sets)

    # [
    #     {'(z3)', '(z13)', '(z11)'}, 
    #     {'(z3)', '(z13)', '(not (z6))', '(not (z11))'}
    # ]
    common_elements_str = ""
    not_common_elements_str = ""
    # Step 1: Find common elements across all clauses (AND factorization)
    common_elements = set.intersection(*clause_sets)

    #### LE TRAITEMENT CI DESSOUS TU LE FAIS A LA FIN
    # si common elements ==> (AND comon_eles not_comon_eles)   !!!!!!!!!!!!!!
    # sinon             ===> not_comon_eles
    
    # POUR L INSTANT CALCUL SEPARE DE chacun (common_elements et not_comon_eles)
    if len(common_elements) > 0:
        if len(common_elements) == 1:
            common_elements_str += list(common_elements)[0]

        if len(common_elements) > 1:
            common_elements_str += "(AND "
            for ele in common_elements:
                common_elements_str += ele + " "
            common_elements_str += ")"
    # print()
    # print("common_elements")
    # print(common_elements)
    # print()
    # print("common_elements_str")
    # print(common_elements_str)

    # Step 2: Identify remaining unique elements for each clause
    unique_parts = [clause - common_elements for clause in clause_sets]

    unique_parts = [s for s in unique_parts if s]  # Removes empty sets


    # (AND common_eles (OR ele1diff ele2diff2))
    if len(unique_parts) > 0:
        if len(unique_parts) == 1:

            not_common_elements_str += list(unique_parts[0])[0]

        if len(unique_parts) > 1:
            
            # 
            not_common_elements_str += " (OR "


            for ele in list(unique_parts):
                
                if len(ele) == 1:
                    
                    # if list(ele)[0] == '(not (z11))':
                    #     print(unique_parts)
                    #     print("GGGGGGG")
                    #     exit()

                    not_common_elements_str += list(ele)[0] + " "
                elif len(ele) > 1:
                    not_common_elements_str += "(AND "
                    for e in list(ele):
                        not_common_elements_str += e + " "
                    not_common_elements_str += ")"



            not_common_elements_str += ")"
    
    # print()
    # print("not_common_elements_str")
    # print(not_common_elements_str)
    # exit()

    
    #### LE TRAITEMENT CI DESSOUS TU LE FAIS A LA FIN
    # si common elements ==> (AND comon_eles not_comon_eles)   !!!!!!!!!!!!!!
    # sinon             ===> not_comon_eles
    retour = ""
    if len(common_elements) > 0:
        retour += "(AND "
        retour += common_elements_str
        retour += not_common_elements_str
        retour += ")"
    else:
        retour += not_common_elements_str

    return retour

# import numpy as np
# import multiprocessing as mp

# check which targets entail a candidate
#
#
#           1) besoin du résultat d'entailed d'avant 
#
#
#           pour chaque candidat, on a besoin de savoir 1) quelles targets l'entail ou pas 
#
#                                                       2) les indices des targets qui on été modifié (ça c'est
#                                                           plus général)
#               

# # Initialize count_per_candidate
# count_per_candidate = np.zeros(num_candidates, dtype=int)

# for idx, count in results:
#     if fine_grained:
#         count_per_candidate[idx] = count * candidates_sum_of_atoms[idx]
#     else:
#         count_per_candidate[idx] = count

# return count_per_candidate






def _check_entailment(args):
    candidate_idx, candidate, EntailingTargetsForThisCandidate , indices_of_targets_modified_previous_iter_, targets_modified_previous_iter_ = args


    ##
    ##  EntailingTargetsForThisCandidate des 1 et des 0s (192  1s or 0s)
    ## 
    ##
    ##

    candidate = candidate.astype(bool)
    # print("!!!! BORDEL !!!!!")
    # print(targets.shape) # (192, 32)
    # print(candidate.shape) # (32,)
    #comparison = (targets & candidate) == candidate

    # indices_of_targets_modified_previous_iter_

    #### targets[indices_of_targets_modified_previous_iter_] tu le donne direct en entree
    ####  targets[indices_of_targets_modified_previous_iter_]

    entailement_of_specific_targets = (targets_modified_previous_iter_ & candidate) == candidate

    # 1) comparison seuelement avec un subset des targets, ie qui ont été modifiés
    #                   ie targets[mask_of_targets_modified_previous_iter_] 
    ##""                    vois quelle shape ressort de ce comparison
    ###"                        mais sinon utilise plutot les indices
    ###   2) 

    # print(comparison.shape) # (192, 32) : 
    # print(comparison[0])
    #entailment = np.all(comparison, axis=1)

    entailment = EntailingTargetsForThisCandidate

    if len(indices_of_targets_modified_previous_iter_) > 0:
        # print("entailement_of_specific_targetsentailement_of_specific_targetsentailement_of_specific_targets")
        # print(entailement_of_specific_targets)
        # print(np.all(entailement_of_specific_targets, axis=1))
        entailment[indices_of_targets_modified_previous_iter_] = np.all(entailement_of_specific_targets, axis=1) # entailement_of_specific_targets
    
    
    
    
    # PUIS change 
    #print(entailment.shape) # (192,) 
    #exit()
    count = np.sum(entailment)
    return candidate_idx, entailment, count

def _check_final_set(args):
    ele, candidates = args
    matrix_ele = np.tile(ele, (candidates.shape[0], 1)).astype(bool)
    temp = np.all(matrix_ele == candidates, axis=1)
    return temp



# def nberOfEntailingTargets(candidates, targets, final_set_, fine_grained = False):

#     candidates_sum_of_atoms = np.array([])
#     if fine_grained:
#         candidates_sum_of_atoms = np.sum(candidates, axis = 1)

#     num_candidates = candidates.shape[0]
#     candidates = candidates.astype(bool)
#     targets = targets.astype(bool)

#     true_indices = set()
#     if len(final_set_) > 0:
#         with mp.Pool(processes=64) as pool:
#             entail_results = pool.map(_check_final_set, [(ele, candidates) for ele in final_set_])

#         entails = np.any(entail_results, axis=0)
#         true_indices = set(np.where(entails)[0])

#     # Prepare arguments for multiprocessing
#     args = [(i, candidates[i], targets) for i in range(num_candidates) if i not in true_indices]

#     with mp.Pool(processes=64) as pool:
#         results = pool.map(_check_entailment, args)

#     # Initialize count_per_candidate
#     count_per_candidate = np.zeros(num_candidates, dtype=int)


#     for idx, count in results:
#         if fine_grained:
#             count_per_candidate[idx] = count*candidates_sum_of_atoms[idx]
#         else:
#             count_per_candidate[idx] = count

#     return count_per_candidate


# combinationsMatrix, binary_S, final_set

def nberOfEntailingTargets(candidates, targets, final_set_, EntailingTargetsPerCombi_, indices_of_targets_modified_previous_iter_, fine_grained=False):

    # EntailingTargetsPerCombi_
    # mask_of_targets_modified_previous_iter_

    candidates_sum_of_atoms = np.array([])
    if fine_grained:
        candidates_sum_of_atoms = np.sum(candidates, axis=1)

    num_candidates = candidates.shape[0]
    candidates = candidates.astype(bool)
    targets = targets.astype(bool)

    # EntailingTargetsPerCombi_

    ## q

    true_indices = set()
    if len(final_set_) > 0:
        with mp.Pool(processes=64) as pool:
            entail_args = [(ele, candidates) for ele in final_set_]
            entail_results = list(tqdm(pool.imap(_check_final_set, entail_args), total=len(entail_args), desc="Checking final set"))

        entails = np.any(entail_results, axis=0)
        true_indices = set(np.where(entails)[0]) # indices of the candidates that already are included in the final_set

    # Prepare arguments for multiprocessing
    # 
    # EntailingTargetsPerCombi_[i] , 1D binary array of shape (192,) that tells which targets entail this candidate 
    # indices_of_targets_modified_previous_iter_
    # targets
    # 

    targets_modified_previous_iter = targets[indices_of_targets_modified_previous_iter_]
    args = [(i, candidates[i], EntailingTargetsPerCombi_[i], indices_of_targets_modified_previous_iter_, targets_modified_previous_iter) for i in range(num_candidates) if i not in true_indices]

    with mp.Pool(processes=64) as pool:
        results = list(tqdm(pool.imap(_check_entailment, args), total=len(args), desc="Checking entailment"))

    # Initialize count_per_candidate
    count_per_candidate = np.zeros(num_candidates, dtype=int)

    entailment_per_candidate = np.zeros((num_candidates, len(targets)), dtype=int)

    # there are 
    for idx, entailment, count in results:
        if fine_grained:
            count_per_candidate[idx] = count * candidates_sum_of_atoms[idx]
        else:
            count_per_candidate[idx] = count

        entailment_per_candidate[idx] = entailment

    return count_per_candidate, entailment_per_candidate



# sorted_arr = np.sort(count_per_candidate)[::-1]
# print("sorted_arrsorted_arr")
# print(sorted_arr[:5])
# print(count_per_candidate[np.argmax(count_per_candidate)])




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
        entailed_mask = np.all((candidates & target) == candidates, axis=1) # mask that tells which candidates are entailed by the target
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



preconds_perEff_perAction = {}

#############       GROUP THE TRANS IDS PER HIGH LVL ACTION      ############# 

path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"

loaded_data = load_dataset(path_to_dataset) # load dataset for the specific experiment
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

dico_lowlvl_highlvl = {} # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION

for ii, ele in enumerate(train_set_no_dupp): # loop over the train set (without dupplicate) # AND group the transitions into their respective High Level Actions
    if np.argmax(ele[1]) not in dico_lowlvl_highlvl:
        dico_lowlvl_highlvl[np.argmax(ele[1])] = np.argmax(ele[2])


dico_transitions_per_high_lvl_actions = {} # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION

for ii, ele in enumerate(train_set_no_dupp): # loop over the train set (without dupplicate) # AND group the transitions into their respective High Level Actions
    if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}
    if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])][np.argmax(ele[1])] = {
            "preprocessed" : ele[0],
        }


pos_preconds = np.loadtxt(base_dir+"/pos_preconds_aligned.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt(base_dir+"/neg_preconds_aligned.csv", delimiter=' ', dtype=int)

add_effs = np.loadtxt(base_dir+"/add_effs_aligned.csv", delimiter=' ', dtype=int)
del_effs = np.loadtxt(base_dir+"/del_effs_aligned.csv", delimiter=' ', dtype=int)

effects_set = []
for i in range(16):
    effects_set.append(f"add_{i}")
for i in range(16):
    effects_set.append(f"del_{i}")
preconds_perEff_perAction = {}

feature_names = []
for i in range(16):
    feature_names.append(f"(z_{i}_1)")
    feature_names.append(f"(z_{i}_0)")

preconds_names = []
for f in feature_names:
    preconds_names.append(format_precond(f.replace("(", "").replace(")", "")))


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






##############################################################################
################  LOOP OVER EACH ACTION and learn the First tree #############
##############################################################################

for num_action in range(0, 22):



    if not os.path.exists("actions/action_"+str(num_action)):
        os.makedirs("actions/action_"+str(num_action))

    

    preconds_perEff_perAction[num_action] = {}

    ##################################################################################
    # Construct the X
    action_transitions_preconds = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        action_transitions_preconds.append(preconds_union)
    X = np.array(action_transitions_preconds)



    X = pd.DataFrame(X)
    new_columns = []
    for i in range(16):
        new_columns.append(f"z_{i}")
    for i in range(16):
        new_columns.append(f"not(z_{i})")
    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns
    X.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())
    indices_with_nine = X[X.isin([9]).any(axis=1)].index
    X = X.drop(index=indices_with_nine)


    # Create a new DataFrame for the modified X matrix
    new_X = pd.DataFrame()
    for i in range(16):
        # Create a new column in new_X
        new_column = []
        for index, row in X.iterrows():
            if row[f'z_{i}'] == 1:
                new_column.append(1)
            elif row[f'not(z_{i})'] == 1:
                new_column.append(0)
            else:
                new_column.append('?')  # Or any other representation you prefer for "otherwise"
        new_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame
    new_X.replace(1, "1", inplace=True)
    new_X.replace(0, "0", inplace=True)
    


    onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity
    onehot_encoded = onehot_encoder.fit_transform(new_X) # Fit and transform the data
    new_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))
    new_X.index = X.index
    
    # removing the "?" columns
    new_X = new_X.loc[:, ~new_X.columns.str.contains(r'\?')]

    # END Construct the X
    ##################################################################################



    ##################################################################################
    # Construct the Y
    action_transitions_effects = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        effects_union = []
        effects_union.extend(add_effs[trans_id])
        effects_union.extend(del_effs[trans_id])
        action_transitions_effects.append(effects_union)
    Y = np.array(action_transitions_effects)
    Y = pd.DataFrame(Y)
    new_columns = []
    for i in range(16):
        new_columns.append(f"add_{i}")
    for i in range(16):
        new_columns.append(f"del_{i}")
    Y.columns = new_columns[:len(Y.columns)] # Use slicing to handle cases where X might have fewer columns
    Y.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())
    Y = Y.drop(index=indices_with_nine)


    # print(new_X.index)
    # print(len(new_X))
    # print()
    # print(Y.index)
    # print(len(Y))
    # exit()

    # END Construct the Y
    ##################################################################################



    ##################################################################################
    # Train the first classifier
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.2, random_state=42)
    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)




    ##################################################################################
    # Print or/and return the DT branches (FIRST DT)
    for idx, estimator in enumerate(clf.estimators_):
        effect_name = effects_set[idx]

        # #### PRINTING
        # plt.figure(figsize=(12, 12))
        # plot_tree(estimator, 
        #         feature_names=feature_names, 
        #         label= 'all',
        #         class_names=[f'Class {k}' for k in range(2)], 
        #         filled=True, 
        #         rounded=True,
        #         impurity = True
        #         )
        # say_one_positive_node = ""
        # clf__ = estimator
        # # Ugly paths are ALL the path of the current tree (one tree per effect)
        # ugly_paths = get_rules(clf__, feature_names, None)
        # if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
        #     one_node = True
        #     summ = int(Y_train[effect_name].sum())
        #     if summ > 0:
        #         say_one_positive_node = "CLASS 1"
        #         print("num_action is {}, effect is {}, CLASS 1:".format(str(num_action), str(effect_name)))
        #     else:
        #         say_one_positive_node = "CLASS 0"
        # plt.title(f"Decision Tree for Effect {effect_name}, {say_one_positive_node}", fontsize = 40)
        # file_name = f"DTs_action_{str(num_action)}/decision_tree_effect_{effect_name}.png"
        # plt.savefig(file_name, format="png", dpi=300)  

        # continue

        clf__ = estimator
        ugly_paths = get_rules(clf__, feature_names, None)



        one_node = False
        the_one_node_is_true = False
        add_stuff = False
    
        if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
            one_node = True
            summ = int(Y_train[effect_name].sum())
            if summ > 0:
                ugly_paths[0][0].replace('0', '1')

                the_one_node_is_true = True
        beauty_paths = []

        if not one_node:
            add_stuff = True    
            for pathh in ugly_paths:
                if 'class is: 0' in pathh[-1]:
                    continue
                elif 'class is: 1' in  pathh[-1]: # and len(pathh[:-1]) > 2:
                    tmp = []
                    integer = int(''.join(x for x in str(pathh[-1].split(", ")[1]) if x.isdigit()))
                    had_some_right_branch = False
                    for cond in pathh[:-1]:
                        if ">" in cond:
                            had_some_right_branch = True
                            tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                        # else: # "<=" in cond
                        #     tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
                        #     #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
                        #     #break
                    
                    if had_some_right_branch:
                        tmp.append("#samples: "+str(integer))
                    if len(tmp) > 0:
                        
                        beauty_paths.append(tmp)


            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
            
        elif the_one_node_is_true:
            tmp = []
            # if num_action == 0:
            #     print("WAS HEREEE")
            #     exit()
            # if num_action == 0:
            #     print("effect_nameeffect_nameeffect_name {}".format(effect_name))
            integer = int(''.join(x for x in str(ugly_paths[0][0].split(", ")[1]) if x.isdigit()))
            tmp.append("#samples: "+str(integer))
            beauty_paths.append(tmp)
            add_stuff = True
            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
        #print("=" * 50)  # Separator between treesµ









two_tabs_space  = "         "

##############################################################################
###########################  WRITE THE PDDL   ################################
#######################  and LEARN THE SECOND TREE ###########################
##############################################################################
with open("domainCondBIS-CSP-DT1s_"+str(precondition_only_entailing)+"_"+str(filter_out_condEffects)+".pddl", "w") as f:


    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(16):
        f.write("(z"+str(i)+")\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")






    atoms = []
    for i in range(16):
        atoms.append(f"(z{i})")
    for i in range(16):
        atoms.append(f"(not (z{i}))")


    dico_disjunctions_preconds_dt = {}

    preconds_perEff_perAction_bis = {}

    for num_action in range(0, 22):


        last_current_time = time.time()


        dico_disjunctions_preconds_dt[num_action] = []





        filename = "final_set_action_"+str(num_action)+".csv"

        final_set = np.loadtxt(filename, delimiter=",", dtype=float)




        # Generate the feature names
        features = [f"(z{i})" for i in range(16)] + [f"(not(z{i}))" for i in range(16)]

        # Convert binary array rows to list of feature names where value is 1
        final_set_str = [
            [features[j] for j in range(32) if row[j] == 1]
            for row in final_set
        ]
        
        preconds_perEff_perAction_bis[num_action] = {}



        # if num_action == 1:
        #     print("uuuuuuuuuuuuuuuuuu&&&&&&&&&&&&&&&&TTTTTTTTTTTTTTTTTTT")
        #     print(preconds_perEff_perAction[1])
        #     exit()




        for effect_name, beauty_pathss in preconds_perEff_perAction[num_action].items():





            #beauty_pathss_coppp = beauty_pathss.copy()

            beauty_pathss_coppp = copy.deepcopy(beauty_pathss)

            for counter, b_path in enumerate(beauty_pathss):

                b_path_copy = b_path[:-1].copy()

                for gen_group in final_set_str:
                    
                    # if gen_group is a subset of b_path_copy
                    # ie if gen_group is entailed by b_path_copy THEN 
                    if set(gen_group).issubset(set(b_path_copy)):

                        for el in gen_group:
                            #b_path[:-1].remove(el)
                            if el in beauty_pathss_coppp[counter]:
                                beauty_pathss_coppp[counter].remove(el)

                        # print("THE BEAUTY PATH AFTER")
                        # print(beauty_pathss_coppp[counter])
                        # exit()


            preconds_perEff_perAction_bis[num_action][effect_name] = beauty_pathss_coppp
            

        # if num_action == 1:
        #     print("uuuuuuuuuuuuuuuuuu&&&&&&&&&&&&&&&&11111111111111111")
        #     print(preconds_perEff_perAction_bis[1])
        #     exit()

        if filter_out_condEffects:
            first_argu = preconds_perEff_perAction_bis[num_action]
        else:
            first_argu = preconds_perEff_perAction[num_action]






        merged_dict = factorize_dict(first_argu, num_action)

        # if num_action == 0:
        #     print("uuuuuuuuuuuuuuuuuu&&&&&&&&&&&&&&&&é2222222222222")
        #     print(preconds_perEff_perAction[1])
        #     exit()
        
        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")

        # # f.write("   :precondition ()\n") # IF NOT PRECONDITIONS

        precondition_part = "" 
        intersection_part = ""
        or_clause = ""
        and_str = ""


        # ### CONSTRUCTING THE GROUPS OF PRECONDITION

        # finat_set = np.array([])
        # S = []
        # maxSize_in_S = 0
        # U = []
        # for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():
        #     S.append(dico_vals["preconds"])
        #     if len(dico_vals["preconds"]) > maxSize_in_S:
        #         maxSize_in_S = len(dico_vals["preconds"])
        #     for atom in dico_vals["preconds"]:
        #         U.append(atom)
        # U = list(set(U))




        # # 1 -- taking the intersection of all the preconditions set of the high lvl action
        # I = intersection_of_lists(S)
        
        # # 3-4 --
        # for s in S:
        #     for i in I:
        #         if i in s:
        #             s.remove(i)
        
        
        # if len(I) > 0:
            
        #     and_str += "(AND "

        #     if len(I) == 1:
        #         and_str += I[0]
        #     else:
        #         and_str += "( "
        #         for el in I:
        #             and_str += el + " "
        #         and_str += " )"
        

        # binary_S = np.zeros((len(S), 32), dtype=np.uint8)
        # binary_S = pd.DataFrame(binary_S)
        # binary_S.columns = atoms
        # for idx, row in binary_S.iterrows():
        #     for col in S[idx]:
        #         if col in binary_S.columns:
        #             binary_S.at[idx, col] = 1


        # binary_S = binary_S.to_numpy()



        
        # # 0) if there is an intersection we build an AND clause

        # if len(I) > 0:
        #     for inter in I:
        #         intersection_part += inter + " "


        # indices_U = [i for i in range(len(atoms)) if atoms[i] in U]

        # maxSize_in_S = 7 #6 #7
        # # 

        # final_set = np.empty((0, 32))  #

        # binary_S_copy = binary_S.copy()

        
          
        
        


        # # 6 parmis 32 = 906192
        # combinationsIndices = GenerateCombinationsRange(indices_U, maxSize_in_S)


        # # time_spent =  time.time() - last_current_time
        # # print("TIME SPENT SO FAR 1", str(time_spent))
        # # last_current_time = time.time()


        # n_features = 32 #len(indices_U)

        # matrix_shape = (len(combinationsIndices), n_features)

        # # Shared memory array
        # shared_array = Array('i', matrix_shape[0] * matrix_shape[1], lock=False)
        # combinationsMatrix = np.frombuffer(shared_array, dtype='int32').reshape(matrix_shape)

        # def worker(args):
        #     idx, comb = args
        #     for j in comb:
        #         combinationsMatrix[idx, j] = 1
        #     return idx  # Just to track progress


        # with Pool(processes=64) as pool:
        #     # Wrap the iterable with tqdm for progress bar
        #     for _ in tqdm(pool.imap(worker, enumerate(combinationsIndices)), total=len(combinationsIndices)):
        #         pass  # tqdm handles progress tracking



        # EntailingTargetsPerCombi = np.zeros((combinationsMatrix.shape[0], len(binary_S)))

     


        # thecounter = 0


        # indices_of_targets_modified_previous_iter = [i for i in range(len(binary_S))]
        
        # while True:


        #     print("thecounter is {}".format(str(thecounter)))


        #     coverage = TotalCoverage(final_set, binary_S_copy)


        #     print(f"Current coverage: {coverage}")
        #     if coverage >= len(binary_S_copy):
        #         break

        #     time_spent =  time.time() - last_current_time
        #     print("TIME SPENT SO FAR 2", str(time_spent))
        #     last_current_time = time.time()


            

        #     # candidates, targets
        #     #  binary_S is the low level preconditions sets
        #     nberOfEntailingTargetsPerCombi, EntailingTargetsPerCombi = nberOfEntailingTargets(combinationsMatrix, binary_S, final_set, EntailingTargetsPerCombi, indices_of_targets_modified_previous_iter, fine_grained = True)            

        #     # print("EntailingTargetsPerCombiEntailingTargetsPerCombiEntailingTargetsPerCombi")
        #     # print(EntailingTargetsPerCombi[:5])
        #     # print(combinationsMatrix.shape)
        #     # print(EntailingTargetsPerCombi.shape)

            
        #     # exit()

        #     best_index = np.argmax(nberOfEntailingTargetsPerCombi)



        #     time_spent =  time.time() - last_current_time
        #     print("TIME SPENT SO FAR 3", str(time_spent))
        #     last_current_time = time.time()

        #     best = combinationsMatrix[best_index]

        #     counter_entailments = 0

        #     indices = []

        #     for iii, s in enumerate(binary_S):
        #         if np.array_equal(best & s, best):
        #             counter_entailments += 1
        #             indices.append(iii)


        #     time_spent =  time.time() - last_current_time
        #     print("TIME SPENT SO FAR 4", str(time_spent))
        #     last_current_time = time.time()


        #     checkfordupplicates(final_set)

        #     time_spent =  time.time() - last_current_time
        #     print("TIME SPENT SO FAR 5", str(time_spent))
        #     last_current_time = time.time()

        #     final_set = np.vstack([final_set, best])

        #     # --- l 11-13

        #     #indices_of_targets_modified_previous_iter = 

        #     matrix_best = np.tile(best, (len(binary_S), 1))
        #     entails = np.all((binary_S & matrix_best) == matrix_best, axis=1) # determines which low lvl preconds entail the current best
            
        #     indices_of_targets_modified_previous_iter = np.where(entails)[0]

        #     # print("indices_of_targets_modified_previous_iter")
        #     # print(indices_of_targets_modified_previous_iter)
        #     # exit()
        #     # print("mask_of_targets_modified_previous_itermask_of_targets_modified_previous_iter")
        #     # print(mask_of_targets_modified_previous_iter)
        #     # exit()

        #     binary_S[entails] = binary_S[entails] - matrix_best[entails] # maj the low lvl preconds by removing the entailed atoms (when an entailement occured)
        #     # the latter only concerns a small portion the low lvl preconds sets
        #     #  such that nberOfEntailingTargets(combinationsMatrix, binary_S, final_set, fine_grained = True)  
        #     #      at least, 
        #     thecounter += 1




        # np.savetxt("final_set_action_BIS_"+str(num_action)+".csv", final_set, delimiter=",", fmt='%d')



        # continue

        ### TO DELETE !!!!!!
        # print(final_set.shape) #(40, 32)
        # #exit()
        # nber_singles = 0
        # atom_groups = []
        # for row in final_set:
        #     atom_list = [atoms[i] for i, val in enumerate(row) if val == 1]
        #     atom_groups.append(atom_list)
        #     if len(atom_list) == 1:
        #         print(atom_lizst)
        #         nber_singles += 1

        # print("nber singles is {}".format(str(nber_singles)))
        # # 26
        # exit()

        print(final_set.shape)

        #final_set_bis = [] 

        #final_set_bis = np.array([])
        final_set_bis = np.empty((0, 32))  #

        # looping over all the unique pair
        #   for each test if one ele entails the other

        for ii, ele1 in enumerate(final_set):

            for jj, ele2 in enumerate(final_set[ii+1:]):
                #print(ele2)

                ele1 = ele1.astype(np.uint8)
                ele2 = ele2.astype(np.uint8)

                #### ele1 ou ele2 

                ####  

                # test if ele1 entails ele2
                entail1 = np.all((ele1 & ele2) == ele2, axis=0)

                # test if ele2 entails ele1
                entail2 = np.all((ele2 & ele1) == ele1, axis=0)


                # #### CASE WHERE we add the the group that are ENTAILED (more "general")
                # if entail1 and not entail2:
                #     # entail1
                #     not_in_list = not any(np.array_equal(ele2, x) for x in final_set_bis)
                #     if not_in_list:
                #         #final_set_bis.append(ele2)
                #         final_set_bis = np.vstack([final_set_bis, ele2])

                # elif entail2 and not entail1:
                #     not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                #     if not_in_list:
                #         #final_set_bis.append(ele1)
                #         final_set_bis = np.vstack([final_set_bis, ele1])




                #### CASE WHERE we add the the group that ENTAIL other groups (more specific)
                if entail1 and not entail2:
                    # entail1
                    not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                    if not_in_list:
                        #final_set_bis.append(ele2)
                        final_set_bis = np.vstack([final_set_bis, ele1])

                elif entail2 and not entail1:
                    not_in_list = not any(np.array_equal(ele2, x) for x in final_set_bis)
                    if not_in_list:
                        #final_set_bis.append(ele1)
                        final_set_bis = np.vstack([final_set_bis, ele2])



                # else:
                #     not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                #     if not_in_list:
                #         final_set_bis.append(ele1)
                    
                # # Check if `arr` is NOT in `arr_list`
                

 

        # print(len(final_set))

        # print()

        # print(len(final_set_bis))
        # print(final_set)

        # exit()

        if precondition_only_entailing:
            final_set = final_set_bis

        # final_set gives for each line, a group (conjunction) for the OR clause of the precondition clause


        # time_spent =  time.time() - last_current_time
        # print("TIME SPENT SO FAR 3", str(time_spent))
        # last_current_time = time.time()


        # [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]

        #exit()
        atom_groups = []
        for row in final_set:
            atom_list = [atoms[i] for i, val in enumerate(row) if val == 1]
            atom_groups.append(atom_list)


        or_str = ""

        if len(atom_groups) > 0:

            or_str += "( OR "

            for group in atom_groups:

                group_str = ""

                if len(group) == 1:
                    group_str += " "+ group[0] +" "

                elif len(group) > 1:
                    group_str += " ( AND "

                    for ell in group:
                        group_str += ell + " "

                    group_str += " ) "  
                
                if group_str != "":
                    or_str += group_str

            or_str += " )"


        # and_str
        # or_str

        # if num_action == 0:
        #     print("and_str")
        #     print(len(and_str))
        #     print(and_str)
        #     print(and_str.count(' '))
        #     print(re.sub(r"\s+", "", and_str))
        #     print("or_str")
        #     print(len(or_str))
        #     print(or_str)
        #     print(or_str.count(' '))
        #     print(re.sub(r"\s+", "", or_str))
        #     exit()

        precond_str = ""
        # if re.sub(r"\s+", "", and_str) == "" and re.sub(r"\s+", "", or_str) == "":
        #     precond_str == "()"

        if len(and_str) == 0 and len(or_str) == 0:
            precond_str += "()"

        else:
            if and_str !=  "":
                precond_str += and_str
                precond_str += or_str
                precond_str += ")"

            else:
                precond_str += or_str


        # if num_action == 0:
        #     print("precond_str")
        #     print(precond_str)
        #     #print(precond_str.strip())
        #     exit()

        

        # CASE : precond is an OR of the set of all high lvl preconds 
        f.write("   :precondition ")
        f.write(precond_str)

        f.write("\n")




        f.write("   :effect (and\n")

        # ##### CODE FOR HAVING THE PRECONDS FOR EACH EFFECT
        # for eff_name, preconds_sets in preconds_perEff_perAction[num_action].items():
        #     two_tabs_space  = "         "
        #     str_preconds = ""
        #     if len(preconds_sets) > 0:
        #         for precond_set in preconds_sets:
        #             # if len(precond_set) > 1:
        #             #     print("KKKKKKK")
        #             #     print(precond_set)
        #             #     exit()
        #             tmp_str = two_tabs_space+"(when "
        #             if len(precond_set) == 1:
        #                 tmp_str += precond_set[0]+"\n"
        #             else:
        #                 sub_str = "(and "+" "
        #                 close_preconds = ""
        #                 for precond in precond_set:
        #                     close_preconds += precond + " "
        #                 sub_str += close_preconds
        #                 sub_str += ")\n"

        #                 tmp_str += sub_str

        #             if eff_name.split('_')[0] == 'del':
        #                 eff_name = "(not (z"+eff_name.split('_')[1]+"))"
        #             else:
        #                 eff_name = "(z"+eff_name.split('_')[1]+")"
        #             tmp_str += two_tabs_space+eff_name + "\n"
        #             tmp_str += two_tabs_space+")\n"
        #             str_preconds += tmp_str
        #     f.write(str_preconds)
        # if there is a contradiction... 
        # 


        # if num_action == 1:

        #     print("merged_dict")
        #     print(merged_dict)
        #     # ('add_8',): [[]] WHAT DOES IT FUCKING MEAN ?????

        #     #  
        #     exit()


        ##### CODE FOR HAVING THE PRECONDS FOR GROUP OF EFFECTS

        for eff_group, preconds_sets in merged_dict.items():

            # factorize_dt1
            if factorize_dt1:
                str_preconds = ""
                if (len(preconds_sets) > 0 and len(preconds_sets[0]) > 0) and ( len(eff_group) > 0 and len(eff_group[0]) > 0):

                    # print('"WAS HERE 1 ')


                    two_tabs_space  = "         "
                    tmp_str = two_tabs_space+"(when "
                    
                    if len(preconds_sets) == 1:
                        
                        precond_set = preconds_sets[0]

                        sub_str = "(and "+" "
                        close_preconds = ""
                        for precond in precond_set:
                            close_preconds += precond + " "
                        sub_str += close_preconds
                        sub_str += ")\n"

                        tmp_str += sub_str

                        # print('"WAS HERE2')
                        # exit()
                    else:

                        factorized_ = logic_factorization(preconds_sets)




                        tmp_str += factorized_ + "\n"

                    # print('"WAS HERE 3')
                    # exit()
                    str_before = ''
                    str_after = ''
                    if len(list(eff_group)) > 1:
                        str_before = two_tabs_space + two_tabs_space + "(and \n"
                        str_after = two_tabs_space + two_tabs_space + ")\n"

                    tmp_group_of_effects = ''
                    for eff_name in  list(eff_group):

                        eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                        
                        tmp_group_of_effects += two_tabs_space+ two_tabs_space + eff_name + "\n"

                    tmp_str += str_before + tmp_group_of_effects + str_after
                    tmp_str += two_tabs_space+")\n"
                
                    
                    # print('"WAS HERE 4')
                    # exit()
                    str_preconds += tmp_str #remove_outer_and(tmp_str)


                else: # precond group is an empty list, therefore the effect must always be applied

                    eff_name = list(eff_group)[0]
                        
                    eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                    #print("eff_name {}".format(str(eff_name)))
                    #tmp_group_of_effects += two_tabs_space+eff_name + "\n"
                    str_preconds += two_tabs_space + eff_name + "\n"

                f.write(str_preconds)

            else:
                    
                two_tabs_space  = "         "
                str_preconds = ""

                if (len(preconds_sets) > 0 and len(preconds_sets[0]) > 0) and ( len(eff_group) > 0 and len(eff_group[0]) > 0):

                    for precond_set in preconds_sets:

                        tmp_str = two_tabs_space+"(when "
                        if len(precond_set) == 1:
                            tmp_str += precond_set[0]+"\n"
                        elif len(precond_set) > 1:
                            sub_str = "(and "+" "
                            close_preconds = ""
                            for precond in precond_set:
                                close_preconds += precond + " "
                            sub_str += close_preconds
                            sub_str += ")\n"

                            tmp_str += sub_str


                        # eff_group
                        str_before = ''
                        str_after = ''
                        if len(list(eff_group)) > 1:
                            str_before = two_tabs_space + two_tabs_space + "(and \n"
                            str_after = two_tabs_space + two_tabs_space + ")\n"
                        #tmp_str += str_before
                        tmp_group_of_effects = ''
                        for eff_name in  list(eff_group):

                            eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                            
                            tmp_group_of_effects += two_tabs_space+ two_tabs_space + eff_name + "\n"

                        tmp_str += str_before + tmp_group_of_effects + str_after
                        tmp_str += two_tabs_space+")\n"

                        str_preconds += tmp_str

                else: # precond group is an empty list, therefore the effect must always be applied

                    eff_name = list(eff_group)[0]
                        
                    eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                    #print("eff_name {}".format(str(eff_name)))
                    #tmp_group_of_effects += two_tabs_space+eff_name + "\n"
                    str_preconds += two_tabs_space + eff_name + "\n"


                f.write(str_preconds)



        f.write("   )\n")
        f.write(")\n")

        # if num_action == 2:

        #     exit()
    
    f.write(")\n")



    #     f.write("   :effect (and\n")

    #     for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():


    #         two_tabs_space  = "         "
    #         tmp_str = ""

    #         if len(dico_vals["preconds"]) > 0:

    #             #tmp_str += two_tabs_space+str(lowlvlkey)+" (when "
    #             tmp_str += two_tabs_space+" (when "

    #             tmp_str += "(and "  
    #             ### adding the preconds            
    #             for pre in dico_vals["preconds"]:
    #                 tmp_str += " "+pre

    #             tmp_str += ")\n"  

                

    #         tmp_str += two_tabs_space
    #         # adding the effects
            
    #         if len(dico_vals["effects"]) == 1:
    #             tmp_str += " "+eff
            
    #         elif len(dico_vals["effects"]) > 1: 
    #             tmp_str += "(and "
    #             for eff in dico_vals["effects"]:
    #                 tmp_str += " "+eff
    #             tmp_str += ")"
    #         if len(dico_vals["preconds"]) > 0:
    #             tmp_str += ")\n"   


    #         f.write(tmp_str)

    #     f.write("   )\n")
    #     f.write(")\n")


    #     #exit()

    # f.write(")\n")