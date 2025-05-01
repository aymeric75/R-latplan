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
from sympy import symbols, And, Or, Not, simplify_logic
import itertools
from tqdm import tqdm
import multiprocessing
import pickle
import copy

base_dir = "/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/"


filter_out_dt1 = True 
factorize_dt1 = True

# False False
# False True
# True False
# True True


precondition_only_entailing = True #True # tells if we consider, in the precondition clause, only the sets that entail other sets
filter_out_condEffects = False #True # tells if we filter out from the preconditions of the conditional effects, the atoms that are already
# present in the precondition clause

def load_dataset(path_to_file):
    
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

# 
def _check_entailment(args):
    candidate_idx, candidate, targets = args
    candidate = candidate.astype(bool)
    comparison = (targets & candidate) == candidate
    entailed = np.all(comparison, axis=1) # mask array of size #targets, where True at index i means target i entails candidate
    count = np.sum(entailed) # count the number of True values
    return candidate_idx, count

# returns a mask array of size #candidates, where True at index i means candidate i is equal to ele
def _check_final_set(args):
    ele, candidates = args
    matrix_ele = np.tile(ele, (candidates.shape[0], 1)).astype(bool)
    temp = np.all(matrix_ele == candidates, axis=1)
    return temp

def nberOfEntailingTargets(candidates, targets, final_set_, indices_of_combis_where_entail_is_sup_zero, candidates_sum_of_atoms_, fine_grained = False):

    last_current_time = time.time()

    # dans candidates 

    candidates_sum_of_atoms = np.array([])
    if fine_grained:
        candidates_sum_of_atoms = candidates_sum_of_atoms_ 

    # time_spent =  time.time() - last_current_time
    # print("nberOfEntailingTargets 0 ", str(time_spent))
    # last_current_time = time.time()


    num_candidates = candidates.shape[0]
    candidates = candidates.astype(bool)
    targets = targets.astype(bool)

    true_indices = set()

    # time_spent =  time.time() - last_current_time
    # print("nberOfEntailingTargets 1 ", str(time_spent))
    # last_current_time = time.time()
    

    if len(final_set_) > 0:
        with mp.Pool(processes=(mp.cpu_count() // 2)) as pool:
            equality_results = pool.map(_check_final_set, [(ele, candidates) for ele in final_set_])

        equals = np.any(equality_results, axis=0)
        true_indices = set(np.where(equals)[0])


    # time_spent =  time.time() - last_current_time
    # print("nberOfEntailingTargets 2", str(time_spent))
    # last_current_time = time.time()



    # 

    # Prepare arguments for multiprocessing
    #args = [(i, candidates[i], targets) for i in range(num_candidates) if i not in true_indices]
    args = [(i, candidates[i], targets) for i in indices_of_combis_where_entail_is_sup_zero if i not in true_indices]
    results = []
    with mp.Pool(processes=(mp.cpu_count() // 2)) as pool:
        for result in tqdm(pool.imap(_check_entailment, args), total=len(args), desc="Processing"):
            results.append(result)

    # time_spent =  time.time() - last_current_time
    # print("nberOfEntailingTargets 3", str(time_spent))
    # last_current_time = time.time()


    # Initialize count_per_candidate
    count_per_candidate = np.zeros(num_candidates, dtype=int)

    for idx, count in results:
        if fine_grained:
            count_per_candidate[idx] = count*candidates_sum_of_atoms[idx]
        else:
            count_per_candidate[idx] = count

    return count_per_candidate

    # if np.max(count_per_candidate) <= 0:
    #     return None
    # else:
    #     return np.argmax(count_per_candidate)





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
        entailed_mask = np.all((candidates & target) == candidates, axis=1)
        entailed_candidates = candidates[entailed_mask]

        if len(entailed_candidates) == 0:
            entails_back = False
        else:
            # Disjunction over entailed candidates
            disjunction = np.bitwise_or.reduce(entailed_candidates, axis=0)
            # if entailed_candidates is 
            # [[0, 1, 0],
            # [1, 0, 0],
            # [0, 0, 1]]
            # then disjunction is [1, 1, 1]



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
    


    onehot_encoder = OneHotEncoder(sparse_output=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity
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
                    for cond in pathh[:-1]:
                        if ">" in cond:
                            tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                        # else: # "<=" in cond
                        #     tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
                        #     #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
                        #     #break
                    tmp.append("#samples: "+str(integer))
                    if len(tmp) > 0:
                        
                        beauty_paths.append(tmp)


            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
            
        elif the_one_node_is_true:
            tmp = []

            integer = int(''.join(x for x in str(ugly_paths[0][0].split(", ")[1]) if x.isdigit()))
            tmp.append("#samples: "+str(integer))
            beauty_paths.append(tmp)
            add_stuff = True
            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
        #print("=" * 50)  # Separator between trees


        ####


# par action, par effet,        BIEN


##" SINON j'ai , par action

def entails_str(list1, list2):
    return all(elem in list1 for elem in list2)


def lists_of_lists_differ(list1, list2):
    return list1 != list2


def generate_all_index_combinations(items):
    indices = list(range(len(items)))
    return [
        combo
        for r in range(1, len(items) + 1)
        for combo in itertools.combinations(indices, r)
    ]

def process_row(args):
    row, mask_matrix_pos = args
    mask_matrix_neg = mask_matrix_pos & ~row
    mask_matrix_neg = np.unique(mask_matrix_neg, axis=0)
    repeated_array = np.tile(row, (mask_matrix_neg.shape[0], 1))
    matrixCombiSafe = np.concatenate((repeated_array, mask_matrix_neg), axis=1)
    return matrixCombiSafe




def transform_atoms_list(atoms_list):
    transformed = []
    for sublist in atoms_list:
        parts = []
        for atom in sublist:
            if atom.startswith('(not (z') and atom.endswith('))'):
                # Example: (not (z1)) -> ~z1
                parts.append('~' + atom[6:-2])
            elif atom.startswith('(z') and atom.endswith(')'):
                # Example: (z1) -> z1
                parts.append(atom[1:-1])
            else:
                raise ValueError(f"Unexpected atom format: {atom}")
        combined = '{ ' + ' '.join(parts) + ' }'
        transformed.append(combined)
    return transformed




def fromBinaryMatrixToStrings(sorted_combinationsMatrix_int):

    combinations_list = []
    for row in sorted_combinationsMatrix_int:
        selected_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
        combinations_list.append(selected_atoms)
    #print(combinations_list)

    combinations_list_tranfo = transform_atoms_list(combinations_list)
    # for ele in combinations_list_tranfo:
    #     print(ele)
    
    return combinations_list_tranfo



### 
###     input: 
###             candidates: a set (binary matrix) of sets of atoms (output from STEP1, ie what's in final_list
###
###             targets: what we want to cover
###
###     output: a str like (OR (AND ) (AND )), or the equivalent in an array

def return_dnf(candidates, targets):


    print("candidates")
    print(fromBinaryMatrixToStrings(candidates))

    print("targets")
    print(fromBinaryMatrixToStrings(targets))




    dico_cands_partial_covering = {}

    # 1) count the partial covering of each candidate w.r.t targets
    for i, ca in enumerate(candidates):
        matrix_ca = np.tile(ca, (len(targets), 1))
        coverings = np.all((targets & matrix_ca) == matrix_ca, axis=1)
        indices = set(np.where(coverings)[0])
        # count the number of Trues
        partial_covering = np.sum(coverings)
        dico_cands_partial_covering[i] = {"cover_size": partial_covering, "ids_covered": indices}


    print(dico_cands_partial_covering)
    #dico_cands_partial_covering = dict(sorted(dico_cands_partial_covering.items(), key=lambda item: item[1]['cover_size'], reverse=True))

    for k, v in dico_cands_partial_covering.items():
        print("k: {}, v: {}".format(k, v))



    # 2) Find p_candidates (i.e. the minimal candidates which disjunction "partially" cover the targets)
    targets_ids_covered = set()
    p_candidates = [] # keys 

    dico_cands_partial_covering_copy = copy.deepcopy(dico_cands_partial_covering)

    ##### évidemment faut pas que un ids_covered_by_cand soit déjà "couvert" par un déjà existant

    while True:
        # iii) break when we have covered all the targets
        if len(targets_ids_covered) >= len(targets):
            break
        #   i) search in dico_cands_partial_covering the cand with the highest cover_size and add its key to p_candidates
        max_key = max(dico_cands_partial_covering_copy, key=lambda k: dico_cands_partial_covering_copy[k]["cover_size"])
        ids_covered_by_cand = dico_cands_partial_covering_copy[max_key]["ids_covered"]
        del dico_cands_partial_covering_copy[max_key]

        if not ids_covered_by_cand.issubset(targets_ids_covered):

            p_candidates.append(max_key)
       
            #  ii) update the targets_ids_covered
            targets_ids_covered = targets_ids_covered | ids_covered_by_cand


    print(fromBinaryMatrixToStrings(candidates[p_candidates]))
    exit()

    # 3) for each can_p (partial candidate), find the completion set in order to cover its respective targets
    #
    #       i.e. we want to have (AND can_p_i (OR atoms_to_help_covering_targets_partially_covered_by_can_p_i  ))

    dico_canPs_ToKeep = {}


    for enu, can_p in enumerate(p_candidates):

        # if enu == 0:
        #     continue

        print(candidates[can_p]) # {~z3}

        # 
        ids_targets_tmp = dico_cands_partial_covering[can_p]["ids_covered"] # subset of targets to cover for the can_p
        
        # i) find the targets partially covered by the can_p
        print("ids_targets_tmp") # {0, 2, 3, 4}
        print(ids_targets_tmp)
     

        SS = targets[sorted(ids_targets_tmp)] # binary matrix of ids_targets_tmp


        ###print(fromBinaryMatrixToStrings(SS)) ['{ z1 ~z3 }', '{ z1 z4 ~z3 }', '{ z1 z4 }', '{ z1 z3 z4 }']


        # ii) we remove the atoms of can_p from SS
        matrix_can_p = np.tile(candidates[can_p], (len(SS), 1)) 
        entails = np.all((SS & matrix_can_p) == matrix_can_p, axis=1)
        SS[entails] = SS[entails] - matrix_can_p[entails]
     
        print(fromBinaryMatrixToStrings(SS)) # ['{ z1 ~z3 }', '{ z1 z4 ~z3 }', '{ z1 z4 }', '{ z1 z3 z4 }']


        # iii) for each sub candidate (i.e. all candidates Except can_p) gives its cover_size and  the ids covered W.R.T SS
        dico_cand_TMP_partial_covering = {} 

        mask = np.ones(candidates.shape, dtype=bool)
        mask[can_p] = False

        all_candidates_minus_cand_p = candidates * mask
        all_candidates_minus_cand_p = all_candidates_minus_cand_p[~np.all(all_candidates_minus_cand_p == 0, axis=1)]
        
        # we also remove all the atoms of can_p from the other candidates
        all_candidates_minus_cand_p = np.clip(all_candidates_minus_cand_p - candidates[can_p], 0, 1)
   



        # iv) count the partial covering of each candidate w.r.t SS
        for i, ca in enumerate(all_candidates_minus_cand_p):
            matrix_ca = np.tile(ca, (len(SS), 1))
            coverings = np.all((SS & matrix_ca) == matrix_ca, axis=1)
            indices = np.where(coverings)[0]
            # count the number of Trues
            partial_covering = np.sum(coverings)
            dico_cand_TMP_partial_covering[i] = {"cover_size": partial_covering, "ids_covered": indices}



        # print("all_candidates_minus_cand_p")
        # print(fromBinaryMatrixToStrings(all_candidates_minus_cand_p)) #
        # print("dico_cand_TMP_partial_covering")
        # print(dico_cand_TMP_partial_covering)
        # print("SS")
        # print(fromBinaryMatrixToStrings(SS))
        # exit()
        # v) perform a Greedy Search in order to find for can_p its completion sets so that 
        #       (AND can_p (OR completion_sets)) "covers" SS

        dico_cand_TMP_partial_covering_copy = copy.deepcopy(dico_cand_TMP_partial_covering)

        to_keep = [] # 


        # putain mais quelle horreur

        #####   BON, ya qu'une solution, trouver des groupes de low level actions

        #####  such that there is no contradiction in there low level preconditions !!!!


        ###############    AT least if there is not contradictions, we can proceed

        ####
        ####
        ####    ET ensuite, tu peux sûrement utoilisé
        ####
        ####
        ####
        ####


        print()
        print()
        print('STAAAAAAAARTINGGGGGGGGGG')

        while True:

            # We stop 

            # 
            if np.all(SS == 0):
                break
            
            #print(dico_cand_TMP_partial_covering_copy)
            max_key = max(dico_cand_TMP_partial_covering_copy, key=lambda k: dico_cand_TMP_partial_covering_copy[k]["cover_size"])
            del dico_cand_TMP_partial_covering_copy[max_key]

            thebest = all_candidates_minus_cand_p[max_key]
            print("thebest")
            print(thebest)
            # print(all_candidates_minus_cand_p)
  

            to_keep.append(list(thebest))

            to_keep_OR = np.bitwise_or.reduce(to_keep)


            SS = np.clip(SS - to_keep_OR, 0, 1)
        
            # # update SS (remove from any candidate in it the atoms that <=> best
            # matrix_best = np.tile(thebest, (len(SS), 1))
            # entails = np.all((SS & matrix_best) == matrix_best, axis=1)
            # SS[entails] = SS[entails] - matrix_best[entails]
            print("SS is")
            print(SS)


        dico_canPs_ToKeep[can_p] = to_keep
  

    

    print("dico_canPs_ToKeep")
    print(dico_canPs_ToKeep)
    for k, v in dico_canPs_ToKeep.items():
        print(candidates[k])
        print(fromBinaryMatrixToStrings(v))
    exit()

    # z1
    # ['{ z4 }', '{ ~z3 }', '{ z3 }']

    # ~z3
    # ['{ z1 }', '{ z1 z4 }', '{ ~z1 ~z4 }']


    # NOW we have: dico_canPs_ToKeep which is key / value where key is a partial candidate and value is the <=> sets to complete the coverage for its targets

    # 4) build i) a STR representing the DIS of CONJS, using each can_p and its associated to_keep
    #
    #                       if only one can_p THEN no need to start the str with an OR, just send (AND can_p (OR to_keep groups))
    #                       if multiple can_p, THEN return (OR (AND can_p1 (OR to_keep1)) (AND can_p2 (OR to_keep2)) )

    return dico_canPs_ToKeep

print(dico_highlvlid_lowlvlactions[0][19])

two_tabs_space  = "         "

name_pddl_file = "domainCondBIS-CSP-R_Latplan_effs_"+str(precondition_only_entailing)+"_GEN_TEST_"+str(filter_out_condEffects)







##############################################################################
###########################  WRITE THE PDDL   ################################
#######################  and LEARN THE SECOND TREE ###########################
##############################################################################
with open(name_pddl_file+".pddl", "w") as f:


    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(16):
        f.write("(z"+str(i)+" )\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")


    atoms = [] # = U
    for i in range(16):
        atoms.append(f"(z{i})")
    for i in range(16):
        atoms.append(f"(not (z{i}))")






    # ##############   MAKE THE SAFE MATRIX OF ALL COMBINATIONS ALLOWED ######################
    # ######## ( AllSafeCombinations )

    # index_tuples = generate_all_index_combinations([ind for ind in range(16)])

    # mask_matrix_pos = np.zeros((len(index_tuples), 16), dtype=bool)

    # # Fill in True values using advanced indexing
    # for i, indices in enumerate(index_tuples):
    #     mask_matrix_pos[i, indices] = True



    # # Create the new row (1 row of 16 False values)
    # new_row = np.zeros((1, mask_matrix_pos.shape[1]), dtype=bool)
    # # Append it to the original matrix
    # mask_matrix_pos = np.vstack([mask_matrix_pos, new_row])


    # all_matrices = []

    # len_mask_matrix_pos = len(mask_matrix_pos)


    # all_args = [(row, mask_matrix_pos) for row in mask_matrix_pos]
    # AllSafeCombinations = None

    # with mp.Pool(mp.cpu_count()) as pool:
    #     for matrixCombiSafe in tqdm(pool.imap(process_row, all_args), total=len(all_args), desc="Processing"):
    #         if AllSafeCombinations is None:
    #             AllSafeCombinations = matrixCombiSafe
    #         else:
    #             AllSafeCombinations = np.concatenate((AllSafeCombinations, matrixCombiSafe), axis=0)


    # # Save to pickle
    # with open(name_pddl_file+"_AllSafeCombinations.pkl", "wb") as f:
    #     pickle.dump(AllSafeCombinations, f)


    with open("AllSafeCombinations.pkl", "rb") as f:
        AllSafeCombinations = pickle.load(f)



    # MAINTENANT cherche [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]

    ##### UNIT TEST 1: check if #rows where 1st 16th = 0 (except first item) = 2**15 (32768)
    # mask = (AllSafeCombinations[:, 0] == 1) & np.all(AllSafeCombinations[:, 1:16] == 0, axis=1)
    # filtered_rows = AllSafeCombinations[mask]
    # print(len(filtered_rows)) # 32768

    # 42 981 185
    # ##############   END MAKE THE SAFE MATRIX OF ALL COMBINATIONS ALLOWED ######################


    for num_action in range(0, 22):


        if num_action != 8:
            continue




        precondition_part = "" 
        intersection_part = ""
        or_clause = ""
        and_str = ""

        last_current_time = time.time()

        # if num_action in [0,2,4,5,6,7,8,11,12,14,15,16,17,20,21]:
        #     continue

        ### CONSTRUCTING THE GROUPS OF PRECONDITIONs

        finat_set = np.array([])
        S = []
        U_pre = []
        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():
            S.append(dico_vals["preconds"])
            for atom in dico_vals["preconds"]:
                U_pre.append(atom)
        U_pre = list(set(U_pre))


 
        S = [
            ['(z1)', '(z2)', '(not (z3))'],
            ['(not (z1))', '(z2)', '(not (z3))', '(not (z4))'],
            ['(z1)', '(z2)', '(not (z3))', '(z4)'],
            ['(z1)', '(z2)', '(z4)'],
            ['(z1)', '(z2)', '(z3)', '(z4)'],

        ]
        U_pre = ['(z1)', '(z2)', '(z3)', '(z4)', '(not (z1))', '(not (z3))', '(not (z4))']




        ##### UNIT TEST 2: check the precondition_and part
        #S = [['(z9)', '(z10)', '(not (z11))', '(not (z12444))'], ['(z9)', '(z10)', '(not (z11))', '(not (z12))', '(not (z13))', '(z15)']]
        # then uncomment the print("and_str") below

        # 1 -- taking the intersection of all the preconditions set of the high lvl action
        I = intersection_of_lists(S)

        if len(I) > 0:
            
            and_str += "(AND "

            if len(I) == 1:
                and_str += I[0]
            else:
                and_str += "( "
                for el in I:
                    and_str += el + " "
                and_str += " )"
        
        print(and_str)


        ##### UNIT TEST 3: check the precondition_and part
        # first uncomment the fake S introduced in UNIT TEST 2 and the print(S) below
        # 3-4 --
        for s in S:
            for i in I:
                if i in s:
                    s.remove(i)
        # l. 6
        for u in U_pre:
            for i in I:
                if i in U_pre:
                    U_pre.remove(i)


        binary_S = np.zeros((len(S), 32), dtype=np.uint8)
        binary_S = pd.DataFrame(binary_S)
        binary_S.columns = atoms
        for idx, row in binary_S.iterrows():
            for col in S[idx]:
                if col in binary_S.columns:
                    binary_S.at[idx, col] = 1

        binary_S = binary_S.to_numpy()

        #### TO REMOVE ??????
        if len(I) > 0:
            for inter in I:
                intersection_part += inter + " "
                

   
        indices_U = [i for i in range(len(atoms)) if atoms[i] in U_pre]


        final_set = np.empty((0, 32))  #

        binary_S_copy = binary_S.copy()



        # UNIT TEST 4: combinationsMatrix must NOT have any "variable" that is in indices_not_in_U_pre
        # uncomment the line just below, then the 'assert'
        #indices_not_in_U_pre = [0] # we fakely remove z0 from U_pre

        indices_not_in_U_pre = []
        for indexx, atom in enumerate(atoms):
            if atom not in U_pre:
                indices_not_in_U_pre.append(indexx)

        mask_only_in_U_pre = np.all(AllSafeCombinations[:, indices_not_in_U_pre] == 0, axis=1)
        combinationsMatrix = AllSafeCombinations[mask_only_in_U_pre]

        # REMOVE the row with only False
        combinationsMatrix = combinationsMatrix[~np.all(combinationsMatrix == False, axis=1)]

        combinationsMatrix_int = combinationsMatrix.astype(int)
        # Step 1: Count the number of ones in each row
        num_ones = np.sum(combinationsMatrix_int, axis=1)
        first_one_index = np.where(combinationsMatrix_int == 1, np.arange(combinationsMatrix_int.shape[1]), combinationsMatrix_int.shape[1])
        first_one_index = np.min(first_one_index, axis=1)
        sort_keys = np.lexsort((first_one_index, num_ones))
        sorted_combinationsMatrix_int = combinationsMatrix_int[sort_keys]
        combinations_list = []
        for row in sorted_combinationsMatrix_int:
            selected_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
            combinations_list.append(selected_atoms)
        #print(combinations_list)

        combinations_list_tranfo = transform_atoms_list(combinations_list)
        for ele in combinations_list_tranfo:
            print(ele)


        combinationsMatrix = sorted_combinationsMatrix_int.astype(bool)

        # assert combinationsMatrix[:,0].sum() == 0
        # print("passed UNIT TEST 4")

        # init of indices of the combinations where encompassement is > 0 to ALL indices (actually it is unknown, but we do it for practicability)
        indices_of_combis_where_entail_is_sup_zero = np.array([inn for inn in range(len(combinationsMatrix))])

        thecounter = 0

        candidates_sum_of_atoms = np.sum(combinationsMatrix, axis = 1)


        # while True:

        #     print("thecounter is {}".format(str(thecounter)))

        #     coverage = TotalCoverage(final_set, binary_S_copy)

        #     print("action is {}".format(num_action))

        #     print(f"Current coverage: {coverage}")
            

        #     if coverage >= len(binary_S_copy):
        #         break

        #     # time_spent =  time.time() - last_current_time
        #     # print("TIME SPENT SO FAR 2", str(time_spent))
        #     # last_current_time = time.time()


        #     # UNIT TEST 5: 
        #     # create a fake binary_S like
        #     # [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #     # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
        #     # binary_S = np.zeros((2, 32))
        #     # binary_S[0,:2] = 1
        #     # binary_S[1,0] = 1
        #     # binary_S[1,-1] = 1

        #     ### for any combi for which entailment was 0 on previous iteration, NO NEED TO REDO THE COMPUTATION

        #     nberOfEntailingTargetsPerCombi = nberOfEntailingTargets(combinationsMatrix, binary_S, final_set, indices_of_combis_where_entail_is_sup_zero, candidates_sum_of_atoms, fine_grained = True)            
            
        #     indices_of_combis_where_entail_is_sup_zero = np.where(nberOfEntailingTargetsPerCombi > 0)[0]


        #     # # Save to pickle
        #     # with open("nberOfEntailingTargetsPerCombi_8.pkl", "wb") as f:
        #     #     pickle.dump(nberOfEntailingTargetsPerCombi, f)
        #     # # with open("nberOfEntailingTargetsPerCombi_8.pkl", "rb") as f:
        #     # #     nberOfEntailingTargetsPerCombi = pickle.load(f)


        #     best_index = np.argmax(nberOfEntailingTargetsPerCombi)

        #     # time_spent =  time.time() - last_current_time
        #     # print("TIME SPENT SO FAR 3", str(time_spent))
        #     # last_current_time = time.time()

        #     best = combinationsMatrix[best_index]

        #     # time_spent =  time.time() - last_current_time
        #     # print("TIME SPENT SO FAR 4", str(time_spent))
        #     # last_current_time = time.time()

        #     checkfordupplicates(final_set)
        #     # time_spent =  time.time() - last_current_time
        #     # print("TIME SPENT SO FAR 5", str(time_spent))
        #     # last_current_time = time.time()

        #     final_set = np.vstack([final_set, best])

        #     # --- l 11-13
        #     # UNIT TEST 6: remove best from S (binary_S)        DONE (work)
        #     # fake binary_S and fake best
        #     # binary_S = np.zeros((2, 32))
        #     # binary_S[0,:2] = 1
        #     # binary_S[1,0] = 1
        #     # binary_S[1,-1] = 1
        #     # best = np.zeros((2, 32))
        #     # best[0] = 1

        #     matrix_best = np.tile(best, (len(binary_S), 1))
        #     entails = np.all((binary_S & matrix_best) == matrix_best, axis=1)
        #     binary_S[entails] = binary_S[entails] - matrix_best[entails]

        #     thecounter += 1



        # combinations_list = []
        # for row in final_set:
        #     selected_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
        #     combinations_list.append(selected_atoms)
        # #print(combinations_list)

        # combinations_list_tranfo = transform_atoms_list(combinations_list)
        # for ele in combinations_list_tranfo:
        #     print(ele)


        ##### LOADING THE GENERAL PRECONDITIONS SET (in final_set like array)
        ##### AND OPTIONALLY FILTERING THEM OUT




        #np.savetxt("FORTEST_final_set_action_"+str(num_action)+".csv", final_set, delimiter=",", fmt='%d')
        final_set = np.loadtxt("FORTEST_final_set_action_"+str(num_action)+".csv", delimiter=",", dtype=int)
        


        # print("final_set")
        # print(final_set)

        # print("binary_S_copy")
        # print(binary_S_copy)
        # exit()

        return_dnf(final_set, binary_S_copy)

        #   tu as AND et le "OR"
        #
        #
        #           et au lieu d'avoir un OR de 5 trucs tu préfère avoir un 
        #
        #
        #
        #                   (OR (AND ) (AND ))
        #
        #


        exit()
        
        
        
        continue




        filename = "final_set_action_"+str(num_action)+".csv"

        final_set = np.loadtxt(filename, delimiter=",", dtype=float)


        print(final_set.shape)

        #final_set_bis = [] 

        ### UNIT TEST 7 (not done here, but on a separate file): final_set_bis should contain only the groups that are entailed by other groups


        #final_set_bis = np.array([])
        final_set_bis = np.empty((0, 32))  #

        # looping over all the unique pair
        #   for each test if one ele entails the other

        for ii, ele1 in enumerate(final_set):

            for jj, ele2 in enumerate(final_set[ii+1:]):

                ele1 = ele1.astype(np.uint8)
                ele2 = ele2.astype(np.uint8)

                # test if ele1 entails ele2
                entail1 = np.all((ele1 & ele2) == ele2, axis=0)

                # test if ele2 entails ele1
                entail2 = np.all((ele2 & ele1) == ele1, axis=0)

                #### CASE WHERE we add the the group that are ENTAILED (more "general")
                if entail1 and not entail2:
                    # entail1
                    not_in_list = not any(np.array_equal(ele2, x) for x in final_set_bis)
                    if not_in_list:
                        #final_set_bis.append(ele2)
                        final_set_bis = np.vstack([final_set_bis, ele2])

                elif entail2 and not entail1:
                    not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                    if not_in_list:
                        #final_set_bis.append(ele1)
                        final_set_bis = np.vstack([final_set_bis, ele1])

                # #### CASE WHERE we add the the group that ENTAIL other groups (more specific)
                # if entail1 and not entail2:
                #     # entail1
                #     not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                #     if not_in_list:
                #         #final_set_bis.append(ele2)
                #         final_set_bis = np.vstack([final_set_bis, ele1])

                # elif entail2 and not entail1:
                #     not_in_list = not any(np.array_equal(ele2, x) for x in final_set_bis)
                #     if not_in_list:
                #         #final_set_bis.append(ele1)
                #         final_set_bis = np.vstack([final_set_bis, ele2])

                # else:
                #     not_in_list = not any(np.array_equal(ele1, x) for x in final_set_bis)
                #     if not_in_list:
                #         final_set_bis.append(ele1)
                    
                # # Check if `arr` is NOT in `arr_list`
                

        # print(len(final_set_bis))
        # print(final_set_bis)

        if precondition_only_entailing:
            final_set = final_set_bis


    
        ## UNIT TEST 8, check the General Precondition clause
        ## check the and_str
        #### fake I and redo the and_str (it's just to check)
        # I = ["(z0)", "(not (z1))"]
        # and_str = ""
        # if len(I) > 0:
        #     and_str += "(AND "
        #     if len(I) == 1:
        #         and_str += I[0]
        #     else:
        #         and_str += "( "
        #         for el in I:
        #             and_str += el + " "
        #         and_str += " )"
        # print(and_str)
        ## FOR THE or_str check unit_test_or.py file            OK (works)

        ##
        ##
        ## check the final clause


        # if filter_out_condEffects:
        #     preconds_perEff_perAction = preconds_perEff_perAction_bis

        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")

        # f.write("   :precondition ()\n") # IF NOT PRECONDITIONS



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

        precond_str = ""
        if and_str == "" and or_str == "":
            precond_str == "()"
        
        else:

            if and_str !=  "":
                precond_str += and_str
                precond_str += or_str
                precond_str += ")"

            else:
                precond_str += or_str


        # CASE : precond is an OR of the set of all high lvl preconds 
        f.write("   :precondition ")
        f.write(precond_str)
        f.write("\n")

        if filter_out_condEffects:
                
            ## UNIT TEST 9: check that the dico for the current low lvl actions dico_highlvlid_lowlvlactions[num_action]
            ## is pruned out of the entailed groups of the General Precondition

            # for row in final_set:
            for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():
                atoms_to_remove = []
                # REMOVING THE AND parts    DONE (works)
                for at in I:

                    if at not in atoms_to_remove and at in dico_highlvlid_lowlvlactions[num_action][lowlvlkey]["preconds"]:
                        #print("at is {}".format(at))
                        atoms_to_remove.append(at)

                # REMOVING THE OR parts
                # atom_groups are the groups of atoms present in the "general" precondition clause
                for gr in atom_groups:
                    # if one "general" group is entailed by the low-lvl precondition set THEN
                    if entails_str(dico_vals["preconds"], gr):
                        # we remove from this low level precondition set, all the corresponding atoms
                        for ele in gr:
                            if ele not in atoms_to_remove:
                                atoms_to_remove.append(ele)  

                for elee in atoms_to_remove:
                    dico_highlvlid_lowlvlactions[num_action][lowlvlkey]["preconds"].remove(elee)


        dico_highlvlid_lowlvlactions_num_action = {}


        already_there = []

        # REMOVE ANY DUPPLICATE IN THE EFFECTS
        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[num_action].items():
            # 
            identifier = str(dico_vals["effects"]) + "_" + str(dico_vals["preconds"])

            if identifier not in already_there:
                dico_highlvlid_lowlvlactions_num_action[lowlvlkey] = dico_vals
                already_there.append(identifier)

        dico_highlvlid_lowlvlactions_num_action_items_ = dico_highlvlid_lowlvlactions_num_action.items()


        
        f.write("   :effect (and\n")

        for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions_num_action_items_:


            # print("DICO VALS EFFECTS")
            # print(dico_vals["effects"])
            # exit()
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